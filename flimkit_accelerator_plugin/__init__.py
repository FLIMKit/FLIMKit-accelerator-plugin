import ctypes
import os
import sys

import numpy as np

from flimkit.GPU import PluginBackend
from flimkit.plugins import fit_backend

FLIMKIT_PLUGIN_API = 1
ABI_VERSION = 2
HERE = os.path.dirname(os.path.abspath(__file__))

def library_name():
    if sys.platform == 'win32':
        return 'flimkit_accelerator.dll'
    if sys.platform == 'darwin':
        return 'libflimkit_accelerator.dylib'
    return 'libflimkit_accelerator.so'

def load_library(path=None):
    if path is None:
        path = HERE + os.sep + library_name()
    if os.path.isfile(path) == False:
        return None
    lib = ctypes.CDLL(path)
    if lib.flimkit_c_abi_version() != ABI_VERSION:
        raise OSError(path + ' was built for another version of this add-on, run build.py again')
    f64 = np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS')
    f32 = np.ctypeslib.ndpointer(np.float32, flags='C_CONTIGUOUS')
    u8 = np.ctypeslib.ndpointer(np.uint8, flags='C_CONTIGUOUS')
    i64 = ctypes.c_int64
    lib.flimkit_c_fixed_tau.argtypes = [f32, f32, i64, i64, f64, f64, i64, f64, f64, f64, u8]
    lib.flimkit_c_fixed_tau.restype = None
    return lib

# returns None outside the plain case, which hands the fit back to FLIMKit
class CFixedTau(PluginBackend):
    def __init__(self, lib):
        self._lib = lib

    def __repr__(self):
        return 'CFixedTau(threads=' + str(self._lib.flimkit_c_threads()) + ')'

    def batch_fixed_tau(self, stack, A, taus_fixed, min_photons, correct_pileup,
                        n_sync_px, progress_callback=None, tvb_profile=None,
                        fit_tvb=False, fit_idx=None, **kwargs):
        if fit_idx is not None or (fit_tvb == True and tvb_profile is not None) or \
                (correct_pileup == True and n_sync_px > 0):
            return None
        ny, nx, n_bins = stack.shape
        n_exp = A.shape[1]
        raw = stack.reshape(ny * nx, n_bins)
        valid_idx = np.flatnonzero(raw.sum(axis=1) >= min_photons)
        maps = self._init_maps(ny, nx, n_exp, intensity=stack.sum(axis=2),
                               taus_fixed_ns=taus_fixed * 1e9, free_tau=False)
        if valid_idx.size == 0:
            return maps
        decay = np.ascontiguousarray(raw[valid_idx], dtype=np.float32)
        bg = np.ascontiguousarray(
            self._estimate_bg_batch(decay, np.ones(decay.shape[0], dtype=bool)),
            dtype=np.float32)
        a = np.ascontiguousarray(A, dtype=np.float64)
        a_pinv = np.ascontiguousarray(np.linalg.pinv(a), dtype=np.float64)
        n = decay.shape[0]
        amps = np.empty((n, n_exp))
        numerator = np.empty(n)
        expected = np.empty(n)
        valid = np.empty(n, dtype=np.uint8)
        self._lib.flimkit_c_fixed_tau(decay, bg, n, n_bins, a_pinv, a, n_exp,
                                      amps, numerator, expected, valid)
        self._scatter_fixed_tau(maps, valid_idx=valid_idx, amps=amps, bg=bg,
                                decay_valid=decay, A=A, taus_ns=taus_fixed * 1e9,
                                ny=ny, nx=nx,
                                chi2_parts=(numerator, expected, valid.astype(bool)))
        if progress_callback is not None:
            progress_callback(valid_idx.size, valid_idx.size)
        return maps

@fit_backend('accelerator', 'FLIMKit accelerator: C fixed-tau (ctypes)', priority=100)
def make_backend():
    lib = load_library()
    return None if lib is None else CFixedTau(lib)
