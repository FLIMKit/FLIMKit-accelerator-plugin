import ctypes
import os
import sys

import numpy as np

from flimkit.FLIM.fit_tools import TAU_FIT_UNIT_S, calibrated_from_terms
from flimkit.GPU import PluginBackend
from flimkit.plugins import fit_backend

__version__ = '0.2.0'
FLIMKIT_PLUGIN_API = 1
ABI_VERSION = 4
HERE = os.path.dirname(os.path.abspath(__file__))

def library_name():
    if sys.platform == 'win32':
        return 'flimkit_accelerator.dll'
    if sys.platform == 'darwin':
        return 'libflimkit_accelerator.dylib'
    return 'libflimkit_accelerator.so'

def cuda_library_name():
    if sys.platform == 'win32':
        return 'flimkit_accelerator_cuda.dll'
    return 'libflimkit_accelerator_cuda.so'

def load_cuda_library(path=None):
    if path is None:
        path = HERE + os.sep + cuda_library_name()
    if os.path.isfile(path) == False:
        return None
    try:
        lib = ctypes.CDLL(path)
        if lib.flimkit_cu_abi_version() != ABI_VERSION or lib.flimkit_cu_available() != 1:
            return None
    except OSError:
        return None
    f64 = np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS')
    u8 = np.ctypeslib.ndpointer(np.uint8, flags='C_CONTIGUOUS')
    i64p = np.ctypeslib.ndpointer(np.int64, flags='C_CONTIGUOUS')
    i32p = np.ctypeslib.ndpointer(np.int32, flags='C_CONTIGUOUS')
    i64 = ctypes.c_int64
    f = ctypes.c_double
    lib.flimkit_cu_free_tau.argtypes = [f64, f64, f64, i64, i64, i64, f64, i64p, i64, f,
                                        f64, f64, f64, i64, f, f64, f64, i32p, f64, f64, u8]
    lib.flimkit_cu_free_tau.restype = ctypes.c_int
    f32 = np.ctypeslib.ndpointer(np.float32, flags='C_CONTIGUOUS')
    lib.flimkit_cu_fixed_tau_poisson.argtypes = [f32, i64, i64, f64, i64, i64, f,
                                                 f64, f64, f64, f64, u8]
    lib.flimkit_cu_fixed_tau_poisson.restype = ctypes.c_int
    return lib

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
    i64p = np.ctypeslib.ndpointer(np.int64, flags='C_CONTIGUOUS')
    i32p = np.ctypeslib.ndpointer(np.int32, flags='C_CONTIGUOUS')
    i64 = ctypes.c_int64
    f = ctypes.c_double
    lib.flimkit_c_fixed_tau.argtypes = [f32, f32, i64, i64, f64, f64, i64, f64, f64, f64, u8]
    lib.flimkit_c_fixed_tau.restype = None
    lib.flimkit_c_fixed_tau_poisson.argtypes = [f32, i64, i64, f64, i64, i64, f,
                                                f64, f64, f64, f64, u8]
    lib.flimkit_c_fixed_tau_poisson.restype = ctypes.c_int
    lib.flimkit_c_free_tau.argtypes = [f64, f64, f64, i64, i64, i64, f64, i64p, i64, f,
                                       f64, f64, f64, i64, f, f64, f64, i32p, f64, f64, u8]
    lib.flimkit_c_free_tau.restype = ctypes.c_int
    return lib

MAX_EXP = 3


def _fixed_tau_python(decay, bg, A, fit_idx=None):
    decay = np.ascontiguousarray(decay, dtype=np.float32)
    bg = np.ascontiguousarray(bg, dtype=np.float32)
    A = np.ascontiguousarray(A, dtype=np.float64)
    if fit_idx is None:
        idx = np.arange(A.shape[0], dtype=np.int64)
    else:
        idx = np.asarray(fit_idx, dtype=np.int64)
    A_fit = np.ascontiguousarray(A[idx], dtype=np.float64)
    n_pix = decay.shape[0]
    n_bins = A_fit.shape[0]
    n_exp = A_fit.shape[1]
    a_pinv = np.ascontiguousarray(np.linalg.pinv(A_fit), dtype=np.float64)
    amps = np.empty((n_pix, n_exp), dtype=np.float64)
    numerator = np.empty(n_pix, dtype=np.float64)
    expected = np.empty(n_pix, dtype=np.float64)
    valid = np.empty(n_pix, dtype=np.uint8)
    for p in range(n_pix):
        d = decay[p]
        b = float(bg[p])
        amp = np.empty(n_exp, dtype=np.float64)
        for e in range(n_exp):
            row = a_pinv[e]
            s = 0.0
            for k in range(n_bins):
                c = float(d[k]) - b
                s += float(row[k]) * (c if c > 0.0 else 0.0)
            amp[e] = s if s > 0.0 else 0.0
        num = 0.0
        exp_sum = 0.0
        ok = 1
        for k in range(n_bins):
            m = b
            for e in range(n_exp):
                m += float(amp[e]) * float(A_fit[k, e])
            dk = float(d[k])
            if (not np.isfinite(dk)) or dk < 0.0 or (not np.isfinite(m)) or m < 0.0:
                ok = 0
            r = dk - m
            num += r * r / (m if m > 1.0 else 1.0)
            exp_sum += m if m < 1.0 else 1.0
        amps[p] = amp
        numerator[p] = num
        expected[p] = exp_sum
        valid[p] = ok
    return amps, numerator, expected, valid


class CKernels(PluginBackend):
    estimators = ('nnls', 'poisson')

    def __init__(self, lib, cuda=None):
        self._lib = lib
        self._cuda = cuda

    def __repr__(self):
        where = 'cuda+cpu' if self._cuda is not None else 'cpu'
        return ('CKernels(' + __version__ + ', ' + where + ', threads='
                + str(self._lib.flimkit_c_threads()) + ')')

    def batch_fixed_tau(self, stack, A, taus_fixed, min_photons, correct_pileup,
                        n_sync_px, progress_callback=None, tvb_profile=None,
                        fit_tvb=False, fit_idx=None, estimator='nnls', **kwargs):
        if (fit_tvb == True and tvb_profile is not None) or \
                (correct_pileup == True and n_sync_px > 0):
            return None
        if estimator not in self.estimators:
            return None
        if estimator == 'poisson' and A.shape[1] > MAX_EXP:
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
        if fit_idx is not None:
            idx = np.asarray(fit_idx, dtype=np.int64)
            if idx.size == 0:
                return maps
            decay = np.ascontiguousarray(decay[:, idx], dtype=np.float32)
            A_used = np.ascontiguousarray(A[idx], dtype=np.float64)
        else:
            A_used = np.ascontiguousarray(A, dtype=np.float64)
        if estimator == 'poisson':
            return self._fixed_tau_poisson(maps, valid_idx, decay, A_used, taus_fixed,
                                           ny, nx, progress_callback, kwargs)
        bg = np.ascontiguousarray(
            self._estimate_bg_batch(decay, np.ones(decay.shape[0], dtype=bool)),
            dtype=np.float32)
        a = np.ascontiguousarray(A_used, dtype=np.float64)
        a_pinv = np.ascontiguousarray(np.linalg.pinv(a), dtype=np.float64)
        n = decay.shape[0]
        amps = np.empty((n, n_exp))
        numerator = np.empty(n)
        expected = np.empty(n)
        valid = np.empty(n, dtype=np.uint8)
        self._lib.flimkit_c_fixed_tau(decay, bg, n, decay.shape[1], a_pinv, a, n_exp,
                                      amps, numerator, expected, valid)
        self._scatter_fixed_tau(maps, valid_idx=valid_idx, amps=amps, bg=bg,
                                decay_valid=decay, A=A_used, taus_ns=taus_fixed * 1e9,
                                ny=ny, nx=nx,
                                chi2_parts=(numerator, expected, valid.astype(bool)))
        if progress_callback is not None:
            progress_callback(valid_idx.size, valid_idx.size)
        return maps

    def _fixed_tau_poisson(self, maps, valid_idx, decay, A_used, taus_fixed, ny, nx,
                           progress_callback, options):
        n = decay.shape[0]
        n_bins = decay.shape[1]
        n_exp = A_used.shape[1]
        a = np.ascontiguousarray(A_used, dtype=np.float64)
        iters = int(options.get('poisson_iters', 1500))
        tol = float(options.get('poisson_tol', 1e-7))
        amps = np.empty((n, n_exp))
        offset = np.empty(n)
        numerator = np.empty(n)
        expected = np.empty(n)
        valid = np.empty(n, dtype=np.uint8)
        rc = -1
        if self._cuda is not None:
            rc = self._cuda.flimkit_cu_fixed_tau_poisson(
                decay, n, n_bins, a, n_exp, iters, tol,
                amps, offset, numerator, expected, valid)
            if rc != 0:
                print('  [per-pixel] add-on CUDA Poisson fit failed (' + str(rc)
                      + '), using the C kernel')
        if rc != 0:
            rc = self._lib.flimkit_c_fixed_tau_poisson(
                decay, n, n_bins, a, n_exp, iters, tol,
                amps, offset, numerator, expected, valid)
        if rc != 0:
            return None
        self._scatter_fixed_tau(maps, valid_idx=valid_idx, amps=amps, bg=offset,
                                decay_valid=decay, A=A_used, taus_ns=taus_fixed * 1e9,
                                ny=ny, nx=nx,
                                chi2_parts=(numerator, expected, valid.astype(bool)))
        if progress_callback is not None:
            progress_callback(valid_idx.size, valid_idx.size)
        return maps

    def batch_free_tau_fit(self, stack, irf_array, tcspc_res, taus_init, tau_min_s,
                           tau_max_s, n_exp, min_photons, correct_pileup, n_sync_px,
                           n_steps=50, lr=None, tvb_profile=None, fit_tvb=False,
                           fit_idx=None, n_sync_model=None, progress_callback=None,
                           max_iter=200, tol=1e-8, **kwargs):
        if n_exp < 1 or n_exp > MAX_EXP or n_sync_model is not None or \
                (fit_tvb == True and tvb_profile is not None):
            return None
        ny, nx, n_bins = stack.shape
        flat = stack.reshape(ny * nx, n_bins)
        valid_idx = np.flatnonzero(flat.sum(axis=1) >= min_photons)
        maps = self._init_maps(ny, nx, n_exp, intensity=stack.sum(axis=2),
                               taus_fixed_ns=taus_init * 1e9, free_tau=True)
        if valid_idx.size == 0:
            return maps
        raw = np.ascontiguousarray(flat[valid_idx], dtype=np.float64)
        data = raw
        if correct_pileup == True and n_sync_px > 0:
            from flimkit.FLIM.fit_tools import coates_pileup_correction
            data = np.ascontiguousarray(
                [coates_pileup_correction(row, n_sync_px) for row in raw], dtype=np.float64)
        bg = np.ascontiguousarray(
            self._estimate_bg_batch(data.astype(np.float32),
                                    np.ones(data.shape[0], dtype=bool)), dtype=np.float64)
        idx = np.arange(n_bins) if fit_idx is None else np.asarray(fit_idx)
        idx = np.ascontiguousarray(idx, dtype=np.int64)
        irf = np.ascontiguousarray(irf_array, dtype=np.float64)
        dt = tcspc_res / TAU_FIT_UNIT_S
        amp_hi = float(raw.max()) * 10.0
        p0 = np.ascontiguousarray(
            np.concatenate([taus_init / TAU_FIT_UNIT_S,
                            np.full(n_exp, float(raw.max()) / n_exp)]), dtype=np.float64)
        lo = np.ascontiguousarray(
            [float(tau_min_s) / TAU_FIT_UNIT_S] * n_exp + [0.0] * n_exp, dtype=np.float64)
        hi = np.ascontiguousarray(
            [float(tau_max_s) / TAU_FIT_UNIT_S] * n_exp + [amp_hi] * n_exp, dtype=np.float64)
        n = data.shape[0]
        out_p = np.empty((n, 2 * n_exp))
        out_cost = np.empty(n)
        out_iter = np.empty(n, dtype=np.int32)
        numerator = np.empty(n)
        expected = np.empty(n)
        valid = np.empty(n, dtype=np.uint8)
        call = self._lib.flimkit_c_free_tau
        rc = -1
        if self._cuda is not None:
            rc = self._cuda.flimkit_cu_free_tau(
                data, raw, bg, n, n_bins, n_exp, irf, idx, idx.size, dt, p0, lo, hi,
                max_iter, tol, out_p, out_cost, out_iter, numerator, expected, valid)
            if rc != 0:
                print('  [per-pixel] add-on CUDA free-tau failed (' + str(rc)
                      + '), using the C kernel')
        if rc != 0:
            rc = call(data, raw, bg, n, n_bins, n_exp, irf, idx, idx.size, dt, p0, lo, hi,
                      max_iter, tol, out_p, out_cost, out_iter, numerator, expected, valid)
        if rc != 0:
            return None
        taus = out_p[:, :n_exp]
        amps = out_p[:, n_exp:2 * n_exp]
        order = np.argsort(taus, axis=1)
        taus = np.take_along_axis(taus, order, 1) * TAU_FIT_UNIT_S
        amps = np.take_along_axis(amps, order, 1)
        ok = valid.astype(bool) & np.all(np.isfinite(out_p), axis=1)
        dof = max(idx.size - 2 * n_exp, 1)
        calibrated = calibrated_from_terms(numerator, expected, ok)
        self._scatter_free_tau(
            maps, valid_idx=valid_idx[ok], taus_s=taus[ok], amps=amps[ok],
            chi2_r=numerator[ok] / dof, calibrated_values=calibrated[ok],
            ny=ny, nx=nx, n_exp=n_exp)
        if progress_callback is not None:
            progress_callback(valid_idx.size, valid_idx.size)
        return maps

@fit_backend('accelerator',
             'FLIMKit accelerator ' + __version__ + ': C fixed-tau and free-tau (ctypes)',
             priority=100)
def make_backend():
    lib = load_library()
    return None if lib is None else CKernels(lib, load_cuda_library())
