import ctypes
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import shutil
import subprocess

module_path = Path(__file__).resolve().parents[1] / 'build.py'
spec = importlib.util.spec_from_file_location('project_build', module_path)
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def test_cuda_arch_flags_support_multiple_visible_gpus(monkeypatch):
    monkeypatch.setattr(shutil, 'which', lambda name: '/usr/bin/nvidia-smi' if name == 'nvidia-smi' else None)

    def fake_run(cmd, capture_output, text, check):
        assert cmd == ['/usr/bin/nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader']
        return SimpleNamespace(stdout='8.6\n7.5\n7.5\n', returncode=0)

    monkeypatch.setattr(subprocess, 'run', fake_run)

    flags = build.cuda_arch_flags()

    assert '-gencode=arch=compute_75,code=sm_75' in flags
    assert '-gencode=arch=compute_86,code=sm_86' in flags
    assert '-gencode=arch=compute_86,code=compute_86' in flags
    assert '-arch=native' not in flags


def load_plugin_module():
    fake_flimkit = types.ModuleType('flimkit')
    fake_flimkit.__path__ = []
    fake_flim = types.ModuleType('flimkit.FLIM')
    fake_flim.__path__ = []
    fake_fit_tools = types.ModuleType('flimkit.FLIM.fit_tools')
    fake_fit_tools.TAU_FIT_UNIT_S = 1.0
    fake_fit_tools.calibrated_from_terms = lambda numerator, expected, ok: expected
    fake_gpu = types.ModuleType('flimkit.GPU')

    class PluginBackend(object):
        pass

    fake_gpu.PluginBackend = PluginBackend
    fake_plugins = types.ModuleType('flimkit.plugins')

    def fit_backend(*args, **kwargs):
        def deco(func):
            return func
        return deco

    fake_plugins.fit_backend = fit_backend
    sys.modules['flimkit'] = fake_flimkit
    sys.modules['flimkit.FLIM'] = fake_flim
    sys.modules['flimkit.FLIM.fit_tools'] = fake_fit_tools
    sys.modules['flimkit.GPU'] = fake_gpu
    sys.modules['flimkit.plugins'] = fake_plugins

    module_path = Path(__file__).resolve().parents[1] / 'flimkit_accelerator_plugin' / '__init__.py'
    spec = importlib.util.spec_from_file_location('flimkit_accelerator_plugin', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_native_fixed_tau_lib():
    project_root = Path(__file__).resolve().parents[1]
    if sys.platform == 'darwin':
        lib_name = 'libflimkit_accelerator.dylib'
    elif sys.platform == 'win32':
        lib_name = 'flimkit_accelerator.dll'
    else:
        lib_name = 'libflimkit_accelerator.so'
    lib_path = project_root / 'flimkit_accelerator_plugin' / lib_name
    return ctypes.CDLL(str(lib_path))


def test_fixed_tau_fit_idx_matches_native_c_to_float_tolerance():
    plugin = load_plugin_module()
    lib = _load_native_fixed_tau_lib()
    f32 = np.ctypeslib.ndpointer(np.float32, flags='C_CONTIGUOUS')
    f64 = np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS')
    u8 = np.ctypeslib.ndpointer(np.uint8, flags='C_CONTIGUOUS')
    lib.flimkit_c_fixed_tau.argtypes = [f32, f32, ctypes.c_int64, ctypes.c_int64, f64, f64, ctypes.c_int64, f64, f64, f64, u8]
    lib.flimkit_c_fixed_tau.restype = None

    rng = np.random.default_rng(0)
    cases = [
        {
            'name': 'photon_rich',
            'n_bins': 9,
            'taus': np.array([0.08, 0.7, 2.4, 9.0], dtype=np.float64),
            'bg': np.array([0.05, 0.4], dtype=np.float32),
            'amps': np.array([[1.2e3, 4.5e2, 7.1e1, 2.0e1], [9.8e2, 2.1e2, 5.5e1, 1.7e1]], dtype=np.float64),
            'fit_idx': np.array([0, 2, 4, 6, 8], dtype=np.int64),
        },
        {
            'name': 'photon_starved',
            'n_bins': 9,
            'taus': np.array([0.08, 0.7, 2.4, 9.0], dtype=np.float64),
            'bg': np.array([2.0, 6.0], dtype=np.float32),
            'amps': np.array([[8.0, 4.0, 1.2, 0.5], [7.5, 3.6, 1.0, 0.4]], dtype=np.float64),
            'fit_idx': np.array([1, 3, 5, 7, 8], dtype=np.int64),
        },
        {
            'name': 'extreme_long_tail',
            'n_bins': 13,
            'taus': np.array([0.02, 0.18, 1.3, 10.0, 80.0], dtype=np.float64),
            'bg': np.array([0.2, 1.8], dtype=np.float32),
            'amps': np.array([[5.0e4, 3.0e3, 2.5e2, 1.5e2, 9.0e1], [4.8e4, 2.6e3, 1.9e2, 1.2e2, 8.0e1]], dtype=np.float64),
            'fit_idx': np.array([1, 3, 5, 7, 9, 11, 12], dtype=np.int64),
        },
        {
            'name': 'extreme_sparse_short',
            'n_bins': 17,
            'taus': np.array([0.003, 0.12, 0.9, 7.0, 38.0], dtype=np.float64),
            'bg': np.array([0.8, 5.5], dtype=np.float32),
            'amps': np.array([[1.5e5, 9.0e3, 2.0e2, 8.5e1, 2.1e1], [1.1e5, 6.5e3, 1.6e2, 7.2e1, 1.8e1]], dtype=np.float64),
            'fit_idx': np.array([0, 2, 5, 8, 11, 14, 16], dtype=np.int64),
        },
        {
            'name': 'high_dynamic_range',
            'n_bins': 21,
            'taus': np.array([0.004, 0.06, 0.35, 4.0, 24.0, 110.0], dtype=np.float64),
            'bg': np.array([2.5, 11.0], dtype=np.float32),
            'amps': np.array([[6.0e5, 8.0e4, 3.4e3, 9.5e2, 1.1e2, 2.4e1], [5.5e5, 7.0e4, 2.8e3, 8.0e2, 9.0e1, 1.7e1]], dtype=np.float64),
            'fit_idx': np.array([1, 4, 7, 9, 12, 15, 17, 20], dtype=np.int64),
        },
    ]

    for case in cases:
        n_bins = case['n_bins']
        taus = case['taus']
        times = np.linspace(0.0, 1.0, n_bins, dtype=np.float64) * (n_bins - 1)
        A = np.stack([np.exp(-(times / tau)) for tau in taus], axis=1)
        A = np.ascontiguousarray(A, dtype=np.float64)
        bg = np.asarray(case['bg'], dtype=np.float32)
        amps = np.asarray(case['amps'], dtype=np.float64)
        decay = np.empty((2, n_bins), dtype=np.float32)
        for i in range(2):
            signal = np.maximum(0.0, amps[i].dot(A.T)) + bg[i]
            decay[i] = rng.poisson(signal).astype(np.float32)

        fit_idx = np.asarray(case['fit_idx'], dtype=np.int64)
        decay_fit = np.ascontiguousarray(decay[:, fit_idx], dtype=np.float32)
        A_fit = np.ascontiguousarray(A[fit_idx], dtype=np.float64)
        bg_fit = np.ascontiguousarray(bg, dtype=np.float32)
        py_amps, py_num, py_exp, py_valid = plugin._fixed_tau_python(decay_fit, bg_fit, A_fit)

        n = decay_fit.shape[0]
        a_pinv = np.ascontiguousarray(np.linalg.pinv(A_fit), dtype=np.float64)
        c_amps = np.empty((n, A_fit.shape[1]), dtype=np.float64)
        c_num = np.empty(n, dtype=np.float64)
        c_exp = np.empty(n, dtype=np.float64)
        c_valid = np.empty(n, dtype=np.uint8)
        lib.flimkit_c_fixed_tau(decay_fit, bg_fit, n, A_fit.shape[0], a_pinv, A_fit, A_fit.shape[1],
                                c_amps, c_num, c_exp, c_valid)

        assert py_amps.dtype == c_amps.dtype
        assert py_num.dtype == c_num.dtype
        assert py_exp.dtype == c_exp.dtype
        assert py_valid.dtype == c_valid.dtype
        # These synthetic cases intentionally push the basis toward extreme
        # lifetimes and sparse bin selections. The correct comparison is a
        # relative floating-point tolerance on the magnitudes involved, not a
        # bit-for-bit cutoff that falsely fails on large-amplitude signals.
        assert np.allclose(py_amps, c_amps, rtol=1e-12, atol=1e-12)
        assert np.allclose(py_num, c_num, rtol=1e-12, atol=1e-12)
        assert np.allclose(py_exp, c_exp, rtol=1e-12, atol=1e-12)
        assert np.array_equal(py_valid, c_valid)
