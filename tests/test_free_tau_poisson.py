"""Low-count regression test for the free-lifetime C kernel.

Observed-count (Neyman) weights bias per-pixel free-lifetime fits at low counts;
the kernel must fit the Poisson likelihood instead.
"""
import numpy as np
import pytest

import flimkit_accelerator_plugin as plugin

lib = plugin.load_library()
pytestmark = pytest.mark.skipif(lib is None, reason='run build.py first')

N_BINS = 256
DT_NS = 0.05
TAUS_NS = (0.5, 3.0)


def _irf():
    t = np.arange(N_BINS) * DT_NS
    g = np.exp(-0.5 * ((t - 2.0) / 0.1) ** 2)
    return g / g.sum()


def _model(irf):
    t = np.arange(N_BINS) * DT_NS
    out = np.zeros(N_BINS)
    for tau in TAUS_NS:
        decay = np.exp(-t / tau)
        out += np.convolve(irf, decay)[:N_BINS]
    return out / out.sum()


def test_free_tau_is_unbiased_at_400_photons_per_pixel():
    irf = _irf()
    shape = _model(irf)
    rng = np.random.default_rng(7)
    n = 256
    data = np.ascontiguousarray(rng.poisson(shape * 400.0, size=(n, N_BINS)), dtype=np.float64)
    backend = plugin.CKernels(lib, None)
    stack = data.reshape(16, 16, N_BINS)
    maps = backend.batch_free_tau_fit(
        stack, irf, DT_NS * 1e-9, np.array(TAUS_NS) * 1e-9 * np.array([1.4, 0.7]),
        5e-11, 2e-8, 2, 10, False, 0, max_iter=200, tol=1e-8)
    assert maps is not None
    taus = np.stack([np.asarray(maps['tau_1']).ravel(), np.asarray(maps['tau_2']).ravel()], 1)
    taus = np.sort(taus, axis=1)
    med = np.nanmedian(taus, axis=0)
    assert abs(med[0] / TAUS_NS[0] - 1) < 0.10
    assert abs(med[1] / TAUS_NS[1] - 1) < 0.05
