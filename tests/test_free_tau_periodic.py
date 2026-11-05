import numpy as np
import pytest

import flimkit_accelerator_plugin as plugin

lib = plugin.load_library()
pytestmark = pytest.mark.skipif(lib is None, reason='run build.py first')

N_PERIOD = 250
DT_NS = 0.05
TAUS_NS = (0.8, 4.0)


def _irf(n):
    t = np.arange(n) * DT_NS
    g = np.exp(-0.5 * ((t - 1.0) / 0.075) ** 2)
    return g / g.sum()


def _pulse_train(n_win):
    t = np.arange(10 * N_PERIOD) * DT_NS
    irf = np.pad(_irf(N_PERIOD), (0, 9 * N_PERIOD))
    single = sum(0.5 * np.convolve(irf, np.exp(-t / tau))[:10 * N_PERIOD] for tau in TAUS_NS)
    return single.reshape(10, N_PERIOD).sum(axis=0)[:n_win]


def _fit(n_win, rep_period_s, photons=2e5):
    shape = _pulse_train(n_win)
    rng = np.random.default_rng(3)
    data = rng.poisson(shape / shape.sum() * photons + 1.0, size=(16, n_win)).astype(np.float64)
    maps = plugin.CKernels(lib, None).batch_free_tau_fit(
        data.reshape(4, 4, n_win), _irf(n_win), DT_NS * 1e-9, np.array(TAUS_NS) * 1e-9 * np.array([1.3, 0.8]),
        5e-11, 2e-8, 2, 10, False, 0, max_iter=200, tol=1e-10, rep_period_s=rep_period_s,
        bg_per_pixel=1.0 if rep_period_s else None)
    assert maps is not None
    return maps


def test_backend_advertises_the_period():
    assert plugin.CKernels(lib, None).free_tau_rep_period == True


def test_short_window_needs_the_period():
    n_win = 200
    on = _fit(n_win, N_PERIOD * DT_NS * 1e-9)
    off = _fit(n_win, None)
    tau_on = np.nanmedian(np.asarray(on['tau_2']))
    tau_off = np.nanmedian(np.asarray(off['tau_2']))
    assert abs(tau_on - 4.0) < abs(tau_off - 4.0)
    assert abs(tau_on / 4.0 - 1) < 0.03
    assert np.nanmedian(np.asarray(on['chi2_r'])) < np.nanmedian(np.asarray(off['chi2_r']))


def test_kernel_model_matches_flimkit():
    pytest.importorskip('flimkit')
    from flimkit.FLIM.models import reconvolution_model
    n_win = 200
    T = N_PERIOD * DT_NS * 1e-9
    maps = _fit(n_win, T, photons=1e7)
    taus = np.array([np.nanmedian(np.asarray(maps['tau_1'])), np.nanmedian(np.asarray(maps['tau_2']))])
    assert taus == pytest.approx(TAUS_NS, rel=0.02)
    params = np.array([4.0e-9, 0.8e-9, 1.0, 1.0, 0.0, 0.0])
    m = reconvolution_model(params, DT_NS * 1e-9, n_win, _irf(n_win), 2, 0.0, False, True, False, rep_period_s=T)
    truth = _pulse_train(n_win) * 2.0
    assert m == pytest.approx(truth, rel=1e-6, abs=1e-9)
