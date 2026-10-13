import numpy as np
import pytest

import flimkit_accelerator_plugin as plugin

lib = plugin.load_library()
pytestmark = pytest.mark.skipif(lib is None, reason='run build.py first')

N_BINS = 128

def basis(taus_bins):
    t = np.arange(N_BINS, dtype=float)
    cols = []
    for tau in taus_bins:
        col = np.exp(-t / tau)
        cols.append(col / col.sum())
    return np.ascontiguousarray(np.stack(cols, axis=1))

def reference(y, A, iters=5000):
    n_exp = A.shape[1]
    M = np.column_stack([A, np.ones(N_BINS) / N_BINS])
    cs = M.sum(axis=0)
    a = np.full(n_exp + 1, y.sum() / (n_exp + 1)) / cs
    for _ in range(iters):
        mu = np.maximum(M @ a, 1e-12)
        a = a * ((y / mu) @ M) / cs
    return a

def run(decay, A, iters=5000, tol=1e-12):
    n = decay.shape[0]
    n_exp = A.shape[1]
    amps = np.empty((n, n_exp))
    offset = np.empty(n)
    numerator = np.empty(n)
    expected = np.empty(n)
    valid = np.empty(n, dtype=np.uint8)
    rc = lib.flimkit_c_fixed_tau_poisson(
        np.ascontiguousarray(decay, dtype=np.float32), n, N_BINS, A, n_exp, iters, tol,
        amps, offset, numerator, expected, valid)
    return rc, amps, offset, numerator, expected, valid

def test_matches_a_dense_reference():
    A = basis([4.0, 20.0])
    rng = np.random.default_rng(1)
    truth = np.array([[300.0, 120.0], [80.0, 40.0], [10.0, 25.0]])
    decay = rng.poisson(truth @ A.T + 0.5)
    rc, amps, offset, _, _, valid = run(decay, A)
    assert rc == 0
    assert valid.all()
    for i in range(decay.shape[0]):
        ref = reference(decay[i].astype(float), A)
        assert np.allclose(amps[i], ref[:2], rtol=1e-3, atol=1e-6)
        assert np.isclose(offset[i], ref[2] / N_BINS, rtol=1e-3, atol=1e-9)

def test_total_counts_are_preserved():
    A = basis([3.0, 15.0, 40.0])
    rng = np.random.default_rng(2)
    decay = rng.poisson(np.array([200.0, 100.0, 50.0]) @ A.T + 1.0, size=(20, N_BINS))
    rc, amps, offset, _, _, _ = run(decay, A)
    assert rc == 0
    fitted = amps @ A.T + offset[:, None]
    assert np.allclose(fitted.sum(axis=1), decay.sum(axis=1), rtol=1e-3)

def test_empty_pixel_is_marked_invalid():
    A = basis([5.0])
    rc, amps, offset, _, _, valid = run(np.zeros((2, N_BINS)), A)
    assert rc == 0
    assert valid.tolist() == [0, 0]
    assert np.all(amps == 0)

def test_unsupported_component_count_is_refused():
    A = basis([2.0, 4.0, 8.0, 16.0])
    rc = run(np.ones((1, N_BINS)), A)[0]
    assert rc == -1

def test_backend_declines_what_it_cannot_do():
    backend = plugin.CKernels(lib)
    stack = np.ones((2, 2, N_BINS), dtype=np.float32)
    A = basis([5.0, 20.0])
    taus = np.array([5.0, 20.0]) * 1e-10
    assert backend.batch_fixed_tau(stack, A, taus, 1, False, 0, None, estimator='lasx') is None
    big = basis([2.0, 4.0, 8.0, 16.0])
    assert backend.batch_fixed_tau(stack, big, np.ones(4) * 1e-9, 1, False, 0, None,
                                   estimator='poisson') is None
