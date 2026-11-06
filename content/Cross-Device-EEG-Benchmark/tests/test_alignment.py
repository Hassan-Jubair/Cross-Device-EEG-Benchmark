import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.alignment import scale_global_rms, online_ea, batched_inv_sqrt, helmert_19d_subspace

def test_causal_zero_future_leakage():
    """Verifies that perturbing future trials (k > 15) has zero effect on aligned trials k <= 15."""
    rng = np.random.default_rng(42)
    X1 = scale_global_rms(rng.standard_normal((30, 20, 300)).astype(np.float32))
    X2 = X1.copy()
    X2[15:] = rng.standard_normal((15, 20, 300)).astype(np.float32)

    for mode in ["none", "kappa", "lw"]:
        out1 = online_ea(X1, mode=mode, kappa=10.0)
        out2 = online_ea(X2, mode=mode, kappa=10.0)
        max_diff = np.max(np.abs(out1[:15] - out2[:15]))
        assert max_diff < 1e-6, f"Causality violated for mode={mode}: max_diff={max_diff}"

def test_batch_whitening_identity():
    """Verifies that full-batch Euclidean Alignment whitens the sample covariance to I_20."""
    rng = np.random.default_rng(42)
    X = scale_global_rms(rng.standard_normal((40, 20, 300)).astype(np.float32))
    T = float(X.shape[-1])
    M = np.mean(np.einsum('nct,ndt->ncd', X.astype(np.float64), X.astype(np.float64)) / T, axis=0)
    W = batched_inv_sqrt(M, eps=1e-12)
    X_w = np.einsum('cd,ndt->nct', W, X)
    cov_w = np.mean(np.einsum('nct,ndt->ncd', X_w.astype(np.float64), X_w.astype(np.float64)) / T, axis=0)
    frob_err = np.linalg.norm(cov_w - np.eye(20), ord='fro')
    assert frob_err < 1e-4, f"Batch whitening error too large: {frob_err}"

def test_helmert_19d_car_orthogonality():
    """Verifies that the 20x19 Helmert basis is orthonormal and orthogonal to 1_20."""
    U19 = helmert_19d_subspace(20)
    assert U19.shape == (20, 19)
    assert np.allclose(U19.T @ U19, np.eye(19), atol=1e-7)
    assert np.allclose(U19.T @ np.ones((20, 1)), 0.0, atol=1e-7)

if __name__ == "__main__":
    test_causal_zero_future_leakage()
    test_batch_whitening_identity()
    test_helmert_19d_car_orthogonality()
    print("[ALL UNIT TESTS PASSED] Causality, Batch Whitening, and 19D CAR Subspace Orthogonality Verified.")
