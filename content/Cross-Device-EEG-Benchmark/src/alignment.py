import numpy as np
import scipy.linalg as la

def scale_global_rms(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Single-trial Global-RMS scaling: removes amplifier gain shifts while preserving inter-channel variance ratios."""
    X_demean = X - np.mean(X, axis=-1, keepdims=True)
    rms = np.sqrt(np.mean(X_demean ** 2, axis=(1, 2), keepdims=True))
    return (X_demean / (rms + eps)).astype(np.float32)

def scale_per_subject(X: np.ndarray, subjects: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Offline multi-trial per-subject scalar RMS normalization."""
    X_out = np.empty_like(X, dtype=np.float32)
    for s in np.unique(subjects):
        idx = np.where(subjects == s)[0]
        X_s = X[idx] - np.mean(X[idx], axis=-1, keepdims=True)
        rms_s = np.sqrt(np.mean(X_s ** 2))
        X_out[idx] = (X_s / (rms_s + eps)).astype(np.float32)
    return X_out

def scale_per_channel_z(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Channel-wise temporal z-scoring (unit-diagonal correlation normalization)."""
    X_demean = X - np.mean(X, axis=-1, keepdims=True)
    std = np.std(X_demean, axis=-1, keepdims=True)
    return (X_demean / (std + eps)).astype(np.float32)

def batched_inv_sqrt(S: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Symmetric positive-definite matrix inverse square root S^(-1/2)."""
    single = (S.ndim == 2)
    if single:
        S = S[None, :, :]
    w, V = np.linalg.eigh(S.astype(np.float64))
    w = np.clip(w, eps * np.max(w, axis=-1, keepdims=True), None)
    out = np.einsum('nij,nj,nkj->nik', V, w ** -0.5, V).astype(np.float32)
    return out[0] if single else out

def online_ea(X: np.ndarray, mode: str = "none", kappa: float = 0.0, gamma: float = 1e-4, return_covs: bool = False):
    """
    Causal Streaming Euclidean Alignment on a single subject's trial stream X (N, C, T).
    Trial k uses strictly trials 1..k (zero future leakage).
    Modes:
      - 'none'  : Trace-regularized recursive online EA (kappa = 0)
      - 'kappa' : Fixed-prior identity shrinkage rho_k = kappa / (k + kappa)
      - 'lw'    : Stream-adaptive Ledoit-Wolf shrinkage rho_k = min(1, b_k^2 / d_k^2) with rho_1 = 1
    """
    N, C, T = X.shape
    Ci = np.einsum('nct,ndt->ncd', X.astype(np.float64), X.astype(np.float64)) / float(T)
    I = np.eye(C, dtype=np.float64)
    k = np.arange(1, N + 1, dtype=np.float64)
    M = np.cumsum(Ci, axis=0) / k[:, None, None]
    q = np.cumsum((Ci ** 2).sum(axis=(1, 2)))
    mu = np.trace(M, axis1=1, axis2=2) / float(C)

    if mode == "none":
        rho = np.zeros(N, dtype=np.float64)
    elif mode == "kappa":
        rho = kappa / (k + kappa)
    elif mode == "lw":
        num = np.maximum(q - k * (M ** 2).sum(axis=(1, 2)), 0.0)
        v = np.where(k > 1.0, num / np.maximum(k - 1.0, 1.0), 0.0)
        b2 = v / np.maximum(k, 1.0)
        d2 = ((M - mu[:, None, None] * I[None, :, :]) ** 2).sum(axis=(1, 2))
        rho = np.clip(b2 / np.maximum(d2, 1e-12), 0.0, 1.0)
        rho[0] = 1.0
    else:
        raise ValueError(f"Unknown mode: {mode}")

    Sig = (1.0 - rho)[:, None, None] * M + (rho * mu)[:, None, None] * I[None, :, :]
    Sig = (1.0 - gamma) * Sig + (gamma * mu)[:, None, None] * I[None, :, :]
    W = batched_inv_sqrt(Sig)
    X_aligned = np.einsum('ncd,ndt->nct', W, X).astype(np.float32)
    if return_covs:
        return X_aligned, Sig, rho
    return X_aligned

def apply_corpus_batch_ea(X_scaled: np.ndarray, subjects: np.ndarray, gamma: float = 1e-4) -> np.ndarray:
    """Offline transductive per-subject Euclidean Alignment."""
    X_out = np.empty_like(X_scaled)
    T, C = float(X_scaled.shape[-1]), X_scaled.shape[1]
    I = np.eye(C, dtype=np.float64)
    for s in np.unique(subjects):
        idx = np.where(subjects == s)[0]
        X_s = X_scaled[idx].astype(np.float64)
        M = np.mean(np.einsum('nct,ndt->ncd', X_s, X_s) / T, axis=0)
        mu = np.trace(M) / float(C)
        Sig = (1.0 - gamma) * M + gamma * mu * I
        W = batched_inv_sqrt(Sig)
        X_out[idx] = np.einsum('cd,ndt->nct', W, X_scaled[idx])
    return X_out

def helmert_19d_subspace(C: int = 20) -> np.ndarray:
    """Returns orthonormal (C x C-1) Helmert basis orthogonal to the CAR constant vector 1_C."""
    H_full = la.helmert(C, full=True)
    return H_full[1:, :].T
