"""Normalisation schemes, named by the scope of data they use."""
from __future__ import annotations

import numpy as np


def per_subject_zscore(X: np.ndarray, groups: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-subject, per-channel z-scoring (as in the original EEGNet notebook).

    Statistics are computed for each subject from that subject's own trials,
    over trials and time, separately for every channel. No labels are used,
    and no statistic is shared across subjects, so test subjects never
    influence the normalisation of training subjects. X: (trials, channels, time).
    """
    Xn = np.empty_like(X, dtype=np.float32)
    for s in np.unique(groups):
        m = groups == s
        mu = X[m].mean(axis=(0, 2), keepdims=True)
        sd = X[m].std(axis=(0, 2), keepdims=True) + eps
        Xn[m] = (X[m] - mu) / sd
    return Xn


def epoch_global_zscore(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """One scalar mean and SD per epoch, over all channels and samples."""
    axes = tuple(range(1, X.ndim))
    mu = X.mean(axis=axes, keepdims=True)
    sd = X.std(axis=axes, keepdims=True) + eps
    return ((X - mu) / sd).astype(np.float32)


def fit_global_zscore(X_fit: np.ndarray) -> tuple[float, float]:
    """Global scalar statistics. Fit on the training partition only for L0/L1;
    fitting on the full dataset is the L2 'normalisation scope' leak."""
    return float(X_fit.mean()), float(X_fit.std() + 1e-8)
