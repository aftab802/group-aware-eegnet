"""Training-set-only augmentation (time shift + Gaussian noise)."""
from __future__ import annotations

import numpy as np


def shift_noise_augment(X: np.ndarray, y: np.ndarray, n_augments: int = 2, noise_std: float = 0.05,
                        max_shift: int = 10, rng: np.random.Generator | None = None):
    """Return the original trials plus `n_augments` shifted, noisy copies.

    Matches the original notebook: each copy is circularly shifted along the
    time axis by one random offset in [-max_shift, max_shift) and has
    N(0, noise_std) noise added. Apply to the training partition only.
    """
    rng = rng or np.random.default_rng(42)
    Xs, ys = [X], [y]
    for _ in range(n_augments):
        shifted = np.roll(X, shift=int(rng.integers(-max_shift, max_shift)), axis=2)
        Xs.append((shifted + rng.normal(0, noise_std, shifted.shape)).astype(np.float32))
        ys.append(y)
    return np.concatenate(Xs, 0), np.concatenate(ys, 0)
