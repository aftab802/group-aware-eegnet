import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def mi_data():
    """Synthetic left/right 'motor imagery': 8 subjects x 20 trials, 8 channels.
    Class 1 carries extra 10 Hz power on channel 0, class 0 on channel 1."""
    rng = np.random.default_rng(0)
    n_sub, n_tr, C, T, sf = 8, 20, 8, 161, 160.0
    t = np.arange(T) / sf
    X, y, g = [], [], []
    for s in range(1, n_sub + 1):
        for i in range(n_tr):
            lab = i % 2
            x = rng.normal(0, 1, (C, T)) * (1 + 0.2 * s)
            x[0 if lab else 1] += 3 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6))
            X.append(x); y.append(lab); g.append(s)
    return np.array(X, dtype=np.float32), np.array(y), np.array(g)


def tf_or_skip():
    return pytest.importorskip("tensorflow")
