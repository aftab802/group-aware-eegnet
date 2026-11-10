import numpy as np
import pytest

pytest.importorskip("tensorflow")
from cybex_eeg import mi_experiments as E  # noqa: E402


def test_eegnet_L0_smoke(mi_data):
    X, y, g = mi_data
    r = E.run_eegnet(X, y, g, "L0", epochs=1, verbose=0, kernLength=16)
    assert r["train_test_subject_overlap"] == 0 and 0 <= r["test_accuracy"] <= 1


def test_conv1d_protocols(mi_data):
    X, y, g = mi_data
    X = X[:, :, :160]
    leak = E.run_conv1d(X, y, g, "as-executed", epochs=1, verbose=0)
    safe = E.run_conv1d(X, y, g, "L0", epochs=1, verbose=0)
    assert leak["validation_is_test"] and leak["train_test_subject_overlap"] > 0
    assert not safe["validation_is_test"] and safe["train_test_subject_overlap"] == 0


def test_csp_cnn_L0_learns(mi_data):
    X, y, g = mi_data
    r = E.run_csp_cnn(X, y, g, "L0", epochs=60, verbose=0)
    assert r["n_csp_features"] == 18 and r["train_test_subject_overlap"] == 0
    assert r["test_accuracy"] > 0.7


def test_fourclass_protocols():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4096)).astype(np.float32)
    y = np.tile(np.arange(4), 30); g = np.repeat(np.arange(10), 12)
    r = E.run_fourclass(X, y, g, "trial", epochs=0, log_every=0)
    assert r["n_test"] == 12 and r["train_test_subject_overlap"] > 0
    r = E.run_fourclass(X, y, g, "L0", epochs=0, log_every=0)
    assert r["train_test_subject_overlap"] == 0
