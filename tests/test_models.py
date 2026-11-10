import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from cybex_eeg.models import build_conv1d_cnn, build_csp_head, build_eegnet, build_fourclass_cnn  # noqa: E402


def _trainable(m):
    return int(sum(np.prod(w.shape) for w in m.trainable_weights))


def test_eegnet_matches_reported_parameter_count():
    m = build_eegnet(2, 64, 641, dropoutRate=0.5, kernLength=80, F1=16, D=2, F2=32)
    assert m.count_params() == 6466 and _trainable(m) == 6306


def test_eegnet_kernel64_would_not_match():
    # the (1, 64) temporal kernel printed in the paper's layer table gives a different total
    assert build_eegnet(2, 64, 641, kernLength=64, F1=16, D=2, F2=32).count_params() == 6210


def test_conv1d_matches_paper_table():
    m = build_conv1d_cnn(320, 64)
    assert m.count_params() == 186882 and _trainable(m) == 185986
    assert m(np.zeros((2, 320, 64), np.float32)).shape == (2, 2)


def test_csp_head_forward_and_size():
    m = build_csp_head(18)
    assert m(np.zeros((3, 18, 1), np.float32)).shape == (3, 1)
    assert _trainable(m) == 9473  # documented discrepancy with the paper's printed ~2,993


def test_fourclass_cnn_forward_and_size():
    m = build_fourclass_cnn()
    assert m(np.random.rand(4, 4096).astype(np.float32), training=False).shape == (4, 4)
    assert 14_900_000 < m.count_params() < 15_100_000
