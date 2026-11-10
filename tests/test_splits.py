import numpy as np
import pytest

from cybex_eeg.splits import LeakageError, assert_subject_disjoint, subject_overlap, subject_split, trial_split


def test_subject_split_is_disjoint_and_complete(mi_data):
    _, y, g = mi_data
    s = subject_split(g, y, test_size=0.25, val_size=0.2, random_state=42)
    assert sorted(np.concatenate(list(s.values()))) == list(range(len(g)))
    for a in s:
        for b in s:
            if a != b:
                assert subject_overlap(g[s[a]], g[s[b]]) == 0


def test_trial_split_shares_subjects(mi_data):
    _, y, g = mi_data
    s = trial_split(y, test_size=0.25, random_state=42)
    assert subject_overlap(g[s["train"]], g[s["test"]]) > 0  # the leak this repo is about


def test_intersection_check_raises():
    with pytest.raises(LeakageError):
        assert_subject_disjoint(train=np.array([1, 2, 3]), test=np.array([3, 4]))
    assert_subject_disjoint(train=np.array([1, 2]), test=np.array([3, 4]))


def test_split_is_deterministic(mi_data):
    _, y, g = mi_data
    a, b = subject_split(g, y, random_state=42), subject_split(g, y, random_state=42)
    assert all((a[k] == b[k]).all() for k in a)
