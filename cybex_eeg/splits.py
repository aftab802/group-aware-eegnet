"""Train/validation/test partitioning with explicit leakage checks.

`subject_split` reproduces the group-aware protocol of the original EEGNet
notebook (GroupShuffleSplit, test 20 %, then validation 15 % of the rest,
random_state 42) and adds the programmatic intersection check described as
Algorithm 1 in the benchmark paper.  `trial_split` is the leakage-prone,
record-level split kept only so the leakage can be measured.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


class LeakageError(AssertionError):
    """Raised when the same subject appears in more than one partition."""


def assert_subject_disjoint(**partitions: np.ndarray) -> None:
    """Fail loudly if any subject identifier occurs in two partitions.

    Example: ``assert_subject_disjoint(train=g_tr, val=g_va, test=g_te)``.
    """
    names = list(partitions)
    sets = {n: set(np.unique(partitions[n]).tolist()) for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = sets[a] & sets[b]
            if shared:
                raise LeakageError(
                    f"Subjects shared between '{a}' and '{b}': {sorted(shared)[:10]}"
                    f"{' ...' if len(shared) > 10 else ''}")


def subject_split(groups: np.ndarray, y: np.ndarray | None = None, test_size: float = 0.20,
                  val_size: float | None = 0.15, random_state: int = 42) -> dict:
    """Subject-wise split (L0 protocol). Returns index arrays.

    The validation split is drawn from the training subjects only, so the test
    subjects never influence early stopping or checkpoint selection.
    """
    groups = np.asarray(groups)
    idx = np.arange(len(groups))
    dummy = np.zeros(len(groups)) if y is None else y
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval, test = next(gss.split(idx, dummy, groups))
    out = {"test": test}
    if val_size:
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
        tr, va = next(gss_val.split(trainval, dummy[trainval], groups[trainval]))
        out["train"], out["val"] = trainval[tr], trainval[va]
        assert_subject_disjoint(train=groups[out["train"]], val=groups[out["val"]], test=groups[test])
    else:
        out["train"] = trainval
        assert_subject_disjoint(train=groups[trainval], test=groups[test])
    return out


def trial_split(y: np.ndarray, test_size: float = 0.20, random_state: int = 42,
                stratify: bool = True) -> dict:
    """Record-level split that ignores subject identity (leakage-prone, L2).

    Provided only to reproduce and quantify protocol leakage. Do not use it for
    a headline subject-independent result.
    """
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, random_state=random_state,
                              stratify=y if stratify else None)
    return {"train": tr, "test": te}


def subject_overlap(groups_a: np.ndarray, groups_b: np.ndarray) -> int:
    """Number of subjects present in both partitions (0 for an L0 split)."""
    return len(set(np.unique(groups_a)) & set(np.unique(groups_b)))
