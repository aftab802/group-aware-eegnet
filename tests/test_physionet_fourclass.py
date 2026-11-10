"""Exercise the four-class extraction on synthetic EDF files with real-format annotations."""
import numpy as np
import pytest

pyedflib = pytest.importorskip("pyedflib")
from pyedflib import highlevel  # noqa: E402

from cybex_eeg.physionet import FOURCLASS_NAMES, load_fourclass_subject, to_64x64  # noqa: E402


def _write_run(path, run, rng):
    sf, dur = 160, 125
    sig = rng.normal(0, 10, (64, sf * dur))
    heads = highlevel.make_signal_headers([f"C{i}" for i in range(64)], sample_frequency=sf,
                                          physical_min=-500, physical_max=500)
    ann = []
    if run != 1:
        t = 0.0
        for k in range(15):
            ann.append([t, 4.1, "T0"]); t += 4.1
            ann.append([t, 4.1, "T1" if k % 2 == 0 else "T2"]); t += 4.1
        ann.append([t, 4.1, "T0"])
    highlevel.write_edf(str(path), np.clip(sig, -499, 499), heads, {"annotations": ann})


def test_fourclass_extraction(tmp_path):
    rng = np.random.default_rng(0)
    paths = {}
    for run in (1, 4, 6, 8, 10, 12, 14):
        paths[run] = tmp_path / f"S001R{run:02d}.edf"
        _write_run(paths[run], run, rng)
    trials, labels = load_fourclass_subject(1, edf_paths=paths, rng=rng)
    assert trials.shape == (84, 640, 64) and labels.shape == (84, 4)
    counts = labels.sum(0)
    # imagery runs give 8 left + 7 right each (x3); fists/feet runs keep only feet (T2) -> 7 x 3
    assert counts[0] == 24 and counts[1] == 21 and counts[2] == 21 and counts[3] == 84 - 66
    assert FOURCLASS_NAMES[2] == "both feet" and "rest" in FOURCLASS_NAMES[3]
    X = to_64x64(trials[None])
    assert X.shape == (84, 4096)
