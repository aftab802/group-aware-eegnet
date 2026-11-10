"""Run every motor-imagery script end to end against synthetic EDF files that mimic
PhysioNet EEGMMIDB (64 channels, 160 Hz, T0/T1/T2 annotations). The download
function is replaced so no network access is needed."""
import glob
import json
import os
import runpy
import sys

import numpy as np
import pytest

mne = pytest.importorskip("mne")
pytest.importorskip("tensorflow")
pytest.importorskip("edfio")
SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
SF = 160.0


def _make_run(path, subject, run, rng):
    n = int(SF * 125)
    t = np.arange(n) / SF
    x = rng.normal(0, 20e-6, (64, n))
    onsets, desc = [], []
    tt, k = 0.0, 0
    while tt + 8.4 < 124:
        onsets.append(tt); desc.append("T0"); tt += 4.2
        lab = "T1" if k % 2 == 0 else "T2"
        onsets.append(tt); desc.append(lab)
        seg = (t >= tt) & (t < tt + 4.2)
        x[0 if lab == "T1" else 1, seg] += 40e-6 * np.sin(2 * np.pi * 10 * t[seg])
        tt += 4.2; k += 1
    info = mne.create_info([f"E{i:02d}" for i in range(64)], SF, "eeg")
    raw = mne.io.RawArray(x, info, verbose=False)
    if run != 1:
        raw.set_annotations(mne.Annotations(onsets, [4.2] * len(onsets), desc))
    mne.export.export_raw(str(path), raw, fmt="edf", physical_range=(-1e-3, 1e-3), verbose=False)


@pytest.fixture
def fake_physionet(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    cache = {}

    def load_data(subject, runs, path=None, update_path=None, **kw):
        out = []
        for r in ([runs] if isinstance(runs, int) else runs):
            p = tmp_path / f"S{subject:03d}R{r:02d}.edf"
            if p not in cache:
                _make_run(p, subject, r, rng); cache[p] = True
            out.append(str(p))
        return out
    from mne.datasets import eegbci
    monkeypatch.setattr(eegbci, "load_data", load_data)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(script, *args):
    old = sys.argv
    sys.argv = [script, *args, "--outdir", "res"]
    sys.path.insert(0, SCRIPTS)
    try:
        runpy.run_path(os.path.join(SCRIPTS, script), run_name="__main__")
    finally:
        sys.argv = old
    return json.load(open(sorted(glob.glob("res/*.json"))[-1]))


def _have(s):
    return os.path.exists(os.path.join(SCRIPTS, s))


@pytest.mark.skipif(not _have("run_group_aware_eegnet.py"), reason="not in this repo")
def test_eegnet_script(fake_physionet):
    r = _run("run_group_aware_eegnet.py", "--subjects", "1-8", "--epochs", "1")
    assert r["train_test_subject_overlap"] == 0 and r["params_total"] == 6466


@pytest.mark.skipif(not _have("run_conv1d_case_study.py"), reason="not in this repo")
def test_conv1d_script(fake_physionet):
    r = _run("run_conv1d_case_study.py", "--subjects", "1-6", "--epochs", "1", "--protocol", "as-executed")
    assert r["validation_is_test"] and r["params_total"] == 186882


@pytest.mark.skipif(not _have("run_csp_cnn.py"), reason="not in this repo")
def test_csp_script(fake_physionet):
    r = _run("run_csp_cnn.py", "--subjects", "1-8", "--epochs", "3")
    assert r["n_csp_features"] == 18 and r["train_test_subject_overlap"] == 0


@pytest.mark.skipif(not _have("run_fourclass_cnn.py"), reason="not in this repo")
def test_fourclass_script(fake_physionet):
    r = _run("run_fourclass_cnn.py", "--subjects", "1-3", "--epochs", "0")
    assert r["n_train"] + r["n_test"] == 3 * 84


@pytest.mark.skipif(not _have("run_benchmark.py"), reason="not in this repo")
def test_benchmark_script_ablation(fake_physionet):
    r = _run("run_benchmark.py", "--models", "conv1d", "--ablation", "--max-subjects", "6", "--epochs", "1")
    assert r["conv1d_L0"]["train_test_subject_overlap"] == 0
    assert r["conv1d_L2"]["train_test_subject_overlap"] > 0 and r["conv1d_as_executed"]["validation_is_test"]
