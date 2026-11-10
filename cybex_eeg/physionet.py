"""Loaders for the PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB).

Dataset: https://physionet.org/content/eegmmidb/1.0.0/ (DOI 10.13026/C28G6P).
Files are downloaded on first use through ``mne.datasets.eegbci`` and cached
under ``~/mne_data`` (or ``data_path``).

Run numbers: R03/R07/R11 executed left/right fist, R04/R08/R12 imagined
left/right fist, R05/R09/R13 executed fists/feet, R06/R10/R14 imagined
fists/feet, R01 eyes-open baseline. Annotation T1 = left fist (or both fists in
R05/06/09/10/13/14), T2 = right fist (or both feet).
"""
from __future__ import annotations

import numpy as np

IMAGERY_LR_RUNS = (4, 8, 12)
EXECUTED_LR_RUNS = (3, 7, 11)
# Subjects with documented sampling-rate/annotation irregularities on EEGMMIDB.
IRREGULAR_SUBJECTS = (88, 89, 92, 100, 104)


def load_lr_subject(subject: int, runs=IMAGERY_LR_RUNS, tmin: float = 0.0, tmax: float = 4.0,
                    l_freq: float = 8.0, h_freq: float = 30.0, resample: float | None = None,
                    data_path: str | None = None):
    """Left (0) vs right (1) fist epochs for one subject: X (trials, 64, time), y.

    Mirrors the original EEGNet notebook: concatenate runs, standardise channel
    names, set the 10-05 montage, FIR band-pass 8-30 Hz, epoch T1/T2 without
    baseline correction. ``resample`` makes subjects recorded at 128 Hz
    compatible with the 160 Hz majority.
    """
    import mne
    from mne.datasets import eegbci
    from mne.io import concatenate_raws, read_raw_edf

    fnames = eegbci.load_data(subject, list(runs), path=data_path, update_path=False)
    raw = concatenate_raws([read_raw_edf(f, preload=True, verbose="ERROR") for f in fnames])
    eegbci.standardize(raw)
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore")
    if resample and abs(raw.info["sfreq"] - resample) > 1e-6:
        raw.resample(resample, verbose="ERROR")
    raw.filter(l_freq, h_freq, fir_design="firwin", skip_by_annotation="edge", verbose="ERROR")
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    epochs = mne.Epochs(raw, events, event_id={"left": event_id["T1"], "right": event_id["T2"]},
                        tmin=tmin, tmax=tmax, proj=True, picks=picks, baseline=None,
                        preload=True, verbose="ERROR")
    X = epochs.get_data(copy=False).astype(np.float32)
    y = (epochs.events[:, -1] == event_id["T2"]).astype(int)
    return X, y


def load_lr_dataset(subjects, skip_errors: bool = True, verbose: bool = True, **kwargs):
    """Stack several subjects. Returns X, y, groups (subject id per trial)."""
    Xs, ys, gs = [], [], []
    for s in subjects:
        try:
            X, y = load_lr_subject(s, **kwargs)
        except Exception as exc:  # pragma: no cover - network/data dependent
            if not skip_errors:
                raise
            if verbose:
                print(f"Subject {s:3d}: skipped ({exc})")
            continue
        Xs.append(X); ys.append(y); gs.append(np.full(len(y), s))
        if verbose:
            print(f"Subject {s:3d}: {len(y)} trials")
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


# ---------------------------------------------------------------------------
# Four-class representation (port of the original EEG-DL style extraction)
# ---------------------------------------------------------------------------
FOURCLASS_NAMES = ("left fist", "right fist", "both feet", "rest (R01 baseline)")
SAMPLE_RATE = 160
EEG_CHANNELS = 64
TRIAL_SECONDS = 4
RUN_LENGTH = 125 * SAMPLE_RATE


def _read_edf(path):
    import pyedflib
    sig = pyedflib.EdfReader(path)
    n = sig.signals_in_file
    buf = np.zeros((n, sig.getNSamples()[0]))
    for j in range(n):
        buf[j, :] = sig.readSignal(j)
    ann = list(zip(*sig.readAnnotations()))
    sig._close()
    return buf.T, ann


def load_fourclass_subject(subject: int, data_path: str | None = None, rng: np.random.Generator | None = None,
                           edf_paths: dict | None = None):
    """84 four-second trials (21 per class) for one subject: (84, 640, 64), one-hot (84, 4).

    Faithful port of the original notebook. Classes, in label order:
      0 left fist (T1 in R04/R08/R12), 1 right fist (T2 in R04/R08/R12),
      2 both feet (T2 in R06/R10/R14; T1 'both fists' trials are dropped),
      3 rest - random 4 s windows from the R01 eyes-open baseline, used to fill
        the remaining slots up to 84 trials.
    ``edf_paths`` maps run -> local file (used by the tests); otherwise files are
    fetched with ``mne.datasets.eegbci``.
    """
    rng = rng or np.random.default_rng()
    mi_runs = [4, 8, 12, 6, 10, 14]
    if edf_paths is None:
        from mne.datasets import eegbci
        files = eegbci.load_data(subject, [1] + mi_runs, path=data_path, update_path=False)
        edf_paths = dict(zip([1] + mi_runs, files))
    X = np.zeros((len(mi_runs), RUN_LENGTH, EEG_CHANNELS))
    events = []
    for i_run, run in enumerate(mi_runs):
        signals, annotations = _read_edf(str(edf_paths[run]))
        n = min(signals.shape[0], RUN_LENGTH)
        X[i_run, :n, :] = signals[:n]
        current = [i_run, 0, 0, 0]
        for onset, _dur, desc in annotations:
            t = int(onset * SAMPLE_RATE)
            action = int(str(desc)[1])
            if action == 0 and current[1] != 0:
                length = TRIAL_SECONDS * SAMPLE_RATE
                pad = (length - (t - current[2])) // 2
                current[2] -= pad + (t - current[2]) % 2
                current[3] = t + pad
                if (run - 6) % 4 != 0 or current[1] == 2:
                    if (run - 6) % 4 == 0:
                        current[1] = 3
                    events.append(current)
            elif action > 0:
                current = [i_run, action, t, 0]
    n_trials = 21 * 4
    trials = np.zeros((n_trials, TRIAL_SECONDS * SAMPLE_RATE, EEG_CHANNELS))
    labels = np.zeros((n_trials, 4))
    for i, ev in enumerate(events[:n_trials]):
        seg = X[ev[0], ev[2]:ev[3]]
        trials[i, :len(seg), :] = seg
        labels[i, ev[1] - 1] = 1.0
    base, _ = _read_edf(str(edf_paths[1]))
    S = TRIAL_SECONDS * SAMPLE_RATE
    for i in range(min(len(events), n_trials), n_trials):
        off = int(rng.integers(0, base.shape[0] - S))
        trials[i] = base[off:off + S]
        labels[i, -1] = 1.0
    return trials, labels


def to_64x64(trials: np.ndarray, time_points: int = 64) -> np.ndarray:
    """(subjects, trials, 640, 64) -> (subjects*trials, 4096) channel-major rows.

    Each channel's 640 samples are reduced to 64 by averaging non-overlapping
    windows of 10 samples, exactly as in the original preprocessing.
    """
    s, t, n, c = trials.shape
    w = n // time_points
    ds = trials[:, :, :w * time_points, :].reshape(s, t, time_points, w, c).mean(axis=3)
    return ds.transpose(0, 1, 3, 2).reshape(s * t, c * time_points).astype(np.float32)
