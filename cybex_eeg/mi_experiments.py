"""Motor-imagery experiments, parameterised by evaluation protocol.

Every ``run_*`` function takes already-loaded arrays and a ``protocol``:

  "L0"     subject-wise train/val/test split with the explicit intersection check
  "L2"     record-level (trial) split, stratified by class only; validation drawn
           from the training trials
  "as-executed" (1D CNN only) record-level split AND the test partition reused
           as Keras validation data - the protocol behind the 71.46 % figure

and returns a JSON-serialisable dict. Hyper-parameters default to the values
printed in the respective papers or, where a paper is silent, to the value in
the original notebook; unspecified values are marked in comments.
"""
from __future__ import annotations

import numpy as np

from .augment import shift_noise_augment
from .normalize import epoch_global_zscore, per_subject_zscore
from .splits import assert_subject_disjoint, subject_overlap, subject_split, trial_split


def _seed(seed: int):
    import random

    import tensorflow as tf
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


def _split(protocol, y, groups, test_size=0.20, val_size=0.15, seed=42):
    if protocol == "L0":
        return subject_split(groups, y, test_size=test_size, val_size=val_size, random_state=seed)
    if protocol in ("L2", "as-executed"):
        s = trial_split(y, test_size=test_size, random_state=seed)
        if protocol == "as-executed":
            s["val"] = s["test"]  # leakage point 2: test partition drives early stopping
        else:
            tr, va = trial_split(y[s["train"]], test_size=val_size, random_state=seed).values()
            s["train"], s["val"] = s["train"][tr], s["train"][va]
        return s
    raise ValueError(f"Unknown protocol {protocol!r}")


def _report(y_true, y_pred, split, groups, extra=None):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    out = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "n_train": int(len(split["train"])), "n_test": int(len(split["test"])),
        "subjects": {k: int(len(np.unique(groups[v]))) for k, v in split.items()},
        "train_test_subject_overlap": subject_overlap(groups[split["train"]], groups[split["test"]]),
    }
    out.update(extra or {})
    return out


# ------------------------------------------------------------------ EEGNet (group-aware)
def run_eegnet(X, y, groups, protocol="L0", epochs=150, batch_size=64, seed=42, verbose=2,
               kernLength=80, F1=16, D=2, F2=32, dropout=0.5):
    """Group-aware EEGNet: notebook settings that produced 75.16 % (subjects 1-70)."""
    import tensorflow as tf

    from .models import build_eegnet
    _seed(seed)
    s = _split(protocol, y, groups, seed=seed)
    norm = {k: per_subject_zscore(X[v], groups[v])[..., None] for k, v in s.items()}
    Xtr, ytr = shift_noise_augment(norm["train"], y[s["train"]], n_augments=2,
                                   rng=np.random.default_rng(seed))
    model = build_eegnet(2, X.shape[1], X.shape[2], dropoutRate=dropout, kernLength=kernLength, F1=F1, D=D, F2=F2)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=30, restore_best_weights=True),
          tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6)]
    h = model.fit(Xtr, ytr, validation_data=(norm["val"], y[s["val"]]), epochs=epochs, batch_size=batch_size,
                  callbacks=cb, verbose=verbose)
    pred = model.predict(norm["test"], verbose=0).argmax(1)
    return _report(y[s["test"]], pred, s, groups, {
        "model": "EEGNet", "protocol": protocol, "params_total": int(model.count_params()),
        "epochs_run": len(h.history["loss"]), "history": {k: [float(v) for v in vs] for k, vs in h.history.items()}})


# ------------------------------------------------------------------ 1D CNN
def run_conv1d(X, y, groups, protocol="as-executed", epochs=40, batch_size=32, seed=42, verbose=2):
    """Three-block 1D CNN of the protocol-leakage case study. X: (epochs, 64, 320)."""
    import tensorflow as tf

    from .models import build_conv1d_cnn
    _seed(seed)
    s = _split(protocol, y, groups, seed=seed)
    Xn = epoch_global_zscore(X).transpose(0, 2, 1)  # per-epoch scalar z-score; (N, time, channels)
    Y = tf.keras.utils.to_categorical(y, 2)
    model = build_conv1d_cnn(X.shape[2], X.shape[1])
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
          tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)]
    h = model.fit(Xn[s["train"]], Y[s["train"]], validation_data=(Xn[s["val"]], Y[s["val"]]), epochs=epochs,
                  batch_size=batch_size, callbacks=cb, verbose=verbose)
    pred = model.predict(Xn[s["test"]], verbose=0).argmax(1)
    return _report(y[s["test"]], pred, s, groups, {
        "model": "1D CNN", "protocol": protocol, "params_total": int(model.count_params()),
        "validation_is_test": protocol == "as-executed", "epochs_run": len(h.history["loss"]),
        "history": {k: [float(v) for v in vs] for k, vs in h.history.items()}})


# ------------------------------------------------------------------ CSP + CNN
def run_csp_cnn(X, y, groups, protocol="L0", epochs=100, batch_size=32, seed=42, verbose=2, sfreq=160.0):
    """Filter-bank CSP (fit on training trials only) + compact Conv1D head.

    Paper-silent settings: validation fraction 15 % of training subjects/trials,
    100 epochs maximum, early-stopping patience 15, batch size 32.
    """
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler

    from .fbcsp import FilterBankCSP
    from .models import build_csp_head
    _seed(seed)
    s = _split(protocol, y, groups, seed=seed)
    csp = FilterBankCSP(sfreq=sfreq).fit(X[s["train"]], y[s["train"]])  # training trials only
    F = {k: csp.transform(X[v]) for k, v in s.items()}
    sc = StandardScaler().fit(F["train"])
    F = {k: sc.transform(v)[..., None] for k, v in F.items()}
    model = build_csp_head(F["train"].shape[1])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True),
          tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)]
    h = model.fit(F["train"], y[s["train"]], validation_data=(F["val"], y[s["val"]]), epochs=epochs,
                  batch_size=batch_size, callbacks=cb, verbose=verbose)
    pred = (model.predict(F["test"], verbose=0)[:, 0] >= 0.5).astype(int)
    train_acc = float(((model.predict(F["train"], verbose=0)[:, 0] >= 0.5).astype(int) == y[s["train"]]).mean())
    return _report(y[s["test"]], pred, s, groups, {
        "model": "CSP+CNN", "protocol": protocol, "train_accuracy": train_acc,
        "params_trainable_head": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
        "n_csp_features": int(csp.n_features_out_), "epochs_run": len(h.history["loss"])})


# ------------------------------------------------------------------ four-class CNN
def run_fourclass(X4096, y, groups, protocol="trial", epochs=100, seed=42, log_every=5):
    """Four-class CNN on the 64x64 representation.

    protocol "trial": the original split - one permutation of all trials
    (random_state 42), first 90 % train, last 10 % test (subjects shared).
    protocol "L0": 90/10 split over subjects (Algorithm 2 of the four-class paper).
    """
    import tensorflow as tf

    from .models import build_fourclass_cnn, train_fourclass_cnn
    _seed(seed)
    if protocol == "trial":
        idx = np.random.RandomState(seed).permutation(len(y))
        cut = int(len(y) * 0.9)
        s = {"train": idx[:cut], "test": idx[cut:]}
    elif protocol == "L0":
        s = subject_split(groups, y, test_size=0.10, val_size=None, random_state=seed)
        assert_subject_disjoint(train=groups[s["train"]], test=groups[s["test"]])
    else:
        raise ValueError(protocol)
    Y = np.eye(4, dtype=np.float32)[y]
    model = build_fourclass_cnn()
    hist = train_fourclass_cnn(model, X4096[s["train"]], Y[s["train"]], X4096[s["test"]], Y[s["test"]],
                               epochs=epochs, seed=seed, log_every=log_every)
    pred = model(X4096[s["test"]], training=False).numpy().argmax(1)
    return _report(y[s["test"]], pred, s, groups, {
        "model": "4-class CNN", "protocol": protocol, "params_total": int(model.count_params()),
        "final_train_accuracy_first200": hist["train_acc"][-1], "history": hist})
