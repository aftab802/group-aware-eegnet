"""Group-Aware EEGNet (subject-independent left/right motor imagery).

Reproduces the configuration behind the 75.16 % result: subjects 1-70, runs
R04/R08/R12, 0-4 s epochs, 8-30 Hz, subject-wise 20 % test / 15 % validation
split (random_state 42) with an explicit intersection check, per-subject
normalisation, training-only shift+noise augmentation, EEGNet F1=16 D=2 F2=32,
kernel length 80, dropout 0.5, batch 64, up to 150 epochs.
"""
from _common import parser, save, subjects_arg

from cybex_eeg.mi_experiments import run_eegnet
from cybex_eeg.physionet import IRREGULAR_SUBJECTS, load_lr_dataset

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--subjects", default="1-70")
    p.add_argument("--protocol", default="L0", choices=["L0", "L2"])
    p.add_argument("--epochs", type=int, default=150)
    a = p.parse_args()
    subs = [s for s in subjects_arg(a.subjects) if s not in IRREGULAR_SUBJECTS]
    X, y, g = load_lr_dataset(subs, tmin=0.0, tmax=4.0, data_path=a.data_path)
    print("Data:", X.shape, "subjects:", len(set(g)))
    save(run_eegnet(X, y, g, protocol=a.protocol, epochs=a.epochs, seed=a.seed), f"eegnet_{a.protocol}", a.outdir)
