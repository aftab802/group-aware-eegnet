# Group-Aware EEGNet: Subject-Independent Motor Imagery Classification from EEG

[![tests](https://github.com/aftab802/group-aware-eegnet/actions/workflows/tests.yml/badge.svg)](https://github.com/aftab802/group-aware-eegnet/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Subject-independent EEGNet with a verified group-aware split, per-subject normalisation and training-only augmentation.

Code accompanying the manuscript **“Group-Aware EEGNet: Subject-Independent Motor Imagery Classification from EEG”** by Muhammad Aftab ur Rehman, Muhammad Awais ur Rehman, Kiran Asghar, Farwa Zahra (submitted, 2026).
Corresponding author: Muhammad Aftab ur Rehman, CYBEX – School of IT Professionals (Pvt) Limited, Faisalabad, Pakistan.

## Installation

```bash
git clone https://github.com/aftab802/group-aware-eegnet.git
cd group-aware-eegnet
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Python 3.10–3.13 (tested on 3.13 with TensorFlow 2.20). A GPU is optional but recommended for the deep-learning experiments.

## Data

PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB), 109 subjects, 64 channels, 160 Hz —
https://physionet.org/content/eegmmidb/1.0.0/ (DOI [10.13026/C28G6P](https://doi.org/10.13026/C28G6P)).
Nothing needs to be downloaded by hand: the loaders fetch the EDF files through `mne.datasets.eegbci`
on first use and cache them under `~/mne_data` (about 3.4 GB for all subjects; pass `--data-path` to change it).

## Reproducing the paper

```bash
python scripts/run_group_aware_eegnet.py             # subjects 1-70, L0 protocol, 150 epochs
python scripts/run_group_aware_eegnet.py --protocol L2   # same model, record-level split, for contrast
```

Every run writes a JSON file to `results/` with the test metrics, confusion matrix, split sizes and, for subject
data, `train_test_subject_overlap` (0 under the leakage-safe protocol).

## Reproduction status

| Result in the paper | Value | Source of the number | Re-run with this repository? |
|---|---|---|---|
| EEGNet held-out test accuracy | 75.16 % | Original notebook, output preserved in `original_notebooks/` | Same configuration; not re-run here (GPU runs differ slightly between machines) |
| Parameters | 6,466 (6,306 trainable) | Original notebook | Verified: `tests/test_models.py` |
| Split | 47 / 9 / 14 subjects | Original notebook | Same GroupShuffleSplit, random_state 42 |

## Settings that produced 75.16 %

Temporal kernel **(1, 80)**, F1 = 16, D = 2, F2 = 32, dropout **0.5**, batch size **64**, Adam 1e-3, up to 150 epochs,
early stopping on validation accuracy (patience 30), ReduceLROnPlateau (factor 0.5, patience 10). Per-subject
z-scoring uses each subject's own trials (no labels, no cross-subject statistics). Subjects 1–70 were used;
the five subjects with known irregularities (S088, S089, S092, S100, S104) lie outside that range.

## Repository layout

```
cybex_eeg/        shared, leakage-aware library (splits, normalisation, features, models)
scripts/          one command per experiment in the paper; results are written to results/*.json
tests/            pytest suite run on synthetic data (no download needed)
original_notebooks/  the original Colab notebooks, with their printed outputs, for provenance
```

## Tests

```bash
pytest -q
```
Tests check the split/leakage guards, parameter counts against the paper, the PhysioNet four-class extraction on synthetic EDF files, and end-to-end runs of every model on synthetic EEG.

## Citation

See [`CITATION.cff`](CITATION.cff). Until the article is published, please cite the manuscript title above and
this repository.

## Related repositories

* [eeg-mi-leakage-benchmark](https://github.com/aftab802/eeg-mi-leakage-benchmark) — A Systematic Benchmark of Motor Imagery Classifiers Under Leakage-Safe Evaluation
* [eeg-dementia-integrity-benchmark](https://github.com/aftab802/eeg-dementia-integrity-benchmark) — Data Integrity and Feature Engineering in Small-Cohort EEG Dementia Classification
* [eeg-mi-protocol-leakage-case-study](https://github.com/aftab802/eeg-mi-protocol-leakage-case-study) — A Case Study of Evaluation-Protocol Leakage in Motor Imagery EEG Classification
* [eeg-mi-four-class-cnn](https://github.com/aftab802/eeg-mi-four-class-cnn) — Four-Class Motor Imagery Classification Using a Residual Convolutional Neural Network
* [eeg-mi-fbcsp-cnn](https://github.com/aftab802/eeg-mi-fbcsp-cnn) — CSP+CNN: Filter-Bank Common Spatial Patterns with a Convolutional Classifier for Motor Imagery Classification from EEG
* [eeg-dementia-label-correction](https://github.com/aftab802/eeg-dementia-label-correction) — Label Correction and Spectral Features for Multiclass Dementia Detection from EEG
* [lead-eeg](https://github.com/aftab802/lead-eeg) — LEAD-EEG: A Lightweight Ensemble Framework for Alzheimer's Detection from EEG

## License

MIT — see [LICENSE](LICENSE). The datasets are distributed by their owners under their own terms.
