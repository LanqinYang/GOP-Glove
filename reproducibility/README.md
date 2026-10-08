# Reproducibility materials

This directory records the dataset identity, current evaluation protocol and
unmodified diagnostic outputs associated with the GoP glove project. Read
[REPRODUCTION_STATUS.md](REPRODUCTION_STATUS.md) before interpreting a result.

## Validate the materials

From the repository root, using Python 3.12:

```bash
python -m pip install -r reproducibility/requirements-audit.txt
python scripts/export_reproducibility.py
python scripts/verify_reproducibility.py
```

The first command installs the small audit environment. Full model training
uses the project requirements and was not rerun during this material check.

## Files

| File or directory | Contents |
|---|---|
| `dataset_manifest.csv` | Filenames, participant/class/repetition IDs, raw sample counts and SHA-256 hashes |
| `generated/iid_seed42.csv` | Current deterministic IID protocol: 422 training, 106 validation, 132 test files |
| `generated/loso_seed42.csv` | Six current deterministic LOSO manifests: 440 training, 110 validation and 110 held-out-subject test files per fold |
| `generated/protocol_metadata.json` | Ordering, split ratios, descriptor export and historical-reproduction status |
| `legacy_features.py` | Exact descriptor class extracted from the archived ADANN source without changing its feature definitions |
| `generated/legacy_unscaled_features.npy` | 660 × 190 descriptor inspection array; no globally fitted scaler |
| `results/DA_LGBM_seed_stability_gated_seeds_42_123_2025_2026_3047/` | Five-seed raw results, parameters and original summaries |
| `results/v7_fusion_delta_final_seed810/` | Independent gate audit, including all 660 branch-confidence/margin records |
| `results/robustness/` | Original, unshifted AWGN, sampling-jitter and sensor-drift results with their parameter references |
| `source_manifest.json` | Hashes of copied project sources and frozen diagnostic parameters |
| `verification.json` | Checks actually performed; does not claim a model retraining or physical device test |
| `results/archived_confusion_matrices/` | Archived count matrices; filenames alone do not establish matched training-run provenance |

The current split manifests are generated from lexicographic filenames. Older
training code read some files through an unsorted glob, and its historical
ordering has not been recovered. These manifests are therefore protocol
definitions generated for inspection, rather than claimed copies of the
original primary-run partitions.

Frozen robustness parameter files remain under their original `outputs/...`
paths because the existing scripts resolve parameters there. They are
explicit inputs, not a fresh run produced by this upload.

Use `python scripts/run_frozen_robustness.py --help` to inspect the parameter-frozen
launcher. It selects the exact files recorded in each diagnostic's metadata.
Training still requires the full dependencies and any historically unrecorded
flags must be supplied deliberately; no baseline-alignment script is included.
