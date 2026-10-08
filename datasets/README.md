# GoP glove sensor recordings

The dataset contains 660 independent recordings from six pseudonymous
participants. Classes 0–9 represent BSL digit gestures; class 10 is Static.
Each participant-class combination has ten repetitions. Acquisition is
nominally 50 Hz for two seconds over five GoP channels.

CSV files contain two comment records followed by these columns:

| Column | Meaning | Unit |
|---|---|---|
| `timestamp_ms` | Timestamp within the recorded acquisition | ms |
| `thumb`, `index`, `middle`, `ring`, `pinky` | Five corresponding sensor channels | ADC code |

Filenames encode participant ID, class, repetition and recording timestamp.
Participant IDs are pseudonymous labels. The files contain sensor readings
and these acquisition identifiers; they do not contain names or contact
information.

Most raw files contain 97 samples per channel; 12 contain 96 and one contains
99. The existing software linearly resamples recordings to 100 points before
feature extraction. See `reproducibility/dataset_manifest.csv` for file-level
SHA-256 hashes and raw sample counts.

Evaluation should keep held-out subjects out of training and fitted
preprocessing. Current-protocol IID and LOSO file manifests are provided under
`reproducibility/generated/`; their historical-reproduction limits are
documented there. See the repository LICENSE and existing project attribution.
