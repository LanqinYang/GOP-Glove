"""Export IID/LOSO file partitions and the shared 190-D descriptor."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproducibility.features import EnhancedFeatureExtractor


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_windows(folder):
    files, labels, subjects, windows = [], [], [], []
    for path in sorted(folder.glob("*.csv")):
        match = re.search(r"user_(\d+)_gesture_(\d+)", path.name)
        if not match:
            raise ValueError(f"Unrecognised dataset filename: {path.name}")
        with path.open(newline="") as stream:
            rows = [r for r in csv.reader(stream) if r and not r[0].startswith("#")]
        if rows[0] != ["timestamp_ms", "thumb", "index", "middle", "ring", "pinky"]:
            raise ValueError(f"Unexpected schema: {path.name}")
        values = np.asarray([[float(x) for x in row[1:6]] for row in rows[1:]])
        grid = np.linspace(0, len(values) - 1, 100)
        resampled = np.column_stack([np.interp(grid, np.arange(len(values)), values[:, c]) for c in range(5)])
        files.append(path.name)
        subjects.append(int(match.group(1)))
        labels.append(int(match.group(2)))
        windows.append(np.round(resampled).astype(np.float32))
    return files, np.asarray(labels), np.asarray(subjects), np.asarray(windows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=ROOT / "datasets/gesture_csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproducibility/generated")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    files, labels, subjects, windows = load_windows(args.csv_dir)
    if len(files) != 660 or set(subjects) != set(range(1, 7)) or set(labels) != set(range(11)):
        raise ValueError("Expected the six-participant, 11-class, 660-window dataset")
    fields = ["filename", "participant_id", "class_id", "fold", "partition", "seed"]
    all_indices = np.arange(len(files))
    trainval, test = train_test_split(all_indices, test_size=0.2, random_state=args.seed, stratify=labels)
    train, validation = train_test_split(trainval, test_size=0.2, random_state=args.seed, stratify=labels[trainval])
    iid = []
    for partition, indices in (("train", train), ("validation", validation), ("test", test)):
        iid.extend(dict(filename=files[i], participant_id=int(subjects[i]), class_id=int(labels[i]), fold="IID", partition=partition, seed=args.seed) for i in indices)
    write_csv(args.output_dir / f"iid_seed{args.seed}.csv", iid, fields)
    loso = []
    for subject in sorted(set(subjects)):
        outer = all_indices[subjects != subject]
        test = all_indices[subjects == subject]
        train, validation = train_test_split(outer, test_size=0.2, random_state=args.seed, stratify=labels[outer])
        for partition, indices in (("train", train), ("validation", validation), ("test", test)):
            loso.extend(dict(filename=files[i], participant_id=int(subjects[i]), class_id=int(labels[i]), fold=int(subject), partition=partition, seed=args.seed) for i in indices)
    write_csv(args.output_dir / f"loso_seed{args.seed}.csv", loso, fields)
    extractor = EnhancedFeatureExtractor()
    features = np.stack([extractor.extract_comprehensive_features(window) for window in windows])
    if features.shape != (660, 190) or not np.isfinite(features).all():
        raise ValueError("Descriptor dimension/finite-value check failed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "manuscript_unscaled_features.npy", features)
    write_csv(args.output_dir / "feature_row_order.csv", [dict(row=i, filename=name) for i, name in enumerate(files)], ["row", "filename"])
    metadata = {
        "manifest_kind": "manuscript_evaluation_protocol",
        "input_order": "lexicographic_filename", "seed": args.seed,
        "iid_test_ratio": 0.2, "iid_validation_ratio_of_remaining": 0.2,
        "iid_counts": {partition: sum(r["partition"] == partition for r in iid) for partition in ("train", "validation", "test")},
        "loso_counts_per_fold": {"train": 440, "validation": 110, "test": 110},
        "features": {"shape": list(features.shape), "preprocessing": "linear resample to 100; round; float32; no fitted scaler", "use": "descriptor inspection; not the train-fitted model input", "periodogram_sampling_rate_hz": 50, "physical_acquisition_hz": 50},
        "input_filename_order_sha256": hashlib.sha256("\n".join(files).encode()).hexdigest(),

    }
    (args.output_dir / "protocol_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
