"""Read explicit file partitions shared by training and evaluation."""
import csv
from pathlib import Path
import numpy as np


def partition_indices(manifest_path, filenames, fold):
    index = {Path(name).name: i for i, name in enumerate(filenames)}
    if len(index) != len(filenames):
        raise ValueError("Duplicate filenames")
    with Path(manifest_path).open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if str(r["fold"]) == str(fold)]
    if len(rows) != len(filenames) or len({r["filename"] for r in rows}) != len(rows):
        raise ValueError("The split must contain every acquisition exactly once")
    if set(index) != {r["filename"] for r in rows}:
        raise ValueError("Dataset filenames do not match the split")
    if str(fold) != "IID":
        if not all((int(r["participant_id"]) == int(fold)) == (r["partition"] == "test") for r in rows):
            raise ValueError("LOSO held-out subject leakage")
    if set(r["partition"] for r in rows) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation and test partitions")
    return tuple(np.asarray([index[r["filename"]] for r in rows if r["partition"] == p], dtype=int)
                 for p in ("train", "validation", "test"))
