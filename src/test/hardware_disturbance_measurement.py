#!/usr/bin/env python3
"""Capture and summarize GoP glove hardware disturbance measurements."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CHANNELS = [f"A{i}" for i in range(5)]
DATASET_CHANNEL_MAP = {
    "thumb": "A0",
    "index": "A1",
    "middle": "A2",
    "ring": "A3",
    "pinky": "A4",
}


def auto_port() -> str:
    candidates: list[str] = []
    for pattern in (
        "/dev/cu.usbmodem*",
        "/dev/cu.usbserial*",
        "/dev/cu.SLAB_USBtoUART*",
        "/dev/cu.wchusbserial*",
    ):
        candidates.extend(sorted(glob.glob(pattern)))
    if not candidates:
        raise RuntimeError("No Arduino serial port found. Connect the Nano and retry.")
    return candidates[0]


def read_line_values(line: str) -> list[int] | None:
    text = line.strip()
    if not text or text.startswith("Ready") or text.startswith("device_us"):
        return None
    parts = text.replace(",", "\t").split()
    if len(parts) != 6:
        return None
    try:
        return [int(part) for part in parts]
    except ValueError:
        return None


def capture(args: argparse.Namespace) -> Path:
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("pyserial is required for capture; use /Users/aqin/miniforge3/bin/python.") from exc

    port = args.port or auto_port()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{args.condition}_{int(args.duration_s)}s_{stamp}.csv"

    with serial.Serial(port, args.baud, timeout=1) as ser, out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "host_time_s", "device_us", *CHANNELS])
        time.sleep(args.settle_s)
        ser.reset_input_buffer()
        ser.write(b"S")
        start = time.time()
        first_device_us: int | None = None
        rows = 0
        while True:
            if args.stop_mode == "host" and time.time() - start >= args.duration_s:
                break
            raw = ser.readline().decode("utf-8", errors="ignore")
            values = read_line_values(raw)
            if values is None:
                continue
            device_us = values[0]
            if first_device_us is None:
                first_device_us = device_us
            elif args.stop_mode == "device" and device_elapsed_s(device_us, first_device_us) > args.duration_s:
                break
            writer.writerow([args.condition, time.time(), *values])
            rows += 1
        ser.write(b"X")

    print(f"Saved {rows} rows to {out_path}")
    return out_path


def device_elapsed_s(current_us: int, first_us: int) -> float:
    if current_us < first_us:
        current_us += 2**32
    return (current_us - first_us) / 1_000_000.0


def infer_condition(path: str | Path) -> str:
    stem = Path(path).stem.lower()
    if "static" in stem:
        return "static_existing"
    if "bend" in stem:
        return "bend"
    return stem.split("_")[0]


def read_csv_flexible(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        frame = pd.read_csv(path)
    except pd.errors.ParserError:
        frame = pd.DataFrame()
    if "timestamp_ms" not in frame.columns and "device_us" not in frame.columns:
        frame = pd.read_csv(path, skiprows=2)
    return frame


def normalize_frame(frame: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    if set(DATASET_CHANNEL_MAP).issubset(frame.columns) and "timestamp_ms" in frame.columns:
        frame = frame.rename(columns=DATASET_CHANNEL_MAP).copy()
        frame["device_us"] = (frame["timestamp_ms"].astype(float) * 1000.0).round().astype(np.uint64)
        frame["host_time_s"] = frame["timestamp_ms"].astype(float) / 1000.0
    if "condition" not in frame.columns:
        frame["condition"] = infer_condition(path)
    return frame


def load_frames(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = normalize_frame(read_csv_flexible(path), path)
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise RuntimeError("No input CSV files were supplied.")
    data = pd.concat(frames, ignore_index=True)
    missing = {"device_us", *CHANNELS} - set(data.columns)
    if missing:
        raise RuntimeError(f"Input data missing required columns: {sorted(missing)}")
    return data


def device_time_seconds(device_us: pd.Series) -> np.ndarray:
    values = device_us.astype(np.uint64).to_numpy()
    unwrapped = values.astype(np.float64).copy()
    offset = 0.0
    previous = float(values[0])
    for idx, value in enumerate(values):
        current = float(value)
        if idx > 0 and current < previous:
            offset += float(2**32)
        unwrapped[idx] = current + offset
        previous = current
    return (unwrapped - unwrapped[0]) / 1_000_000.0


def robust_spike_rate(values: np.ndarray) -> float:
    diffs = np.diff(values.astype(float))
    if diffs.size == 0:
        return float("nan")
    median = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - median)))
    threshold = max(5.0, abs(median) + 6.0 * 1.4826 * mad)
    return float(np.mean(np.abs(diffs - median) > threshold))


def slope_per_minute(times_s: np.ndarray, values: np.ndarray) -> float:
    if len(times_s) < 3:
        return float("nan")
    step = max(1, len(times_s) // 1200)
    x = times_s[::step]
    y = values.astype(float)[::step]
    if len(x) < 3 or np.allclose(x, x[0]):
        return float("nan")
    try:
        from scipy import stats

        slope = stats.theilslopes(y, x).slope
    except Exception:
        slope = np.polyfit(x, y, deg=1)[0]
    return float(slope * 60.0)


def summarize(data: pd.DataFrame, output_dir: str | Path, min_drift_s: float = 30.0) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    noise_rows = []
    jitter_rows = []
    drift_rows = []

    for (condition, source_file), group in data.groupby(["condition", "source_file"], sort=False):
        group = group.sort_values("host_time_s" if "host_time_s" in group.columns else "device_us").reset_index(drop=True)
        t = device_time_seconds(group["device_us"])
        intervals_ms = np.diff(t) * 1000.0
        if intervals_ms.size:
            jitter_rows.append(
                {
                    "condition": condition,
                    "source_file": source_file,
                    "n_samples": len(group),
                    "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
                    "mean_interval_ms": float(np.mean(intervals_ms)),
                    "std_interval_ms": float(np.std(intervals_ms, ddof=1)) if intervals_ms.size > 1 else 0.0,
                    "min_interval_ms": float(np.min(intervals_ms)),
                    "max_interval_ms": float(np.max(intervals_ms)),
                    "p95_interval_ms": float(np.percentile(intervals_ms, 95)),
                    "p05_interval_ms": float(np.percentile(intervals_ms, 5)),
                    "effective_sampling_rate_hz": float(1000.0 / np.mean(intervals_ms)),
                }
            )

        edge = min(max(int(round(10 * 50)), 10), max(len(group) // 4, 10))
        for channel in CHANNELS:
            values = group[channel].to_numpy(dtype=float)
            noise_rows.append(
                {
                    "condition": condition,
                    "source_file": source_file,
                    "channel": channel,
                    "mean_adc": float(np.mean(values)),
                    "std_adc": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "peak_to_peak_adc": float(np.max(values) - np.min(values)),
                    "spike_dropout_rate_pct": 100.0 * robust_spike_rate(values),
                    "range_endpoint_rate_pct": 100.0 * float(np.mean((values <= 0) | (values >= 1023))),
                }
            )
            duration_s = float(t[-1] - t[0]) if len(t) > 1 else 0.0
            if len(group) >= 20 and duration_s >= min_drift_s:
                start_med = float(np.median(values[:edge]))
                end_med = float(np.median(values[-edge:]))
                drift_rows.append(
                    {
                        "condition": condition,
                        "source_file": source_file,
                        "channel": channel,
                        "duration_s": duration_s,
                        "slope_adc_per_min": slope_per_minute(t, values),
                        "baseline_shift_adc": end_med - start_med,
                        "start_median_adc": start_med,
                        "end_median_adc": end_med,
                    }
                )

    paths = {
        "noise": out_dir / "hardware_noise_summary.csv",
        "jitter": out_dir / "hardware_jitter_summary.csv",
        "drift": out_dir / "hardware_drift_summary.csv",
        "combined": out_dir / "hardware_disturbance_summary.json",
        "tex": out_dir / "hardware_disturbance_table.tex",
    }
    noise = pd.DataFrame(noise_rows)
    jitter = pd.DataFrame(jitter_rows)
    drift = pd.DataFrame(drift_rows)
    noise.to_csv(paths["noise"], index=False)
    jitter.to_csv(paths["jitter"], index=False)
    drift.to_csv(paths["drift"], index=False)

    combined = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "notes": [
            "ADC values use the existing project convention 1023 - analogRead(raw).",
            "Spike/dropout rate is based on robust first-difference outliers.",
            "Range endpoint rate records samples at ADC code 0 or 1023 as descriptive operating-point information.",
            f"Drift slope is reported only for captures >= {min_drift_s:g} s.",
        ],
        "noise_overall": grouped_mean(noise, ["std_adc", "peak_to_peak_adc", "spike_dropout_rate_pct"]),
        "jitter_overall": grouped_mean(jitter, ["mean_interval_ms", "std_interval_ms", "effective_sampling_rate_hz"]),
        "drift_overall": grouped_mean(drift, ["slope_adc_per_min", "baseline_shift_adc"]),
    }
    paths["combined"].write_text(json.dumps(combined, indent=2), encoding="utf-8")
    paths["tex"].write_text(render_latex(noise, jitter, drift), encoding="utf-8")
    return paths


def grouped_mean(frame: pd.DataFrame, columns: list[str]) -> dict:
    if frame.empty or "condition" not in frame.columns:
        return {}
    return frame.groupby("condition")[columns].mean().to_dict()


def mean_pm_std(values: pd.Series, precision: int = 2) -> str:
    clean = values.dropna().astype(float)
    if clean.empty:
        return "--"
    mean = clean.mean()
    std = clean.std(ddof=1) if len(clean) > 1 else 0.0
    if math.isnan(std):
        std = 0.0
    return f"{mean:.{precision}f} $\\pm$ {std:.{precision}f}"


def render_latex(noise: pd.DataFrame, jitter: pd.DataFrame, drift: pd.DataFrame) -> str:
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Condition & Noise std. (ADC) & Spike/dropout (\\%) & Mean interval (ms) & Jitter std. (ms) & Drift slope (ADC/min) \\\\",
        "\\midrule",
    ]
    conditions = sorted(set(noise.get("condition", [])) | set(jitter.get("condition", [])) | set(drift.get("condition", [])))
    for condition in conditions:
        n = mean_pm_std(noise.loc[noise["condition"] == condition, "std_adc"]) if not noise.empty else "--"
        s = mean_pm_std(noise.loc[noise["condition"] == condition, "spike_dropout_rate_pct"]) if not noise.empty else "--"
        m = mean_pm_std(jitter.loc[jitter["condition"] == condition, "mean_interval_ms"]) if not jitter.empty else "--"
        j = mean_pm_std(jitter.loc[jitter["condition"] == condition, "std_interval_ms"]) if not jitter.empty else "--"
        d = mean_pm_std(drift.loc[drift["condition"] == condition, "slope_adc_per_min"]) if not drift.empty else "--"
        lines.append(f"{condition} & {n} & {s} & {m} & {j} & {d} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser("capture", help="Capture one condition from the Arduino serial stream.")
    capture_parser.add_argument("--condition", required=True, help="Condition label, e.g. static or bend.")
    capture_parser.add_argument("--duration-s", type=float, default=120.0)
    capture_parser.add_argument("--settle-s", type=float, default=2.0)
    capture_parser.add_argument("--port", default=None)
    capture_parser.add_argument("--baud", type=int, default=115200)
    capture_parser.add_argument("--output-dir", default="outputs/hardware_disturbance")
    capture_parser.add_argument(
        "--stop-mode",
        choices=["device", "host"],
        default="device",
        help="Stop by Arduino device timestamp span (default) or host wall-clock time.",
    )

    analyze_parser = subparsers.add_parser("analyze", help="Summarize one or more captured CSV files.")
    analyze_parser.add_argument("inputs", nargs="+")
    analyze_parser.add_argument("--output-dir", default="outputs/hardware_disturbance")
    analyze_parser.add_argument("--min-drift-s", type=float, default=30.0)

    args = parser.parse_args()
    if args.command == "capture":
        capture(args)
    elif args.command == "analyze":
        paths = summarize(load_frames(args.inputs), args.output_dir, min_drift_s=args.min_drift_s)
        for name, path in paths.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
