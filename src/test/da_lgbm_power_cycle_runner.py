#!/usr/bin/env python3
"""
Coordinate DA-LGBM Arduino cycles with FNB58 power logging.

This runner:
  - opens the Arduino serial port,
  - opens the FNB58 HID stream,
  - records power at 100 Hz,
  - sends 's' to trigger repeated DA-LGBM recognition cycles,
  - parses PHASE markers and latency lines from BSL_Gesture_Demo.ino,
  - writes power/events/latency/phase-summary files.

Before running, upload:
  arduino/tinyml_inference/ADANN_LightGBM_inference/BSL_Gesture_Demo
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import re
import statistics
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import serial
import serial.tools.list_ports

from fnb58_power_logger import (
    PACKET_SIZE,
    PID_FNB58_CANDIDATES,
    connect_backend,
    decode_packet,
    keepalive,
    request_stream,
    summarize,
)


LATENCY_RE = re.compile(
    r"windowing=(?P<windowing>[0-9.]+).*?"
    r"features=(?P<features>[0-9.]+).*?"
    r"standardize=(?P<standardize>[0-9.]+).*?"
    r"ADANN=(?P<adann>[0-9.]+).*?"
    r"LGBM=(?P<lgbm>[0-9.]+).*?"
    r"gate=(?P<gate>[0-9.]+)"
)
PRED_RE = re.compile(r"Predicted = (?P<label_id>\d+) \((?P<label>[^)]+)\)")


def auto_arduino_port() -> str:
    ports = list(serial.tools.list_ports.comports())
    preferred = []
    for p in ports:
        text = " ".join(str(x) for x in (p.device, p.description, p.manufacturer, p.product))
        if "Nano 33 BLE" in text or "Arduino" in text:
            preferred.append(p.device)
    if preferred:
        return preferred[0]
    usb_ports = [p.device for p in ports if "usbmodem" in p.device]
    if usb_ports:
        return usb_ports[0]
    raise SystemExit("No Arduino serial port found. Pass --arduino-port explicitly.")


def serial_reader(ser: serial.Serial, out_queue: queue.Queue, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            raw = ser.readline()
        except serial.SerialException as exc:
            out_queue.put(("error", time.time(), str(exc)))
            return
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").strip()
        out_queue.put(("line", time.time(), line))


def power_reader(
    *,
    output_path: Path,
    stop_event: threading.Event,
    tag: str,
    backend_name: str,
    hid_report_mode: str,
    init_mode: str,
    samples_out: list,
    errors_out: list,
) -> None:
    try:
        backend, backend_used, pid = connect_backend(backend_name, hid_report_mode=hid_report_mode)
        if backend is None:
            raise RuntimeError("FNB58 not found.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        request_stream(backend, is_fnb58=(pid == PID_FNB58_CANDIDATES[0]), init_mode=init_mode)
        time.sleep(0.1)
    except Exception as exc:
        errors_out.append(str(exc))
        return

    fieldnames = [
        "tag",
        "host_time_s",
        "rel_time_s",
        "sample_in_packet",
        "voltage_V",
        "current_A",
        "power_W",
        "dp_V",
        "dn_V",
        "temp_C",
        "energy_J",
        "capacity_C",
    ]
    t0 = time.time()
    next_keepalive = t0 + 1.0
    energy_j = 0.0
    capacity_c = 0.0
    temp_ema = None

    try:
        with output_path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            while not stop_event.is_set():
                try:
                    data = backend.read(size=PACKET_SIZE, timeout_ms=1000)
                except TimeoutError:
                    keepalive(backend)
                    continue
                except Exception as exc:
                    errors_out.append(str(exc))
                    return

                decoded, energy_j, capacity_c, temp_ema = decode_packet(
                    data,
                    energy_j=energy_j,
                    capacity_c=capacity_c,
                    temp_ema=temp_ema,
                    temp_alpha=0.9,
                )
                for sample in decoded:
                    samples_out.append(sample)
                    writer.writerow(
                        {
                            "tag": tag,
                            "host_time_s": f"{sample.host_time_s:.6f}",
                            "rel_time_s": f"{sample.host_time_s - t0:.6f}",
                            "sample_in_packet": sample.sample_in_packet,
                            "voltage_V": f"{sample.voltage_v:.5f}",
                            "current_A": f"{sample.current_a:.5f}",
                            "power_W": f"{sample.power_w:.6f}",
                            "dp_V": f"{sample.dp_v:.3f}",
                            "dn_V": f"{sample.dn_v:.3f}",
                            "temp_C": f"{sample.temp_c:.3f}",
                            "energy_J": f"{sample.energy_j:.9f}",
                            "capacity_C": f"{sample.capacity_c:.9f}",
                        }
                    )

                if time.time() >= next_keepalive:
                    keepalive(backend)
                    next_keepalive = time.time() + 1.0
    finally:
        backend.close()


def wait_for_ready(line_queue: queue.Queue, events: list, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            kind, ts, payload = line_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if kind == "line":
            events.append({"host_time_s": ts, "kind": "serial", "line": payload})
            print(payload)
            if "Press 's'" in payload or "--- Ready ---" in payload or "BSL Fusion Ready" in payload:
                return
        else:
            raise RuntimeError(payload)
    raise TimeoutError("Timed out waiting for Arduino ready prompt.")


def parse_serial_line(line: str, ts: float, events: list, latencies: list, current_cycle: int) -> int:
    event = {"host_time_s": ts, "kind": "serial", "cycle": current_cycle, "line": line}
    if line.startswith("PHASE,"):
        parts = line.split(",")
        if len(parts) >= 4:
            phase = parts[1]
            edge = parts[2]
            arduino_ms = int(parts[3])
            if phase == "cycle" and edge == "start":
                current_cycle += 1
            event.update(
                {
                    "kind": "phase",
                    "cycle": current_cycle,
                    "phase": phase,
                    "edge": edge,
                    "arduino_ms": arduino_ms,
                }
            )

    m = LATENCY_RE.search(line)
    if m:
        row = {"host_time_s": ts, "cycle": current_cycle}
        row.update({k + "_ms": float(v) for k, v in m.groupdict().items()})
        row["feature_inference_ms"] = (
            row["features_ms"]
            + row["standardize_ms"]
            + row["adann_ms"]
            + row["lgbm_ms"]
            + row["gate_ms"]
        )
        row["full_cycle_ms"] = row["windowing_ms"] + row["feature_inference_ms"]
        latencies.append(row)

    p = PRED_RE.search(line)
    if p:
        event.update(
            {
                "kind": "prediction",
                "cycle": current_cycle,
                "label_id": int(p.group("label_id")),
                "label": p.group("label"),
            }
        )

    events.append(event)
    return current_cycle


def write_events(path: Path, events: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "host_time_s",
        "rel_time_s",
        "kind",
        "cycle",
        "phase",
        "edge",
        "arduino_ms",
        "label_id",
        "label",
        "line",
    ]
    t0 = events[0]["host_time_s"] if events else time.time()
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for event in events:
            row = {field: event.get(field, "") for field in fields}
            row["rel_time_s"] = f"{event['host_time_s'] - t0:.6f}"
            writer.writerow(row)


def write_latencies(path: Path, latencies: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "host_time_s",
        "cycle",
        "windowing_ms",
        "features_ms",
        "standardize_ms",
        "adann_ms",
        "lgbm_ms",
        "gate_ms",
        "feature_inference_ms",
        "full_cycle_ms",
    ]
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(latencies)


def summarize_phases(events: list, samples: list, path: Path) -> list:
    phase_starts = {}
    rows = []
    for event in events:
        if event.get("kind") != "phase":
            continue
        key = (event.get("cycle"), event.get("phase"))
        if event.get("edge") == "start":
            phase_starts[key] = event["host_time_s"]
        elif event.get("edge") == "end" and key in phase_starts:
            start = phase_starts.pop(key)
            end = event["host_time_s"]
            ps = [s.power_w for s in samples if start <= s.host_time_s <= end]
            vs = [s.voltage_v for s in samples if start <= s.host_time_s <= end]
            is_ = [s.current_a for s in samples if start <= s.host_time_s <= end]
            if not ps:
                continue
            rows.append(
                {
                    "cycle": event.get("cycle"),
                    "phase": event.get("phase"),
                    "start_host_time_s": start,
                    "end_host_time_s": end,
                    "duration_s": end - start,
                    "samples": len(ps),
                    "avg_voltage_V": statistics.mean(vs),
                    "avg_current_A": statistics.mean(is_),
                    "avg_power_W": statistics.mean(ps),
                    "max_power_W": max(ps),
                    "energy_J": sum(ps) * 0.01,
                }
            )

    fields = [
        "cycle",
        "phase",
        "start_host_time_s",
        "end_host_time_s",
        "duration_s",
        "samples",
        "avg_voltage_V",
        "avg_current_A",
        "avg_power_W",
        "max_power_W",
        "energy_J",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run aligned DA-LGBM power-cycle tests.")
    parser.add_argument("--arduino-port", default=None, help="Arduino serial port.")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--baseline-s", type=float, default=5.0)
    parser.add_argument("--between-s", type=float, default=1.0)
    parser.add_argument("--cycle-timeout-s", type=float, default=15.0)
    parser.add_argument("--tag", default="da_lgbm_aligned")
    parser.add_argument(
        "--output-dir",
        default="outputs/power_measurements",
        help="Directory for CSV/JSON outputs.",
    )
    parser.add_argument("--backend", choices=("auto", "hid", "pyusb"), default="auto")
    parser.add_argument("--hid-report-mode", choices=("plain", "prefix0"), default="plain")
    parser.add_argument("--init-mode", default="fnb58")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    stem = args.tag
    power_path = out_dir / f"{stem}_power.csv"
    events_path = out_dir / f"{stem}_events.csv"
    latency_path = out_dir / f"{stem}_latency.csv"
    phases_path = out_dir / f"{stem}_phase_summary.csv"
    summary_path = out_dir / f"{stem}_summary.json"

    arduino_port = args.arduino_port or auto_arduino_port()
    print(f"Arduino port: {arduino_port}")
    print(f"Power CSV: {power_path}")

    stop_power = threading.Event()
    stop_serial = threading.Event()
    serial_queue: queue.Queue = queue.Queue()
    samples = []
    power_errors = []
    events = []
    latencies = []
    cycle = 0

    power_thread = threading.Thread(
        target=power_reader,
        kwargs={
            "output_path": power_path,
            "stop_event": stop_power,
            "tag": args.tag,
            "backend_name": args.backend,
            "hid_report_mode": args.hid_report_mode,
            "init_mode": args.init_mode,
            "samples_out": samples,
            "errors_out": power_errors,
        },
        daemon=True,
    )
    power_thread.start()
    power_ready_deadline = time.time() + 8.0
    while time.time() < power_ready_deadline and not samples and not power_errors:
        time.sleep(0.1)
    if power_errors:
        stop_power.set()
        power_thread.join(timeout=2.0)
        raise RuntimeError(f"FNB58 power logger failed before test start: {power_errors[-1]}")
    if not samples:
        stop_power.set()
        power_thread.join(timeout=2.0)
        raise RuntimeError("FNB58 produced no samples before test start; replug/power-cycle it.")

    try:
        with serial.Serial(arduino_port, args.baud, timeout=0.2) as ser:
            reader = threading.Thread(
                target=serial_reader,
                args=(ser, serial_queue, stop_serial),
                daemon=True,
            )
            reader.start()
            try:
                wait_for_ready(serial_queue, events, timeout_s=5.0)
            except TimeoutError:
                print("No fresh ready prompt seen; continuing because the sketch may already be waiting.")
            print(f"Baseline for {args.baseline_s:.1f}s...")
            time.sleep(args.baseline_s)

            for target_cycle in range(1, args.cycles + 1):
                print(f"Trigger cycle {target_cycle}/{args.cycles}")
                cycle = target_cycle
                ser.write(b"s")
                deadline = time.time() + args.cycle_timeout_s
                ended = False
                while time.time() < deadline:
                    try:
                        kind, ts, payload = serial_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if kind == "error":
                        raise RuntimeError(payload)
                    print(payload)
                    cycle = parse_serial_line(payload, ts, events, latencies, cycle)
                    if payload.startswith("PHASE,cycle,end") or LATENCY_RE.search(payload):
                        ended = True
                        break
                if not ended:
                    raise TimeoutError(f"Cycle {target_cycle} did not finish in time.")
                time.sleep(args.between_s)

            ser.write(b"R\n")
            drain_until = time.time() + 3.0
            while time.time() < drain_until:
                try:
                    kind, ts, payload = serial_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if kind == "line":
                    print(payload)
                    cycle = parse_serial_line(payload, ts, events, latencies, cycle)

    finally:
        stop_serial.set()
        stop_power.set()
        power_thread.join(timeout=3.0)

    write_events(events_path, events)
    write_latencies(latency_path, latencies)
    phase_rows = summarize_phases(events, samples, phases_path)

    summary = summarize(samples)
    summary.update(
        {
            "tag": args.tag,
            "cycles_requested": args.cycles,
            "latency_rows": len(latencies),
            "phase_rows": len(phase_rows),
            "power_csv": str(power_path),
            "events_csv": str(events_path),
            "latency_csv": str(latency_path),
            "phase_summary_csv": str(phases_path),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
