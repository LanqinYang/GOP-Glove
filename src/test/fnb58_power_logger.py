#!/usr/bin/env python3
"""
Log FNIRSI FNB58 USB power measurements to CSV.

The FNB58 exposes measurement packets through a HID interface on the PC
Micro-USB port. This script records voltage/current samples, computes power
and host-side energy integration, and writes data suitable for the embedded
power table in the paper revision.

Usage:
  python src/test/fnb58_power_logger.py --duration 30 --tag idle \
    --output outputs/power_measurements/idle.csv

Dependencies:
  pip install hidapi pyusb

Notes:
  - Sampling is 100 Hz from the meter. Use --decimate to write fewer rows.
  - On Linux, USB HID permissions may require sudo or a udev rule.
  - On macOS, install libusb if pyusb cannot find a backend.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import hid
except ImportError:
    hid = None

try:
    import usb.core
    import usb.util
except ImportError:
    usb = None


VID_FNB58 = 0x2E3C
PID_FNB58_CANDIDATES = (0x5558, 0x5555)
SAMPLES_PER_PACKET = 4
SAMPLE_RATE_HZ = 100.0
SAMPLE_INTERVAL_S = 1.0 / SAMPLE_RATE_HZ
PACKET_SIZE = 64


@dataclass
class Sample:
    host_time_s: float
    sample_in_packet: int
    voltage_v: float
    current_a: float
    power_w: float
    dp_v: float
    dn_v: float
    temp_c: float
    energy_j: float
    capacity_c: float


def find_pyusb_device():
    if usb is None:
        return None, None
    for pid in PID_FNB58_CANDIDATES:
        dev = usb.core.find(idVendor=VID_FNB58, idProduct=pid)
        if dev is not None:
            return dev, pid
    return None, None


def find_hid_device_path():
    if hid is None:
        return None, None
    for dev in hid.enumerate():
        if dev["vendor_id"] != VID_FNB58:
            continue
        if dev["product_id"] in PID_FNB58_CANDIDATES:
            return dev["path"], dev["product_id"]
    return None, None


def list_devices() -> None:
    print("HID devices:")
    if hid is None:
        print("  hidapi is not installed.")
    else:
        for dev in hid.enumerate():
            print(
                "  "
                f"VID=0x{dev['vendor_id']:04x} "
                f"PID=0x{dev['product_id']:04x} "
                f"manufacturer={dev.get('manufacturer_string')!r} "
                f"product={dev.get('product_string')!r}"
            )

    print("USB/libusb devices:")
    if usb is None:
        print("  pyusb is not installed.")
    else:
        try:
            devs = list(usb.core.find(find_all=True))
        except usb.core.NoBackendError:
            print("  pyusb has no libusb backend.")
        else:
            for dev in devs:
                print(f"  VID=0x{dev.idVendor:04x} PID=0x{dev.idProduct:04x}")


class HidBackend:
    def __init__(self, path, pid, *, report_mode: str):
        self.dev = hid.device()
        self.report_mode = report_mode
        try:
            self.dev.open_path(path)
        except OSError:
            self.dev.open(VID_FNB58, pid)

    def write(self, payload: bytes) -> None:
        if self.report_mode == "prefix0":
            # hidapi commonly uses byte 0 as report ID on macOS.
            self.dev.write(b"\x00" + payload)
        elif self.report_mode == "plain":
            # Some HID stacks expose the interrupt report without an added ID byte.
            self.dev.write(payload)
        else:
            raise ValueError(f"Unsupported HID report mode: {self.report_mode}")

    def read(self, size: int, timeout_ms: int):
        data = self.dev.read(size, timeout_ms=timeout_ms)
        if not data:
            raise TimeoutError("Timed out waiting for FNB58 HID data.")
        return data

    def close(self) -> None:
        self.dev.close()


class PyUsbBackend:
    def __init__(self, dev):
        dev.reset()
        detach_kernel_drivers_if_needed(dev)
        dev.set_configuration()
        hid_interface = find_hid_interface(dev)
        self.ep_in, self.ep_out = endpoint_pair(dev, hid_interface)

    def write(self, payload: bytes) -> None:
        self.ep_out.write(payload)

    def read(self, size: int, timeout_ms: int):
        return self.ep_in.read(size_or_buffer=size, timeout=timeout_ms)

    def close(self) -> None:
        return None


def connect_backend(backend_name: str, *, hid_report_mode: str):
    if backend_name in ("auto", "hid"):
        path, pid = find_hid_device_path()
        if path is not None:
            return HidBackend(path, pid, report_mode=hid_report_mode), "hid", pid
        if backend_name == "hid":
            return None, None, None

    if backend_name in ("auto", "pyusb"):
        dev, pid = find_pyusb_device()
        if dev is not None:
            return PyUsbBackend(dev), "pyusb", pid

    return None, None, None


def find_hid_interface(dev):
    for cfg in dev:
        for interface in cfg:
            if interface.bInterfaceClass == 0x03:
                return interface
    raise RuntimeError("No HID interface found on FNB58.")


def detach_kernel_drivers_if_needed(dev) -> None:
    for cfg in dev:
        for interface in cfg:
            try:
                if dev.is_kernel_driver_active(interface.bInterfaceNumber):
                    dev.detach_kernel_driver(interface.bInterfaceNumber)
            except (NotImplementedError, usb.core.USBError):
                pass


def endpoint_pair(dev, interface):
    cfg = dev.get_active_configuration()
    intf = cfg[(interface.bInterfaceNumber, 0)]
    ep_in = usb.util.find_descriptor(
        intf,
        custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress)
        == usb.util.ENDPOINT_IN,
    )
    ep_out = usb.util.find_descriptor(
        intf,
        custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress)
        == usb.util.ENDPOINT_OUT,
    )
    if ep_in is None or ep_out is None:
        raise RuntimeError("Could not find HID IN/OUT endpoints.")
    return ep_in, ep_out


def request_stream(backend, *, is_fnb58: bool, init_mode: str) -> None:
    if init_mode == "none":
        return

    # Initialization sequence used by FNB58-compatible PC logging tools.
    backend.write(b"\xaa\x81" + b"\x00" * 61 + b"\x8e")
    if init_mode == "aa81-only":
        return

    backend.write(b"\xaa\x82" + b"\x00" * 61 + b"\x96")
    if init_mode == "aa81-aa82":
        return

    if is_fnb58:
        backend.write(b"\xaa\x82" + b"\x00" * 61 + b"\x96")
    else:
        backend.write(b"\xaa\x83" + b"\x00" * 61 + b"\x9e")

    if init_mode == "with-aa83":
        backend.write(b"\xaa\x83" + b"\x00" * 61 + b"\x9e")


def keepalive(backend) -> None:
    backend.write(b"\xaa\x83" + b"\x00" * 61 + b"\x9e")


def u32_le(data, offset: int) -> int:
    return (
        data[offset]
        | (data[offset + 1] << 8)
        | (data[offset + 2] << 16)
        | (data[offset + 3] << 24)
    )


def u16_le(data, offset: int) -> int:
    return data[offset] | (data[offset + 1] << 8)


def decode_packet(
    data,
    *,
    energy_j: float,
    capacity_c: float,
    temp_ema: float | None,
    temp_alpha: float,
) -> tuple[list[Sample], float, float, float | None]:
    if len(data) != PACKET_SIZE or data[1] != 0x04:
        return [], energy_j, capacity_c, temp_ema

    packet_time0 = time.time() - SAMPLES_PER_PACKET * SAMPLE_INTERVAL_S
    samples: list[Sample] = []

    for i in range(SAMPLES_PER_PACKET):
        offset = 2 + 15 * i
        voltage_v = u32_le(data, offset) / 100000.0
        current_a = u32_le(data, offset + 4) / 100000.0
        dp_v = u16_le(data, offset + 8) / 1000.0
        dn_v = u16_le(data, offset + 10) / 1000.0
        temp_c_raw = u16_le(data, offset + 13) / 10.0

        if temp_ema is None:
            temp_ema = temp_c_raw
        else:
            temp_ema = temp_c_raw * (1.0 - temp_alpha) + temp_ema * temp_alpha

        power_w = voltage_v * current_a
        energy_j += power_w * SAMPLE_INTERVAL_S
        capacity_c += current_a * SAMPLE_INTERVAL_S

        samples.append(
            Sample(
                host_time_s=packet_time0 + i * SAMPLE_INTERVAL_S,
                sample_in_packet=i,
                voltage_v=voltage_v,
                current_a=current_a,
                power_w=power_w,
                dp_v=dp_v,
                dn_v=dn_v,
                temp_c=temp_ema,
                energy_j=energy_j,
                capacity_c=capacity_c,
            )
        )

    return samples, energy_j, capacity_c, temp_ema


def summarize(samples: Iterable[Sample]) -> dict[str, float | int | None]:
    rows = list(samples)
    if not rows:
        return {
            "samples": 0,
            "duration_s": 0.0,
            "avg_voltage_V": None,
            "avg_current_A": None,
            "avg_power_W": None,
            "max_power_W": None,
            "energy_J": 0.0,
        }

    duration_s = max(rows[-1].host_time_s - rows[0].host_time_s, SAMPLE_INTERVAL_S)
    avg_voltage = sum(s.voltage_v for s in rows) / len(rows)
    avg_current = sum(s.current_a for s in rows) / len(rows)
    avg_power = sum(s.power_w for s in rows) / len(rows)
    max_power = max(s.power_w for s in rows)
    return {
        "samples": len(rows),
        "duration_s": duration_s,
        "avg_voltage_V": avg_voltage,
        "avg_current_A": avg_current,
        "avg_power_W": avg_power,
        "max_power_W": max_power,
        "energy_J": rows[-1].energy_j,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record FNIRSI FNB58 voltage/current/power samples to CSV."
    )
    parser.add_argument("--output", help="CSV output path.")
    parser.add_argument("--summary", help="Optional JSON summary output path.")
    parser.add_argument("--duration", type=float, help="Recording duration in seconds.")
    parser.add_argument("--tag", default="measurement", help="Experiment tag written to CSV.")
    parser.add_argument(
        "--decimate",
        type=int,
        default=1,
        help="Write every Nth sample while still integrating all samples.",
    )
    parser.add_argument(
        "--temp-alpha",
        type=float,
        default=0.9,
        help="EMA smoothing factor for temperature.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print device details.")
    parser.add_argument(
        "--backend",
        choices=("auto", "hid", "pyusb"),
        default="auto",
        help="Device access backend. macOS usually works best with hid.",
    )
    parser.add_argument(
        "--hid-report-mode",
        choices=("prefix0", "plain"),
        default="plain",
        help="How to format HID writes on macOS.",
    )
    parser.add_argument(
        "--init-mode",
        choices=("fnb58", "with-aa83", "aa81-aa82", "aa81-only", "none"),
        default="fnb58",
        help="FNB58 stream initialization sequence to try.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait until an FNB58 appears instead of exiting immediately.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print HID/libusb devices and exit.",
    )
    parser.add_argument(
        "--max-empty-reads",
        type=int,
        default=6,
        help="Stop after this many consecutive 5s read timeouts.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Open the device, run the selected init sequence, try one read, then exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.decimate < 1:
        raise SystemExit("--decimate must be >= 1.")

    if args.list_devices:
        list_devices()
        return 0

    if not args.output:
        raise SystemExit("--output is required unless --list-devices is used.")

    backend = None
    backend_used = None
    pid = None
    while backend is None:
        backend, backend_used, pid = connect_backend(
            args.backend,
            hid_report_mode=args.hid_report_mode,
        )
        if backend is not None or not args.wait:
            break
        print(
            "FNB58 not found yet. Connect the PC Micro-USB port, enable PC mode, "
            "and press Ctrl-C to stop waiting.",
            file=sys.stderr,
        )
        time.sleep(2.0)

    if backend is None:
        raise SystemExit(
            "FNB58 not found. Connect the FNB58 PC Micro-USB port, use a data cable, "
            "and enable PC communication mode if your unit has that switch/menu."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")

    if args.verbose:
        print(
            f"Found FNB58 VID=0x{VID_FNB58:04x} PID=0x{pid:04x} backend={backend_used}",
            file=sys.stderr,
        )

    request_stream(
        backend,
        is_fnb58=(pid == PID_FNB58_CANDIDATES[0]),
        init_mode=args.init_mode,
    )
    time.sleep(0.1)

    if args.probe:
        try:
            data = backend.read(size=PACKET_SIZE, timeout_ms=1500)
        except TimeoutError as exc:
            print(f"Probe: no data ({exc}).")
        else:
            print("Probe: received", len(data), "bytes:", bytes(data).hex())
        finally:
            backend.close()
        return 0

    fieldnames = [
        "tag",
        "host_time_s",
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

    started = time.time()
    next_keepalive = started + 1.0
    energy_j = 0.0
    capacity_c = 0.0
    temp_ema = None
    seen = 0
    empty_reads = 0
    kept_samples: list[Sample] = []

    try:
        with output_path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()

            while True:
                if args.duration is not None and time.time() - started >= args.duration:
                    break

                try:
                    data = backend.read(size=PACKET_SIZE, timeout_ms=5000)
                except TimeoutError as exc:
                    empty_reads += 1
                    print(
                        f"Warning: {exc} ({empty_reads}/{args.max_empty_reads}); "
                        "sending keepalive.",
                        file=sys.stderr,
                    )
                    keepalive(backend)
                    if empty_reads >= args.max_empty_reads:
                        raise SystemExit(
                            "FNB58 is connected but did not stream measurement packets. "
                            "Check that PC communication mode is enabled, replug the PC "
                            "Micro-USB cable, or power-cycle the FNB58."
                        )
                    continue

                empty_reads = 0
                samples, energy_j, capacity_c, temp_ema = decode_packet(
                    data,
                    energy_j=energy_j,
                    capacity_c=capacity_c,
                    temp_ema=temp_ema,
                    temp_alpha=args.temp_alpha,
                )

                for sample in samples:
                    seen += 1
                    kept_samples.append(sample)
                    if (seen - 1) % args.decimate != 0:
                        continue
                    writer.writerow(
                        {
                            "tag": args.tag,
                            "host_time_s": f"{sample.host_time_s:.6f}",
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

    except KeyboardInterrupt:
        print("Stopped by user.", file=sys.stderr)
    finally:
        backend.close()

    summary = summarize(kept_samples)
    summary.update({"tag": args.tag, "output": str(output_path), "decimate": args.decimate})
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
