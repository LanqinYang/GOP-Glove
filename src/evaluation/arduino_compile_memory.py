#!/usr/bin/env python3
"""
Arduino compile-time memory reporter (Flash/RAM) for LOSO latency sketches.

This script compiles each Latency_loso sketch and parses the compiler output to
extract:
  - Flash (Program storage space)
  - Static RAM (Global variables use)

Board: Arduino Nano 33 BLE Sense Rev2 (mbed core)
Requires: arduino-cli (https://arduino.github.io/arduino-cli/latest/)

Usage:
  python src/evaluation/arduino_compile_memory.py \
    --fqbn arduino:mbed_nano:nano33blesense \
    --export outputs/arduino_memory_compile_report.json

If arduino-cli is not found, the script will print a helpful message.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


SKETCHES = {
    "1D-CNN": "arduino/tinyml_inference/1D_CNN_inference/Latency_loso",
    "LightGBM": "arduino/tinyml_inference/LightGBM_Arduino_inference/Latency_loso",
    "XGBoost": "arduino/tinyml_inference/XGBoost_Arduino_inference/Latency_loso",
    "ADANN": "arduino/tinyml_inference/ADANN_inference/Latency_loso",
    "DA-LGBM": "arduino/tinyml_inference/ADANN_LightGBM_inference/Latency_loso",
    # Include Transformer-Encoder for parity with export/testing flow.
    # Compilation is expected to fail on current TFLite Micro, but we still capture stats/logs.
    "Transformer-Encoder": "arduino/tinyml_inference/Transformer_Encoder_inference/Latency_loso",
}


def run_compile(sketch_dir: str, fqbn: str) -> Tuple[int, str]:
    """Run arduino-cli compile and return (exit_code, combined_output)."""
    cmd = [
        "arduino-cli", "compile",
        "-b", fqbn,
        sketch_dir,
        "--clean",
        "--warnings", "none",
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return proc.returncode, proc.stdout
    except FileNotFoundError:
        return 127, "arduino-cli not found"


def parse_flash_ram(output: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse compiler output to extract (flash_bytes, ram_bytes)."""
    # Common patterns across cores
    # Example (mbed/nano33blesense):
    #   Sketch uses 123456 bytes (12%) of program storage space.
    #   Global variables use 7890 bytes (3%) of dynamic memory.
    m_flash = re.search(r"Sketch uses\s+(\d+)\s+bytes", output)
    if not m_flash:
        # AVR-like:
        #   Program storage space used: 123456 bytes
        m_flash = re.search(r"Program storage space used:\s*(\d+)\s*bytes", output)

    m_ram = re.search(r"Global variables use\s+(\d+)\s+bytes", output)
    if not m_ram:
        # Some cores may phrase as "variable"; keep it broad
        m_ram = re.search(r"variables use\s+(\d+)\s+bytes", output)

    flash = int(m_flash.group(1)) if m_flash else None
    ram = int(m_ram.group(1)) if m_ram else None
    return flash, ram


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fqbn", default="arduino:mbed_nano:nano33blesense",
                        help="Fully Qualified Board Name for arduino-cli compile")
    parser.add_argument("--export", default="outputs/arduino_memory_compile_report.json",
                        help="Path to write JSON report")
    args = parser.parse_args()

    if shutil.which("arduino-cli") is None:
        print("ERROR: arduino-cli not found. Install: brew install arduino-cli OR follow docs.")
        print("After installation, run: arduino-cli core install arduino:mbed_nano")
        sys.exit(127)

    report: Dict[str, Dict[str, object]] = {}
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    for name, rel_dir in SKETCHES.items():
        sketch_dir = os.path.join(root, rel_dir)
        if not os.path.isdir(sketch_dir):
            report[name] = {"error": f"Missing sketch dir: {sketch_dir}"}
            continue
        code, out = run_compile(sketch_dir, args.fqbn)
        flash, ram = parse_flash_ram(out)
        report[name] = {
            "sketch_dir": rel_dir,
            "compile_exit_code": code,
            "flash_bytes": flash,
            "ram_bytes": ram,
            "raw_output_tail": "\n".join(out.splitlines()[-30:]),  # helpful snippet
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.export)), exist_ok=True)
    with open(args.export, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote compile-time memory report to: {args.export}")


if __name__ == "__main__":
    main()



