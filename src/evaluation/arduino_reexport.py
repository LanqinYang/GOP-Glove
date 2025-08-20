#!/usr/bin/env python3
"""
Arduino header re-export utility.

Re-exports LightGBM (and ADANN, if present) headers from an existing
ADANN_LightGBM model package without retraining.

Usage examples:

python -m src.evaluation.arduino_reexport \
  --package models/trained/ADANN_LightGBM/standard/full/ADANN_LightGBM_standard_20250101_120000.pth \
  --mode standard \
  --export_pure_lgb

python -m src.evaluation.arduino_reexport \
  --package models/trained/ADANN_LightGBM/loso/arduino/ADANN_LightGBM_loso_final_20250101_120000.pth \
  --mode loso_final \
  --export_pure_lgb
"""

import argparse
import os
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Re-export Arduino headers from an ADANN_LightGBM package")
    parser.add_argument("--package", required=True, help="Path to .pth or .keras package of ADANN_LightGBM")
    parser.add_argument("--mode", default="standard", choices=["standard", "loso", "loso_final"], help="Copy target mode for Arduino dirs")
    parser.add_argument("--export_pure_lgb", action="store_true", help="Also export canonical bsl_model_LightGBM.h into LightGBM_Arduino_inference")

    args = parser.parse_args()

    pkg_path = args.package
    if pkg_path.endswith(".keras"):
        # convert to .pth alongside
        pkg_path = pkg_path.replace(".keras", ".pth")

    if not os.path.isfile(pkg_path):
        print(f"❌ Package file not found: {pkg_path}")
        return 1

    # Lazy imports to avoid heavy deps if not needed
    try:
        import torch
    except ImportError:
        print("❌ PyTorch is required to load the ADANN_LightGBM package")
        return 1

    # Allow safe globals for custom classes if needed
    try:
        from src.training.train_adann import AdversarialFeatureExtractor
        torch.serialization.add_safe_globals([AdversarialFeatureExtractor])
    except Exception:
        pass

    # Load package
    try:
        try:
            package = torch.load(pkg_path, map_location="cpu", weights_only=False)
        except TypeError:
            package = torch.load(pkg_path, map_location="cpu")
    except Exception as e:
        print(f"❌ Failed to load package: {e}")
        return 1

    # Extract components
    lgb_estimator = package.get("lightgbm_model") or package.get("lightgbm")
    lgb_scaler = package.get("lgb_scaler")
    adann_model = package.get("adann_model") or package.get("adann")
    adann_scaler = package.get("adann_scaler")

    if lgb_estimator is None or lgb_scaler is None:
        print("⚠️ No LightGBM branch or scaler found in package. Skipping LightGBM header re-export.")
    if adann_model is None or adann_scaler is None:
        print("⚠️ No ADANN branch or scaler found in package. Skipping ADANN header re-export.")

    # Generate headers
    from src.training.pipeline import (
        generate_lightgbm_arduino_header,
        generate_adann_c_header_inline,
        copy_header_to_arduino_dir,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.dirname(os.path.abspath(pkg_path))

    # LightGBM header (hybrid path)
    if lgb_estimator is not None and lgb_scaler is not None:
        try:
            header_path = generate_lightgbm_arduino_header(
                lgb_estimator, lgb_scaler, "ADANN_LightGBM_LGB_BRANCH", timestamp, out_dir
            )
            print(f"✅ LightGBM header re-exported: {header_path}")
            # Copy into ADANN_LightGBM Arduino dirs
            copy_header_to_arduino_dir("ADANN_LightGBM", header_path, args.mode)
        except Exception as e:
            print(f"❌ Failed to generate/copy LightGBM header: {e}")

        # Optionally also export the canonical LightGBM header and copy into LightGBM_Arduino_inference
        if args.export_pure_lgb:
            try:
                pure_header = generate_lightgbm_arduino_header(
                    lgb_estimator, lgb_scaler, "LightGBM", timestamp, out_dir
                )
                print(f"✅ Canonical LightGBM header re-exported: {pure_header}")
                copy_header_to_arduino_dir("LightGBM", pure_header, args.mode)
            except Exception as e:
                print(f"❌ Failed to generate/copy canonical LightGBM header: {e}")

    # ADANN header (for hybrid path)
    if adann_model is not None and adann_scaler is not None:
        try:
            adann_header = generate_adann_c_header_inline(adann_model, adann_scaler, timestamp, out_dir)
            print(f"✅ ADANN header re-exported: {adann_header}")
            # Copy into ADANN dirs (canonical)
            copy_header_to_arduino_dir("ADANN", adann_header, args.mode)
            # Also mirror into ADANN_LightGBM dirs for combined inference
            try:
                import shutil
                mapping_dir = os.path.join("arduino", "tinyml_inference", "ADANN_LightGBM_inference")
                os.makedirs(mapping_dir, exist_ok=True)
                target_name = "bsl_model_ADANN.h"
                shutil.copyfile(adann_header, os.path.join(mapping_dir, target_name))
                for mode_dir in ["Latency_standard", "Latency_loso"]:
                    os.makedirs(os.path.join(mapping_dir, mode_dir), exist_ok=True)
                    shutil.copyfile(adann_header, os.path.join(mapping_dir, mode_dir, target_name))
                print(f"ADANN header mirrored to ADANN_LightGBM Arduino dirs: {mapping_dir}")
            except Exception as e:
                print(f"⚠️ Failed to mirror ADANN header to ADANN_LightGBM dirs: {e}")
        except Exception as e:
            print(f"❌ Failed to generate/copy ADANN header: {e}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


