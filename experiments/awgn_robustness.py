#!/usr/bin/env python3
"""
Controlled AWGN / ADC-like noise robustness experiment for LOSO evaluation.

The script trains each model using clean LOSO training subjects, injects
additive white Gaussian noise only into the held-out test windows, and reports
Macro-F1 degradation at clean, 20 dB, 10 dB, and 5 dB SNR.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEED = 42
VAL_SIZE = 0.2
DEFAULT_SNRS = ["clean", "20", "10", "5"]
DEFAULT_MAIN_OUTPUT_MODELS = ["DA_LGBM", "ADANN", "LightGBM"]
SEQUENCE_LENGTH = 100
N_FEATURES = 5
MODEL_ALIASES = {
    "DA_LGBM": "ADANN_LightGBM",
    "DA-LGBM": "ADANN_LightGBM",
    "ADANN_LightGBM": "ADANN_LightGBM",
    "ADANN": "ADANN",
    "LightGBM": "LightGBM",
    "DSCNN": "DSCNN",
}
DISPLAY_NAMES = {
    "ADANN_LightGBM": "DA-LGBM",
    "ADANN": "ADANN",
    "LightGBM": "LightGBM",
    "DSCNN": "DS-CNN",
}


def set_seeds(seed: int, enable_tf: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    if not enable_tf:
        return
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except BaseException:
        pass


def load_data(csv_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    csv_files = glob.glob(str(PROJECT_ROOT / csv_dir / "*.csv"))
    all_data, all_labels, all_subjects = [], [], []

    for csv_file in csv_files:
        try:
            with open(csv_file, "r") as f:
                lines = f.readlines()

            data = []
            for line in lines:
                if "timestamp" in line or line.startswith("#"):
                    continue
                parts = line.strip().split(",")
                if len(parts) >= 6:
                    try:
                        data.append([float(parts[i]) for i in range(1, 6)])
                    except ValueError:
                        continue

            if not data:
                continue

            sample = np.asarray(data, dtype=np.float32)
            if len(sample) != SEQUENCE_LENGTH:
                indices = np.linspace(0, len(sample) - 1, SEQUENCE_LENGTH)
                resampled = np.zeros((SEQUENCE_LENGTH, N_FEATURES), dtype=np.float32)
                for i in range(N_FEATURES):
                    resampled[:, i] = np.interp(indices, range(len(sample)), sample[:, i])
                sample = resampled

            gesture_match = re.search(r"gesture_(\d+)", csv_file)
            subject_match = re.search(r"user_(\d+)", csv_file)
            if not gesture_match:
                continue

            all_data.append(np.round(sample).astype(np.float32))
            all_labels.append(int(gesture_match.group(1)))
            all_subjects.append(int(subject_match.group(1)) if subject_match else -1)
        except Exception as exc:
            print(f"Warning: could not process {csv_file}: {exc}")

    return np.asarray(all_data), np.asarray(all_labels), np.asarray(all_subjects)


def augment_data_local(
    X_train: np.ndarray,
    y_train: np.ndarray,
    augment_params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    factor = int(augment_params.get("augment_factor", 1))
    if factor <= 0:
        return X_train, y_train

    X_to_augment = np.repeat(X_train, factor, axis=0)
    y_to_augment = np.repeat(y_train, factor, axis=0)
    augment_prob = float(augment_params.get("augment_prob", 0.5))
    rng = np.random.RandomState(SEED)
    try:
        from tsaug import AddNoise, TimeWarp

        augmenter = (
            AddNoise(scale=float(augment_params.get("jitter_noise_level", 0.005))) @ augment_prob
            + TimeWarp(
                n_speed_change=5,
                max_speed_ratio=int(augment_params.get("time_warp_max_speed", 2)),
            )
            @ augment_prob
        )
        X_augmented = augmenter.augment(X_to_augment)
    except ImportError:
        X_augmented = X_to_augment.astype(np.float32).copy()
        jitter_level = float(augment_params.get("jitter_noise_level", 0.005))
        channel_std = np.std(X_train.astype(np.float32), axis=(0, 1), keepdims=True)
        channel_std = np.where(channel_std > 1e-6, channel_std, 1.0)
        mask = rng.rand(len(X_augmented)) < augment_prob
        X_augmented[mask] += rng.normal(0.0, jitter_level * channel_std, size=X_augmented[mask].shape)

    scale_min, scale_max = augment_params.get("scale_range", [0.98, 1.02])
    for i in range(X_augmented.shape[0]):
        if rng.rand() < augment_prob:
            X_augmented[i] = X_augmented[i] * rng.uniform(scale_min, scale_max)

    X_final = np.vstack([X_train, X_augmented])
    y_final = np.concatenate([y_train, y_to_augment])
    return np.round(X_final).astype(np.int16), y_final


def load_best_params(
    model_type: str,
    fold_id: int,
    optimization_mode: str,
    recursive: bool = False,
) -> Tuple[Dict[str, Any], Optional[Path]]:
    search_root = PROJECT_ROOT / "outputs" / model_type / "loso" / optimization_mode
    pattern = f"best_params_{model_type}_loso_{optimization_mode}_fold_{fold_id}_*.json"
    if recursive:
        matches = [Path(p) for p in glob.glob(str(search_root / "**" / pattern), recursive=True)]
    else:
        matches = [Path(p) for p in glob.glob(str(search_root / pattern))]

    if not matches:
        return {}, None

    def sort_key(path: Path) -> Tuple[str, float]:
        match = re.search(r"_(\d{8}_\d{6})\.json$", path.name)
        timestamp = match.group(1) if match else ""
        return timestamp, path.stat().st_mtime

    best_path = sorted(matches, key=sort_key)[-1]
    with best_path.open("r") as f:
        payload = json.load(f)
    return payload.get("best_params", payload), best_path


def with_default_params(model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(params)
    if model_type == "ADANN":
        merged.setdefault("learning_rate", 1e-3)
        merged.setdefault("batch_size", 32)
        merged.setdefault("feature_size", 64)
        merged.setdefault("gesture_loss_weight", 1.0)
        merged.setdefault("domain_loss_weight", 1.0)
        merged.setdefault("weight_decay", 1e-5)
        merged.setdefault("n_epochs", 100)
    elif model_type == "ADANN_LightGBM":
        merged.setdefault("adann_learning_rate", 1e-3)
        merged.setdefault("adann_feature_size", 64)
        merged.setdefault("adann_dropout", 0.3)
        merged.setdefault("adann_classifier_dropout", 0.2)
        merged.setdefault("adann_epochs", 100)
        merged.setdefault("gesture_loss_weight", 1.0)
        merged.setdefault("domain_loss_weight", 1.0)
        merged.setdefault("grl_gamma", 10.0)
        merged.setdefault("grl_max", 0.99)
        merged.setdefault("batch_size", 32)
        merged.setdefault("lgb_num_leaves", 31)
        merged.setdefault("lgb_learning_rate", 0.1)
        merged.setdefault("lgb_feature_fraction", 0.8)
        merged.setdefault("lgb_bagging_fraction", 0.8)
        merged.setdefault("lgb_min_child_samples", 20)
        merged.setdefault("lgb_n_estimators", 100)
        merged.setdefault("lgb_max_depth", -1)
        merged.setdefault("ensemble_adann_weight", 0.5)
        merged.setdefault("class_balanced_batches", False)
        merged.setdefault("auto_tune_ensemble_weight", False)
        merged.setdefault("auto_tune_gate_thresholds", False)
    elif model_type == "DSCNN":
        merged.setdefault("n_conv_layers", 2)
        merged.setdefault("use_batch_norm", True)
        merged.setdefault("use_conv_dropout", False)
        merged.setdefault("use_dense_dropout", True)
        merged.setdefault("activation", "relu")
        merged.setdefault("dense_units", 64)
        merged.setdefault("learning_rate", 1e-3)
        merged.setdefault("batch_size", 32)
        merged.setdefault("conv1_filters", 32)
        merged.setdefault("conv1_kernel", 3)
        merged.setdefault("conv2_filters", 64)
        merged.setdefault("conv2_kernel", 3)
        merged.setdefault("dense_dropout", 0.3)
    return merged


def maybe_cap_epochs(params: Dict[str, Any], model_type: str, max_epochs: Optional[int]) -> Dict[str, Any]:
    if max_epochs is None:
        return params
    capped = dict(params)
    if model_type == "ADANN":
        capped["n_epochs"] = min(int(capped.get("n_epochs", max_epochs)), max_epochs)
    elif model_type == "ADANN_LightGBM":
        capped["adann_epochs"] = min(int(capped.get("adann_epochs", max_epochs)), max_epochs)
    return capped


def build_augment_params(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "augment_factor" not in params:
        return None
    scale_range = params.get("scale_range")
    if scale_range is None:
        scale_range = [params.get("scale_min", 0.98), params.get("scale_max", 1.02)]
    return {
        "augment_factor": int(params.get("augment_factor", 1)),
        "jitter_noise_level": float(params.get("jitter_noise_level", 0.005)),
        "time_warp_max_speed": int(params.get("time_warp_max_speed", 2)),
        "scale_range": scale_range,
        "augment_prob": float(params.get("augment_prob", 0.3)),
    }


def augment_with_subjects(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    augment_params = build_augment_params(params)
    if augment_params is None or augment_params["augment_factor"] <= 0:
        return X, y, subjects

    X_aug, y_aug = augment_data_local(X, y, augment_params)
    repeated_subjects = np.repeat(subjects, augment_params["augment_factor"], axis=0)
    subjects_aug = np.concatenate([subjects, repeated_subjects])
    if len(subjects_aug) != len(X_aug):
        raise RuntimeError("Subject augmentation bookkeeping mismatch.")
    return X_aug, y_aug, subjects_aug


def add_awgn(
    X: np.ndarray,
    snr_db: Optional[float],
    rng: np.random.Generator,
    reference: str = "centered",
    clip_min: float = 0.0,
    clip_max: float = 1023.0,
) -> np.ndarray:
    if snr_db is None:
        return X.copy()

    X_float = X.astype(np.float32)
    if reference == "raw":
        signal = X_float
    else:
        signal = X_float - X_float.mean(axis=1, keepdims=True)

    power = np.mean(signal**2, axis=(1, 2), keepdims=True)
    fallback_power = float(np.mean(signal**2)) if float(np.mean(signal**2)) > 1e-12 else 1.0
    power = np.where(power > 1e-12, power, fallback_power)
    noise_power = power / (10.0 ** (float(snr_db) / 10.0))
    noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power), size=X_float.shape)
    noisy = X_float + noise
    return np.clip(noisy, clip_min, clip_max).astype(np.float32)


def predict_labels(model_type: str, creator: Any, trained: Any, X: np.ndarray, scaler: Any = None) -> np.ndarray:
    if model_type == "LightGBM":
        X_feat, _ = creator.extract_and_scale_features(X, scaler=scaler, arduino_mode=False)
        return np.asarray(trained.predict(X_feat)).astype(int)
    if model_type == "DSCNN":
        if creator == "torch_dscnn":
            import torch

            X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            device = next(trained.parameters()).device
            trained.eval()
            preds = []
            with torch.no_grad():
                for start in range(0, len(X_scaled), 64):
                    batch = torch.tensor(X_scaled[start : start + 64], dtype=torch.float32).to(device)
                    preds.append(trained(batch).argmax(dim=1).cpu().numpy())
            return np.concatenate(preds).astype(int)

        X_scaled, _ = creator.extract_and_scale_features(X, scaler=scaler, arduino_mode=False)
        proba = trained.predict(X_scaled, verbose=0)
        return np.argmax(proba, axis=1).astype(int)
    if model_type == "ADANN":
        return np.asarray(creator.predict(trained, X)).astype(int)
    if model_type == "ADANN_LightGBM":
        return np.asarray(creator.predict(trained, X)).astype(int)
    raise ValueError(f"Unsupported model_type={model_type}")


def train_one_fold(
    model_type: str,
    params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    subjects_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    subjects_val: np.ndarray,
    epochs: int,
    dscnn_backend: str = "torch",
) -> Tuple[Any, Any, Any]:
    if model_type == "LightGBM":
        from src.training.train_lightgbm import LightgbmModelCreator

        creator = LightgbmModelCreator()
        X_train_feat, scaler = creator.extract_and_scale_features(X_train, fit=True, arduino_mode=False)
        X_val_feat, _ = creator.extract_and_scale_features(X_val, scaler=scaler, arduino_mode=False)
        model = creator.create_model(params, arduino_mode=False)
        model.fit(X_train_feat, y_train, validation_data=(X_val_feat, y_val), verbose=0)
        return creator, model, scaler

    if model_type == "DSCNN":
        if dscnn_backend == "torch":
            import copy
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            class Activation(nn.Module):
                def __init__(self, name: str):
                    super().__init__()
                    self.name = name

                def forward(self, x):
                    if self.name == "tanh":
                        return torch.tanh(x)
                    if self.name == "swish":
                        return x * torch.sigmoid(x)
                    return torch.relu(x)

            class SeparableConv1d(nn.Module):
                def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
                    super().__init__()
                    self.depthwise = nn.Conv1d(
                        in_channels,
                        in_channels,
                        kernel_size=kernel_size,
                        groups=in_channels,
                    )
                    self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

                def forward(self, x):
                    return self.pointwise(self.depthwise(x))

            class TorchDSCNN(nn.Module):
                def __init__(self, p: Dict[str, Any]):
                    super().__init__()
                    activation = p.get("activation", "relu")
                    layers: List[nn.Module] = []
                    in_ch = N_FEATURES
                    n_layers = int(p.get("n_conv_layers", 2))
                    for i in range(n_layers):
                        out_ch = int(p.get(f"conv{i + 1}_filters", 32 if i == 0 else 64))
                        kernel = int(p.get(f"conv{i + 1}_kernel", 3))
                        layers.extend([SeparableConv1d(in_ch, out_ch, kernel), Activation(activation)])
                        if p.get("use_batch_norm", True):
                            layers.append(nn.BatchNorm1d(out_ch))
                        if i < 3:
                            layers.append(nn.MaxPool1d(2))
                        if p.get("use_conv_dropout", False):
                            layers.append(nn.Dropout(float(p.get("conv_dropout", 0.2))))
                        in_ch = out_ch

                    dense_units = int(p.get("dense_units", 64))
                    dense_layers: List[nn.Module] = [
                        nn.AdaptiveAvgPool1d(1),
                        nn.Flatten(),
                        nn.Linear(in_ch, dense_units),
                        Activation(activation),
                    ]
                    if p.get("use_dense_dropout", True):
                        dense_layers.append(nn.Dropout(float(p.get("dense_dropout", 0.3))))
                    dense_layers.append(nn.Linear(dense_units, 11))
                    self.features = nn.Sequential(*layers)
                    self.classifier = nn.Sequential(*dense_layers)

                def forward(self, x):
                    return self.classifier(self.features(x.transpose(1, 2)))

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = TorchDSCNN(params).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=float(params.get("learning_rate", 1e-3)))
            criterion = nn.CrossEntropyLoss()
            train_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_train_scaled, dtype=torch.float32),
                    torch.tensor(y_train, dtype=torch.long),
                ),
                batch_size=int(params.get("batch_size", 32)),
                shuffle=True,
            )
            X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

            best_state = copy.deepcopy(model.state_dict())
            best_val_acc = -1.0
            patience_counter = 0
            for _epoch in range(epochs):
                model.train()
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val_tensor).argmax(dim=1)
                    val_acc = float((val_pred == y_val_tensor).float().mean().item())
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 15:
                        break

            model.load_state_dict(best_state)
            model.eval()
            return "torch_dscnn", model, scaler

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        from tensorflow.keras.callbacks import EarlyStopping

        from src.training.train_dscnn1d import Cnn1dModelCreator

        creator = Cnn1dModelCreator()
        X_train_scaled, scaler = creator.extract_and_scale_features(X_train, fit=True, arduino_mode=False)
        X_val_scaled, _ = creator.extract_and_scale_features(X_val, scaler=scaler, arduino_mode=False)
        model = creator.create_model(params, arduino_mode=False)
        callbacks = [EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True)]
        model.fit(
            X_train_scaled,
            y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=int(params.get("batch_size", 32)),
            callbacks=callbacks,
            verbose=0,
        )
        return creator, model, scaler

    if model_type == "ADANN":
        from src.training.train_adann import AdannModelCreator

        creator = AdannModelCreator()
        wrapper = creator.create_model(params, arduino_mode=False)
        trained, _ = creator.train_model(
            wrapper.pytorch_model,
            X_train,
            y_train,
            subjects_train,
            X_val,
            y_val,
            subjects_val,
            params,
            return_history=False,
        )
        return creator, trained, None

    if model_type == "ADANN_LightGBM":
        from src.training.train_adann_lightgbm import AdannLightgbmModelCreator

        creator = AdannLightgbmModelCreator()
        wrapper = creator.create_model(params, arduino_mode=False)
        trained, _ = creator.train_model(
            wrapper.hybrid_model,
            X_train,
            y_train,
            subjects_train,
            X_val,
            y_val,
            subjects_val,
            params,
            return_history=False,
        )
        return creator, trained, None

    raise ValueError(f"Unsupported model_type={model_type}")


def parse_snrs(values: Iterable[str]) -> List[Optional[float]]:
    parsed: List[Optional[float]] = []
    for value in values:
        if str(value).lower() == "clean":
            parsed.append(None)
        else:
            parsed.append(float(value))
    return parsed


def snr_label(snr_db: Optional[float]) -> str:
    return "clean" if snr_db is None else f"{snr_db:g}dB"


def ci95(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return 0.0
    return float(1.96 * values.std(ddof=1) / np.sqrt(values.size))


def select_output_models(summary: pd.DataFrame, model_names: Iterable[str]) -> pd.DataFrame:
    selected = [MODEL_ALIASES[name] for name in model_names]
    filtered = summary[summary["model"].isin(selected)].copy()
    order_map = {model: idx for idx, model in enumerate(selected)}
    filtered["_model_order"] = filtered["model"].map(order_map)
    filtered = filtered.sort_values(["_model_order", "snr_label"]).drop(columns=["_model_order"])
    return filtered


def iter_model_groups(summary: pd.DataFrame) -> Iterable[Tuple[str, pd.DataFrame]]:
    seen = []
    for _, row in summary[["model", "model_display"]].drop_duplicates().iterrows():
        seen.append((row["model"], row["model_display"]))
    for model, model_display in seen:
        yield model_display, summary[summary["model"] == model]


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    order = ["clean", "20dB", "10dB", "5dB"]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for model_name, group in iter_model_groups(summary):
        group = group.set_index("snr_label").reindex(order)
        ax.errorbar(
            x,
            group["macro_f1_mean"],
            yerr=group["macro_f1_ci95"],
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=model_name,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_xlabel("Test-time SNR")
    ax.set_ylabel("LOSO Macro-F1")
    ax.set_ylim(0, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=min(3, summary["model_display"].nunique()))
    fig.tight_layout()
    fig.savefig(output_dir / "awgn_macro_f1.png", dpi=300)
    fig.savefig(output_dir / "awgn_macro_f1.pdf")
    fig.savefig(output_dir / "awgn_macro_f1.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for model_name, group in iter_model_groups(summary):
        group = group.set_index("snr_label").reindex(order)
        ax.plot(x, group["delta_macro_f1_mean"], marker="o", linewidth=1.8, label=model_name)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_xlabel("Test-time SNR")
    ax.set_ylabel("Delta Macro-F1 vs clean")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=min(3, summary["model_display"].nunique()))
    fig.tight_layout()
    fig.savefig(output_dir / "awgn_delta_macro_f1.png", dpi=300)
    fig.savefig(output_dir / "awgn_delta_macro_f1.pdf")
    fig.savefig(output_dir / "awgn_delta_macro_f1.svg")
    plt.close(fig)


def plot_combined_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    order = ["clean", "20dB", "10dB", "5dB"]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharex=True)

    for model_name, group in iter_model_groups(summary):
        group = group.set_index("snr_label").reindex(order)
        axes[0].errorbar(
            x,
            group["macro_f1_mean"],
            yerr=group["macro_f1_ci95"],
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=model_name,
        )
        axes[1].plot(x, group["delta_macro_f1_mean"], marker="o", linewidth=1.8, label=model_name)

    axes[0].set_title("(a) Macro-F1 under AWGN")
    axes[0].set_ylabel("LOSO Macro-F1")
    axes[0].set_ylim(0, 1.02)

    axes[1].set_title("(b) Change from clean baseline")
    axes[1].set_ylabel("Delta Macro-F1 vs clean")
    axes[1].axhline(0.0, color="black", linewidth=0.8)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(order)
        ax.set_xlabel("Test-time SNR")
        ax.grid(True, axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=min(3, len(labels)))
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(output_dir / "awgn_macro_f1_combined.png", dpi=300)
    fig.savefig(output_dir / "awgn_macro_f1_combined.pdf")
    fig.savefig(output_dir / "awgn_macro_f1_combined.svg")
    plt.close(fig)


def make_latex_table(summary: pd.DataFrame, output_dir: Path) -> None:
    order = ["clean", "20dB", "10dB", "5dB"]
    rows = []
    for model_name, group in iter_model_groups(summary):
        group = group.set_index("snr_label").reindex(order)
        row = [model_name]
        for label in order:
            record = group.loc[label]
            row.append(f"{record['macro_f1_mean']:.4f} $\\pm$ {record['macro_f1_ci95']:.4f}")
        rows.append(row)

    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Model & Clean & 20 dB & 10 dB & 5 dB \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (output_dir / "awgn_table.tex").write_text("\n".join(lines))


def run(args: argparse.Namespace) -> None:
    os.chdir(PROJECT_ROOT)
    requested_models = [MODEL_ALIASES[name] for name in args.models]
    set_seeds(args.seed, enable_tf=("DSCNN" in requested_models and args.dscnn_backend == "tensorflow"))

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    snrs = parse_snrs(args.snrs)
    X, y, subjects = load_data(args.csv_dir)
    unique_subjects = sorted(int(s) for s in np.unique(subjects[subjects != -1]))
    if args.subjects:
        requested_subjects = {int(s) for s in args.subjects}
        unique_subjects = [s for s in unique_subjects if s in requested_subjects]

    rows: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {
        "seed": args.seed,
        "csv_dir": args.csv_dir,
        "snrs": [snr_label(s) for s in snrs],
        "snr_reference": args.snr_reference,
        "models": requested_models,
        "param_files": {},
    }

    for model_type in requested_models:
        model_display = DISPLAY_NAMES[model_type]
        metadata["param_files"].setdefault(model_type, {})
        for fold_id in unique_subjects:
            print(f"\n=== {model_display} | LOSO subject {fold_id} ===", flush=True)
            train_idx = np.where(subjects != fold_id)[0]
            test_idx = np.where(subjects == fold_id)[0]
            X_train_full, y_train_full = X[train_idx], y[train_idx]
            subjects_train_full = subjects[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            params, params_path = load_best_params(
                model_type,
                fold_id,
                args.optimization_mode,
                recursive=args.recursive_params,
            )
            params = with_default_params(model_type, params)
            params = maybe_cap_epochs(params, model_type, args.max_epochs)
            if model_type == "DSCNN":
                params.setdefault("scale_range", [params.get("scale_min", 0.98), params.get("scale_max", 1.02)])
            metadata["param_files"][model_type][str(fold_id)] = str(params_path) if params_path else None

            X_train, X_val, y_train, y_val, subjects_train, subjects_val = train_test_split(
                X_train_full,
                y_train_full,
                subjects_train_full,
                test_size=VAL_SIZE,
                random_state=args.seed,
                stratify=y_train_full,
            )
            if not args.no_augment:
                X_train, y_train, subjects_train = augment_with_subjects(X_train, y_train, subjects_train, params)

            creator, trained, scaler = train_one_fold(
                model_type,
                params,
                X_train,
                y_train,
                subjects_train,
                X_val,
                y_val,
                subjects_val,
                epochs=args.epochs,
                dscnn_backend=args.dscnn_backend,
            )

            clean_f1: Optional[float] = None
            for repeat in range(args.noise_repeats):
                for snr_db in snrs:
                    rng = np.random.default_rng(args.seed + 1000 * fold_id + 100 * repeat + (0 if snr_db is None else int(snr_db)))
                    X_eval = add_awgn(
                        X_test,
                        snr_db,
                        rng,
                        reference=args.snr_reference,
                        clip_min=args.clip_min,
                        clip_max=args.clip_max,
                    )
                    y_pred = predict_labels(model_type, creator, trained, X_eval, scaler)
                    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
                    accuracy = float(accuracy_score(y_test, y_pred))
                    if snr_db is None and repeat == 0:
                        clean_f1 = macro_f1
                    delta = macro_f1 - clean_f1 if clean_f1 is not None else 0.0
                    rows.append(
                        {
                            "model": model_type,
                            "model_display": model_display,
                            "fold": fold_id,
                            "repeat": repeat,
                            "snr_db": "clean" if snr_db is None else float(snr_db),
                            "snr_label": snr_label(snr_db),
                            "macro_f1": macro_f1,
                            "accuracy": accuracy,
                            "delta_macro_f1": float(delta),
                        }
                    )
                    print(
                        f"{model_display} fold {fold_id} {snr_label(snr_db)}: "
                        f"Macro-F1={macro_f1:.4f}, Acc={accuracy:.4f}",
                        flush=True,
                    )

    raw = pd.DataFrame(rows)
    raw_path = output_dir / "awgn_macro_f1_by_fold.csv"
    raw.to_csv(raw_path, index=False)

    summary_rows = []
    for (model_type, model_display, label), group in raw.groupby(["model", "model_display", "snr_label"], sort=False):
        macro = group["macro_f1"].to_numpy(dtype=float)
        acc = group["accuracy"].to_numpy(dtype=float)
        delta = group["delta_macro_f1"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "model": model_type,
                "model_display": model_display,
                "snr_label": label,
                "macro_f1_mean": float(macro.mean()),
                "macro_f1_std": float(macro.std(ddof=1)) if macro.size > 1 else 0.0,
                "macro_f1_ci95": ci95(macro),
                "accuracy_mean": float(acc.mean()),
                "accuracy_ci95": ci95(acc),
                "delta_macro_f1_mean": float(delta.mean()),
                "delta_macro_f1_ci95": ci95(delta),
                "n": int(len(group)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "awgn_summary.csv", index=False)
    output_summary = select_output_models(summary, args.output_models)
    output_summary.to_csv(output_dir / "awgn_summary_main.csv", index=False)
    plot_summary(output_summary, output_dir)
    plot_combined_summary(output_summary, output_dir)
    make_latex_table(output_summary, output_dir)

    metadata["raw_csv"] = str(raw_path)
    metadata["summary_csv"] = str(output_dir / "awgn_summary.csv")
    (output_dir / "awgn_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved AWGN robustness outputs to {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOSO AWGN robustness experiment.")
    parser.add_argument("--csv_dir", default="datasets/gesture_csv")
    parser.add_argument("--output_dir", default="outputs/robustness/awgn")
    parser.add_argument("--models", nargs="+", default=["DA_LGBM", "ADANN", "LightGBM", "DSCNN"], choices=sorted(MODEL_ALIASES))
    parser.add_argument(
        "--output_models",
        nargs="+",
        default=DEFAULT_MAIN_OUTPUT_MODELS,
        choices=sorted(MODEL_ALIASES),
        help="Models included in paper-facing figures and LaTeX table. Raw CSV still keeps every trained model.",
    )
    parser.add_argument("--snrs", nargs="+", default=DEFAULT_SNRS, help='Use "clean" or numeric SNR values in dB.')
    parser.add_argument("--subjects", nargs="*", type=int, help="Optional subset of LOSO subject IDs for smoke tests.")
    parser.add_argument("--optimization_mode", default="full", choices=["full", "arduino"])
    parser.add_argument("--recursive_params", action="store_true", help="Also search nested output folders for best_params files.")
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs for DSCNN training.")
    parser.add_argument("--dscnn_backend", default="torch", choices=["torch", "tensorflow"])
    parser.add_argument("--max_epochs", type=int, default=None, help="Optional cap for ADANN/DA-LGBM epochs.")
    parser.add_argument("--noise_repeats", type=int, default=1)
    parser.add_argument("--snr_reference", default="centered", choices=["centered", "raw"])
    parser.add_argument("--clip_min", type=float, default=0.0)
    parser.add_argument("--clip_max", type=float, default=1023.0)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
