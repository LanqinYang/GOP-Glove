#!/usr/bin/env python3
"""
Generate t-SNE visualisations of ADANN bottleneck features with and without GRL.

The script trains two otherwise identical ADANN-style encoders:
  1. no_grl: gesture + subject supervision without gradient reversal
  2. with_grl: gesture supervision plus adversarial subject/domain loss

It then extracts the bottleneck features for all windows and renders a 2x2
t-SNE panel coloured by subject/domain and vocabulary class.
"""

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))

import matplotlib.pyplot as plt

from src.training.train_adann import AdversarialFeatureExtractor, EnhancedFeatureExtractor


SEQUENCE_LENGTH = 100
N_FEATURES = 5
N_CLASSES = 11
GESTURE_NAMES = {
    0: "D0 Zero",
    1: "D1 One",
    2: "D2 Two",
    3: "D3 Three",
    4: "D4 Four",
    5: "D5 Five",
    6: "D6 Six",
    7: "D7 Seven",
    8: "D8 Eight",
    9: "D9 Nine",
    10: "Static",
}


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data(csv_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    csv_files = sorted(csv_dir.glob("*.csv"))
    all_data, all_labels, all_subjects = [], [], []

    for csv_file in csv_files:
        gesture_match = re.search(r"gesture_(\d+)", csv_file.name)
        subject_match = re.search(r"user_(\d+)", csv_file.name)
        if not gesture_match or not subject_match:
            continue

        rows = []
        with csv_file.open("r", newline="") as handle:
            for raw_line in handle:
                if "timestamp" in raw_line or raw_line.startswith("#"):
                    continue
                parts = raw_line.strip().split(",")
                if len(parts) < 6:
                    continue
                try:
                    rows.append([float(parts[i]) for i in range(1, 6)])
                except ValueError:
                    continue

        if not rows:
            continue

        data = np.asarray(rows, dtype=np.float32)
        if len(data) != SEQUENCE_LENGTH:
            indices = np.linspace(0, len(data) - 1, SEQUENCE_LENGTH)
            resampled = np.zeros((SEQUENCE_LENGTH, N_FEATURES), dtype=np.float32)
            for channel in range(N_FEATURES):
                resampled[:, channel] = np.interp(indices, np.arange(len(data)), data[:, channel])
            data = resampled

        all_data.append(np.round(data).astype(np.int16))
        all_labels.append(int(gesture_match.group(1)))
        all_subjects.append(int(subject_match.group(1)))

    if not all_data:
        raise RuntimeError(f"No usable CSV files found in {csv_dir}")

    return np.asarray(all_data), np.asarray(all_labels), np.asarray(all_subjects)


def extract_handcrafted_features(X: np.ndarray) -> np.ndarray:
    extractor = EnhancedFeatureExtractor()
    features = [extractor.extract_comprehensive_features(sample) for sample in X]
    features = np.asarray(features, dtype=np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)


def make_split(y: np.ndarray, subjects: np.ndarray, val_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    stratify_key = np.asarray([f"{subj}_{label}" for subj, label in zip(subjects, y)])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(y)), stratify_key))
    return train_idx, val_idx


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.FloatTensor(X),
        torch.LongTensor(y),
        torch.LongTensor(subjects),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    domain_criterion: nn.Module | None = None,
    return_details: bool = False,
) -> dict[str, float]:
    model.eval()
    y_true, y_pred, s_true, s_pred, s_entropy = [], [], [], [], []
    total_domain_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for data, gesture_labels, subject_labels in loader:
            data = data.to(device)
            subject_labels_device = subject_labels.to(device)
            gesture_logits, domain_logits, _ = model(data, reverse_gradient=False)
            domain_prob = torch.softmax(domain_logits, dim=1)
            entropy = -(domain_prob * torch.log(domain_prob + 1e-12)).sum(dim=1)
            if domain_criterion is not None:
                total_domain_loss += float(domain_criterion(domain_logits, subject_labels_device).item()) * data.size(0)
                total_samples += data.size(0)
            y_true.extend(gesture_labels.numpy().tolist())
            y_pred.extend(gesture_logits.argmax(1).cpu().numpy().tolist())
            s_true.extend(subject_labels.numpy().tolist())
            s_pred.extend(domain_logits.argmax(1).cpu().numpy().tolist())
            s_entropy.extend(entropy.cpu().numpy().tolist())

    n_subjects = max(len(set(s_true)), 1)
    metrics = {
        "gesture_accuracy": float(accuracy_score(y_true, y_pred)),
        "gesture_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "domain_accuracy": float(accuracy_score(s_true, s_pred)),
        "domain_entropy": float(np.mean(s_entropy)),
        "domain_entropy_normalized": float(np.mean(s_entropy) / np.log(n_subjects)) if n_subjects > 1 else 0.0,
    }
    if domain_criterion is not None:
        metrics["domain_loss"] = float(total_domain_loss / max(total_samples, 1))
    if return_details:
        metrics.update({
            "gesture_true": y_true,
            "gesture_pred": y_pred,
            "domain_true": s_true,
            "domain_pred": s_pred,
            "domain_entropy_samples": s_entropy,
        })
    return metrics


def train_variant(
    X_train: np.ndarray,
    y_train: np.ndarray,
    subjects_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    subjects_val: np.ndarray,
    *,
    use_grl: bool,
    feature_size: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    n_epochs: int,
    domain_loss_weight: float,
    device: torch.device,
) -> tuple[nn.Module, dict[str, list[float]], dict[str, float]]:
    model = AdversarialFeatureExtractor(
        input_size=X_train.shape[1],
        feature_size=feature_size,
        n_gestures=N_CLASSES,
        n_subjects=len(np.unique(np.concatenate([subjects_train, subjects_val]))),
    ).to(device)

    train_loader = make_loader(X_train, y_train, subjects_train, batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, subjects_val, batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    gesture_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_gesture_loss": [],
        "train_domain_loss": [],
        "train_gesture_accuracy": [],
        "val_gesture_accuracy": [],
        "val_gesture_macro_f1": [],
        "val_domain_accuracy": [],
        "val_domain_loss": [],
        "grl_alpha": [],
    }

    best_state = None
    best_macro_f1 = -1.0

    for epoch in range(n_epochs):
        model.train()
        alpha = 2.0 / (1.0 + np.exp(-10.0 * float(epoch) / max(n_epochs, 1))) - 1.0
        model.set_alpha(alpha if use_grl else 0.0)

        total_loss = 0.0
        total_gesture_loss = 0.0
        total_domain_loss = 0.0
        total_samples = 0
        total_correct = 0

        for data, gesture_labels, subject_labels in train_loader:
            data = data.to(device)
            gesture_labels = gesture_labels.to(device)
            subject_labels = subject_labels.to(device)

            optimizer.zero_grad()
            gesture_logits, domain_logits, _ = model(data, reverse_gradient=use_grl)
            gesture_loss = gesture_criterion(gesture_logits, gesture_labels)
            domain_loss = domain_criterion(domain_logits, subject_labels)
            loss = gesture_loss + domain_loss_weight * domain_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item()) * data.size(0)
            total_gesture_loss += float(gesture_loss.item()) * data.size(0)
            total_domain_loss += float(domain_loss.item()) * data.size(0)
            total_samples += data.size(0)
            total_correct += int((gesture_logits.argmax(1) == gesture_labels).sum().item())

        metrics = evaluate(model, val_loader, device, domain_criterion=domain_criterion)
        history["train_loss"].append(total_loss / max(total_samples, 1))
        history["train_gesture_loss"].append(total_gesture_loss / max(total_samples, 1))
        history["train_domain_loss"].append(total_domain_loss / max(total_samples, 1))
        history["train_gesture_accuracy"].append(total_correct / max(total_samples, 1))
        history["val_gesture_accuracy"].append(metrics["gesture_accuracy"])
        history["val_gesture_macro_f1"].append(metrics["gesture_macro_f1"])
        history["val_domain_accuracy"].append(metrics["domain_accuracy"])
        history["val_domain_loss"].append(metrics["domain_loss"])
        history["grl_alpha"].append(float(alpha if use_grl else 0.0))

        if metrics["gesture_macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["gesture_macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if epoch % 25 == 0 or epoch == n_epochs - 1:
            label = "with_grl" if use_grl else "no_grl"
            print(
                f"{label} epoch {epoch:03d}: "
                f"loss={history['train_loss'][-1]:.4f}, "
                f"val_macro_f1={metrics['gesture_macro_f1']:.4f}, "
                f"domain_acc={metrics['domain_accuracy']:.4f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate(model, val_loader, device, domain_criterion=domain_criterion, return_details=True)
    return model, history, final_metrics


def extract_bottleneck(model: nn.Module, X_scaled: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    features = []
    with torch.no_grad():
        for start in range(0, len(X_scaled), batch_size):
            batch = torch.FloatTensor(X_scaled[start:start + batch_size]).to(device)
            _, _, bottleneck = model(batch, reverse_gradient=False)
            features.append(bottleneck.cpu().numpy())
    return np.vstack(features)


def run_tsne(features: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        max_iter=1000,
    ).fit_transform(features)


def scatter_panel(ax, coords, color_values, names, title, cmap_name):
    cmap = plt.get_cmap(cmap_name, len(names))
    for idx, name in enumerate(names):
        mask = color_values == idx
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            alpha=0.78,
            color=cmap(idx),
            linewidths=0,
            label=name,
        )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#C7CDD8")


def save_plot(
    coords_by_variant: dict[str, np.ndarray],
    y_encoded: np.ndarray,
    subject_encoded: np.ndarray,
    gesture_names: list[str],
    subject_names: list[str],
    output_path: Path,
) -> None:
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.8))
    fig.subplots_adjust(left=0.04, right=0.77, top=0.88, bottom=0.08, wspace=0.06, hspace=0.22)

    scatter_panel(
        axes[0, 0],
        coords_by_variant["no_grl"],
        subject_encoded,
        subject_names,
        "(a) Without GRL, coloured by subject",
        "tab10",
    )
    scatter_panel(
        axes[0, 1],
        coords_by_variant["no_grl"],
        y_encoded,
        gesture_names,
        "(b) Without GRL, coloured by gesture",
        "tab20",
    )
    scatter_panel(
        axes[1, 0],
        coords_by_variant["with_grl"],
        subject_encoded,
        subject_names,
        "(c) With GRL, coloured by subject",
        "tab10",
    )
    scatter_panel(
        axes[1, 1],
        coords_by_variant["with_grl"],
        y_encoded,
        gesture_names,
        "(d) With GRL, coloured by gesture",
        "tab20",
    )

    subject_handles, subject_labels = axes[1, 0].get_legend_handles_labels()
    gesture_handles, gesture_labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(
        subject_handles,
        subject_labels,
        loc="upper left",
        bbox_to_anchor=(0.79, 0.80),
        ncol=1,
        frameon=False,
        title="Subject/domain",
    )
    fig.legend(
        gesture_handles,
        gesture_labels,
        loc="upper left",
        bbox_to_anchor=(0.79, 0.47),
        ncol=1,
        frameon=False,
        title="Gesture class",
    )
    fig.suptitle("t-SNE of ADANN bottleneck feature space", fontsize=11, y=0.97)
    fig.savefig(output_path)
    plt.close(fig)


def write_coordinates(
    output_path: Path,
    coords_by_variant: dict[str, np.ndarray],
    y: np.ndarray,
    subjects: np.ndarray,
) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "sample_index", "tsne_x", "tsne_y", "gesture_id", "subject_id"])
        for variant, coords in coords_by_variant.items():
            for idx, (x_value, y_value) in enumerate(coords):
                writer.writerow([variant, idx, float(x_value), float(y_value), int(y[idx]), int(subjects[idx])])


def save_domain_confusion_plot(
    confusion_by_variant: dict[str, np.ndarray],
    metrics_by_variant: dict[str, dict[str, float]],
    subject_names: list[str],
    output_path: Path,
) -> None:
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    fig.subplots_adjust(left=0.08, right=0.89, top=0.82, bottom=0.18, wspace=0.38)
    titles = {
        "no_grl": "(a) Without GRL",
        "with_grl": "(b) With GRL",
    }

    im = None
    for ax, variant in zip(axes, ["no_grl", "with_grl"]):
        cm = confusion_by_variant[variant]
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(
            f"{titles[variant]}\n"
            f"Acc={metrics_by_variant[variant]['domain_accuracy']:.3f}, "
            f"H={metrics_by_variant[variant]['domain_entropy_normalized']:.3f}",
            fontsize=9,
        )
        ax.set_xlabel("Predicted subject")
        ax.set_ylabel("True subject")
        ax.set_xticks(np.arange(len(subject_names)))
        ax.set_yticks(np.arange(len(subject_names)))
        ax.set_xticklabels(subject_names, rotation=45, ha="right")
        ax.set_yticklabels(subject_names)
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                value = cm_norm[row, col]
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 else "#1F2937",
                    fontsize=7,
                )

    if im is not None:
        cax = fig.add_axes([0.92, 0.22, 0.02, 0.54])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Row-normalised proportion")
    fig.suptitle("Subject/domain prediction from the ADANN domain head", fontsize=10, y=0.98)
    fig.savefig(output_path)
    plt.close(fig)


def write_domain_confusion_csv(
    output_path: Path,
    confusion_by_variant: dict[str, np.ndarray],
    subject_names: list[str],
) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "true_subject", "predicted_subject", "count", "row_normalised"])
        for variant, cm in confusion_by_variant.items():
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
            for row, true_subject in enumerate(subject_names):
                for col, predicted_subject in enumerate(subject_names):
                    writer.writerow([
                        variant,
                        true_subject,
                        predicted_subject,
                        int(cm[row, col]),
                        float(cm_norm[row, col]),
                    ])


def save_grl_stability_plot(
    history: dict[str, list[float]],
    output_path: Path,
) -> None:
    epochs = np.arange(1, len(history["val_gesture_macro_f1"]) + 1)
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.92, bottom=0.10, hspace=0.26)

    axes[0].plot(epochs, history["val_gesture_macro_f1"], color="#1f77b4", linewidth=1.8)
    axes[0].set_ylabel("Validation\nMacro-F1")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_title("(a) Gesture recognition stability")

    axes[1].plot(epochs, history["train_domain_loss"], color="#ff7f0e", linewidth=1.5, label="Training")
    axes[1].plot(epochs, history["val_domain_loss"], color="#d62728", linewidth=1.5, linestyle="--", label="Validation")
    axes[1].set_ylabel("Domain loss")
    axes[1].set_title("(b) Domain-discrimination loss")
    axes[1].legend(frameon=False, ncol=2, loc="upper right")

    axes[2].plot(epochs, history["val_domain_accuracy"], color="#2ca02c", linewidth=1.6, label="Domain accuracy")
    axes[2].plot(epochs, history["grl_alpha"], color="#9467bd", linewidth=1.4, linestyle=":", label="GRL coefficient")
    axes[2].set_ylabel("Value")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].set_title("(c) Domain-head reliability and GRL schedule")
    axes[2].legend(frameon=False, ncol=2, loc="upper right")

    for ax in axes:
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("GRL training stability of the ADANN branch", fontsize=10)
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADANN bottleneck t-SNE with and without GRL")
    parser.add_argument("--csv_dir", type=Path, default=PROJECT_ROOT / "datasets" / "gesture_csv")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "outputs" / "adann_tsne_feature_space")
    parser.add_argument("--paper_figure", type=Path, default=PROJECT_ROOT / "V6" / "Figure22.png")
    parser.add_argument("--paper_domain_figure", type=Path, default=PROJECT_ROOT / "V6" / "Figure23.png")
    parser.add_argument("--paper_stability_figure", type=Path, default=PROJECT_ROOT / "V6" / "Figure24.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--feature_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--domain_loss_weight", type=float, default=1.0)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "outputs" / ".matplotlib").mkdir(parents=True, exist_ok=True)

    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    X, y, subjects = load_data(args.csv_dir)
    print(f"Loaded data: X={X.shape}, classes={len(np.unique(y))}, subjects={len(np.unique(subjects))}", flush=True)

    train_idx, val_idx = make_split(y, subjects, args.val_size, args.seed)
    handcrafted = extract_handcrafted_features(X)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(handcrafted[train_idx])
    X_val = scaler.transform(handcrafted[val_idx])
    X_all = scaler.transform(handcrafted)

    gesture_encoder = LabelEncoder().fit(y)
    subject_encoder = LabelEncoder().fit(subjects)
    y_encoded = gesture_encoder.transform(y)
    subjects_encoded = subject_encoder.transform(subjects)
    y_train = gesture_encoder.transform(y[train_idx])
    y_val = gesture_encoder.transform(y[val_idx])
    subjects_train = subject_encoder.transform(subjects[train_idx])
    subjects_val = subject_encoder.transform(subjects[val_idx])

    variant_outputs = {}
    coords_by_variant = {}
    confusion_by_variant = {}
    for variant, use_grl in [("no_grl", False), ("with_grl", True)]:
        model, history, metrics = train_variant(
            X_train,
            y_train,
            subjects_train,
            X_val,
            y_val,
            subjects_val,
            use_grl=use_grl,
            feature_size=args.feature_size,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            n_epochs=args.epochs,
            domain_loss_weight=args.domain_loss_weight,
            device=device,
        )
        bottleneck = extract_bottleneck(model, X_all, args.batch_size, device)
        coords = run_tsne(bottleneck, args.seed, args.tsne_perplexity)
        coords_by_variant[variant] = coords
        domain_true = metrics.pop("domain_true")
        domain_pred = metrics.pop("domain_pred")
        metrics.pop("gesture_true")
        metrics.pop("gesture_pred")
        metrics.pop("domain_entropy_samples")
        confusion_by_variant[variant] = confusion_matrix(
            domain_true,
            domain_pred,
            labels=list(range(len(subject_encoder.classes_))),
        )
        variant_outputs[variant] = {
            "validation_metrics": metrics,
            "history": history,
            "domain_confusion_matrix": confusion_by_variant[variant].tolist(),
        }
        np.save(args.output_dir / f"{variant}_bottleneck_features.npy", bottleneck)

    subject_names = [f"S{int(label)}" for label in subject_encoder.classes_]
    gesture_names = [GESTURE_NAMES.get(int(label), f"G{int(label)}") for label in gesture_encoder.classes_]

    figure_path = args.output_dir / "adann_bottleneck_tsne_panel.png"
    save_plot(coords_by_variant, y_encoded, subjects_encoded, gesture_names, subject_names, figure_path)
    if args.paper_figure:
        args.paper_figure.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(figure_path, args.paper_figure)

    domain_figure_path = args.output_dir / "adann_domain_confusion_panel.png"
    save_domain_confusion_plot(
        confusion_by_variant,
        {variant: values["validation_metrics"] for variant, values in variant_outputs.items()},
        subject_names,
        domain_figure_path,
    )
    if args.paper_domain_figure:
        args.paper_domain_figure.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(domain_figure_path, args.paper_domain_figure)

    stability_figure_path = args.output_dir / "adann_grl_stability_panel.png"
    save_grl_stability_plot(variant_outputs["with_grl"]["history"], stability_figure_path)
    if args.paper_stability_figure:
        args.paper_stability_figure.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stability_figure_path, args.paper_stability_figure)

    coordinates_path = args.output_dir / "adann_bottleneck_tsne_coordinates.csv"
    write_coordinates(coordinates_path, coords_by_variant, y, subjects)
    domain_confusion_path = args.output_dir / "adann_domain_confusion_long.csv"
    write_domain_confusion_csv(domain_confusion_path, confusion_by_variant, subject_names)

    summary = {
        "seed": args.seed,
        "csv_dir": str(args.csv_dir),
        "n_samples": int(len(X)),
        "n_subjects": int(len(np.unique(subjects))),
        "n_classes": int(len(np.unique(y))),
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "epochs": int(args.epochs),
        "feature_size": int(args.feature_size),
        "tsne_perplexity": float(args.tsne_perplexity),
        "outputs": {
            "figure": str(figure_path),
            "paper_figure": str(args.paper_figure) if args.paper_figure else None,
            "domain_confusion_figure": str(domain_figure_path),
            "paper_domain_confusion_figure": str(args.paper_domain_figure) if args.paper_domain_figure else None,
            "coordinates": str(coordinates_path),
            "domain_confusion": str(domain_confusion_path),
        },
        "variants": variant_outputs,
    }
    with (args.output_dir / "adann_tsne_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved figure: {figure_path}", flush=True)
    if args.paper_figure:
        print(f"Copied paper figure: {args.paper_figure}", flush=True)
    print(f"Saved domain confusion figure: {domain_figure_path}", flush=True)
    if args.paper_domain_figure:
        print(f"Copied paper domain confusion figure: {args.paper_domain_figure}", flush=True)
    print(f"Saved coordinates: {coordinates_path}", flush=True)
    print(f"Saved domain confusion CSV: {domain_confusion_path}", flush=True)
    print(f"Saved summary: {args.output_dir / 'adann_tsne_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
