#!/usr/bin/env python3
"""
Generate LOSO-fold averaged GRL stability curves for the ADANN branch.

For each held-out subject, the ADANN-GRL encoder is trained on the remaining
subjects. Gesture Macro-F1 is evaluated on the held-out subject at each epoch.
Domain loss and domain accuracy are evaluated on a source-subject validation
split because the domain head is only trained to discriminate source subjects.
"""

import argparse
import csv
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))

import matplotlib.pyplot as plt

from experiments.adann_tsne_feature_space import load_data, extract_handcrafted_features, set_all_seeds
from src.training.train_adann import AdversarialFeatureExtractor


N_CLASSES = 11


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    batch_size: int,
    shuffle: bool,
    balanced: bool = False,
) -> DataLoader:
    dataset = TensorDataset(
        torch.FloatTensor(X),
        torch.LongTensor(y),
        torch.LongTensor(subjects),
    )
    if balanced:
        counts = np.bincount(y, minlength=int(np.max(y)) + 1).astype(np.float64)
        counts[counts == 0] = 1.0
        weights = 1.0 / counts[y]
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(weights),
            num_samples=len(weights),
            replacement=True,
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def split_source_train_val(
    y: np.ndarray,
    subjects: np.ndarray,
    val_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    stratify_key = np.asarray([f"{subj}_{label}" for subj, label in zip(subjects, y)])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(y)), stratify_key))
    return train_idx, val_idx


def evaluate_gesture(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data, gesture_labels, _ in loader:
            data = data.to(device)
            gesture_logits, _, _ = model(data, reverse_gradient=False)
            y_true.extend(gesture_labels.numpy().tolist())
            y_pred.extend(gesture_logits.argmax(1).cpu().numpy().tolist())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_domain(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    s_true, s_pred = [], []
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for data, _, subject_labels in loader:
            data = data.to(device)
            subject_labels = subject_labels.to(device)
            _, domain_logits, _ = model(data, reverse_gradient=False)
            total_loss += float(criterion(domain_logits, subject_labels).item()) * data.size(0)
            total_samples += data.size(0)
            s_true.extend(subject_labels.cpu().numpy().tolist())
            s_pred.extend(domain_logits.argmax(1).cpu().numpy().tolist())
    return {
        "loss": float(total_loss / max(total_samples, 1)),
        "accuracy": float(accuracy_score(s_true, s_pred)),
    }


def train_loso_fold(
    fold_subject: int,
    handcrafted: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    *,
    seed: int,
    val_size: float,
    epochs: int,
    batch_size: int,
    feature_size: int,
    learning_rate: float,
    weight_decay: float,
    domain_loss_weight: float,
    patience: int,
    device: torch.device,
) -> dict[str, object]:
    source_mask = subjects != fold_subject
    test_mask = subjects == fold_subject
    source_indices = np.flatnonzero(source_mask)
    test_indices = np.flatnonzero(test_mask)

    source_train_rel, source_val_rel = split_source_train_val(
        y[source_indices],
        subjects[source_indices],
        val_size,
        seed + int(fold_subject),
    )
    train_indices = source_indices[source_train_rel]
    val_indices = source_indices[source_val_rel]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(handcrafted[train_indices])
    X_source_val = scaler.transform(handcrafted[val_indices])
    X_test = scaler.transform(handcrafted[test_indices])

    gesture_encoder = LabelEncoder().fit(y)
    source_subject_encoder = LabelEncoder().fit(subjects[source_indices])
    y_train = gesture_encoder.transform(y[train_indices])
    y_source_val = gesture_encoder.transform(y[val_indices])
    y_test = gesture_encoder.transform(y[test_indices])
    s_train = source_subject_encoder.transform(subjects[train_indices])
    s_source_val = source_subject_encoder.transform(subjects[val_indices])
    s_test_dummy = np.zeros_like(y_test)

    model = AdversarialFeatureExtractor(
        input_size=X_train.shape[1],
        feature_size=feature_size,
        n_gestures=N_CLASSES,
        n_subjects=len(source_subject_encoder.classes_),
    ).to(device)

    train_loader = make_loader(X_train, y_train, s_train, batch_size, shuffle=True, balanced=True)
    source_val_loader = make_loader(X_source_val, y_source_val, s_source_val, batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, s_test_dummy, batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    gesture_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    history = {
        "epoch": [],
        "heldout_macro_f1": [],
        "heldout_accuracy": [],
        "source_val_macro_f1": [],
        "source_val_accuracy": [],
        "source_domain_loss": [],
        "source_domain_accuracy": [],
        "grl_alpha": [],
    }
    best_source_val_f1 = -1.0
    selected_epoch = 1
    selected_heldout_macro_f1 = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        alpha = 2.0 / (1.0 + np.exp(-10.0 * float(epoch) / max(epochs, 1))) - 1.0
        model.set_alpha(float(alpha))

        for data, gesture_labels, subject_labels in train_loader:
            data = data.to(device)
            gesture_labels = gesture_labels.to(device)
            subject_labels = subject_labels.to(device)

            optimizer.zero_grad()
            gesture_logits, domain_logits, _ = model(data, reverse_gradient=True)
            gesture_loss = gesture_criterion(gesture_logits, gesture_labels)
            domain_loss = domain_criterion(domain_logits, subject_labels)
            loss = gesture_loss + domain_loss_weight * domain_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        source_gesture_metrics = evaluate_gesture(model, source_val_loader, device)
        gesture_metrics = evaluate_gesture(model, test_loader, device)
        domain_metrics = evaluate_domain(model, source_val_loader, domain_criterion, device)
        history["epoch"].append(epoch + 1)
        history["heldout_macro_f1"].append(gesture_metrics["macro_f1"])
        history["heldout_accuracy"].append(gesture_metrics["accuracy"])
        history["source_val_macro_f1"].append(source_gesture_metrics["macro_f1"])
        history["source_val_accuracy"].append(source_gesture_metrics["accuracy"])
        history["source_domain_loss"].append(domain_metrics["loss"])
        history["source_domain_accuracy"].append(domain_metrics["accuracy"])
        history["grl_alpha"].append(float(alpha))

        if source_gesture_metrics["macro_f1"] > best_source_val_f1:
            best_source_val_f1 = source_gesture_metrics["macro_f1"]
            selected_epoch = epoch + 1
            selected_heldout_macro_f1 = gesture_metrics["macro_f1"]
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    return {
        "heldout_subject": int(fold_subject),
        "n_train": int(len(train_indices)),
        "n_source_val": int(len(val_indices)),
        "n_test": int(len(test_indices)),
        "n_source_domains": int(len(source_subject_encoder.classes_)),
        "selected_epoch": int(selected_epoch),
        "selected_source_val_macro_f1": float(best_source_val_f1),
        "selected_heldout_macro_f1": float(selected_heldout_macro_f1),
        "history": history,
    }


def mean_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    if values.shape[0] <= 1:
        return mean, np.zeros_like(mean)
    std = values.std(axis=0, ddof=1)
    return mean, 2.571 * std / np.sqrt(values.shape[0])


def stack_histories(fold_results: list[dict[str, object]], key: str) -> np.ndarray:
    max_len = max(len(result["history"][key]) for result in fold_results)
    rows = []
    for result in fold_results:
        values = np.asarray(result["history"][key], dtype=float)
        if len(values) < max_len:
            values = np.pad(values, (0, max_len - len(values)), mode="edge")
        rows.append(values)
    return np.vstack(rows)


def save_history_csv(fold_results: list[dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "heldout_subject",
            "epoch",
            "heldout_macro_f1",
            "heldout_accuracy",
            "source_val_macro_f1",
            "source_val_accuracy",
            "source_domain_loss",
            "source_domain_accuracy",
            "grl_alpha",
        ])
        for result in fold_results:
            subject = result["heldout_subject"]
            history = result["history"]
            for idx, epoch in enumerate(history["epoch"]):
                writer.writerow([
                    subject,
                    epoch,
                    history["heldout_macro_f1"][idx],
                    history["heldout_accuracy"][idx],
                    history["source_val_macro_f1"][idx],
                    history["source_val_accuracy"][idx],
                    history["source_domain_loss"][idx],
                    history["source_domain_accuracy"][idx],
                    history["grl_alpha"][idx],
                ])


def save_loso_stability_plot(fold_results: list[dict[str, object]], output_path: Path) -> None:
    max_len = max(len(result["history"]["epoch"]) for result in fold_results)
    epochs = np.arange(1, max_len + 1, dtype=float)
    macro_rows = []
    for result in fold_results:
        values = np.asarray(result["history"]["heldout_macro_f1"], dtype=float)
        selected_idx = int(result["selected_epoch"]) - 1
        selected_value = float(result["selected_heldout_macro_f1"])
        values = values.copy()
        values[selected_idx:] = selected_value
        if len(values) < max_len:
            values = np.pad(values, (0, max_len - len(values)), mode="edge")
        macro_rows.append(values)
    macro = np.vstack(macro_rows)
    domain_loss = stack_histories(fold_results, "source_domain_loss")
    domain_acc = stack_histories(fold_results, "source_domain_accuracy")
    alpha = stack_histories(fold_results, "grl_alpha").mean(axis=0)

    macro_mean, macro_ci = mean_ci(macro)
    loss_mean, loss_ci = mean_ci(domain_loss)
    acc_mean, acc_ci = mean_ci(domain_acc)
    n_source_domains = int(fold_results[0]["n_source_domains"])
    domain_loss_chance = float(np.log(n_source_domains))
    domain_acc_chance = float(1.0 / n_source_domains)

    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.91, bottom=0.10, hspace=0.26)

    axes[0].plot(epochs, macro_mean, color="#1f77b4", linewidth=1.8)
    axes[0].fill_between(epochs, macro_mean - macro_ci, macro_mean + macro_ci, color="#1f77b4", alpha=0.18, linewidth=0)
    axes[0].set_ylabel("Held-out\nMacro-F1")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_title("(a) LOSO held-out gesture performance")

    axes[1].plot(epochs, loss_mean, color="#ff7f0e", linewidth=1.7, label="Mean")
    axes[1].fill_between(epochs, loss_mean - loss_ci, loss_mean + loss_ci, color="#ff7f0e", alpha=0.18, linewidth=0, label="95% CI")
    axes[1].axhline(domain_loss_chance, color="#4b5563", linestyle="--", linewidth=1.1, label=f"log({n_source_domains})")
    axes[1].set_ylabel("Source-domain\nloss")
    axes[1].set_title("(b) Domain-discrimination loss on source validation")
    axes[1].legend(frameon=False, ncol=3, loc="upper right")

    axes[2].plot(epochs, acc_mean, color="#2ca02c", linewidth=1.7, label="Domain accuracy")
    axes[2].fill_between(epochs, acc_mean - acc_ci, acc_mean + acc_ci, color="#2ca02c", alpha=0.18, linewidth=0)
    axes[2].plot(epochs, alpha, color="#9467bd", linewidth=1.4, linestyle=":", label="GRL coefficient")
    axes[2].axhline(domain_acc_chance, color="#4b5563", linestyle="--", linewidth=1.1, label=f"Chance 1/{n_source_domains}")
    axes[2].set_ylabel("Value")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].set_title("(c) Source-domain reliability and GRL schedule")
    axes[2].legend(frameon=False, ncol=3, loc="upper right")

    for ax in axes:
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("LOSO-fold averaged GRL training stability of ADANN", fontsize=10)
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOSO-fold averaged ADANN-GRL stability curves")
    parser.add_argument("--csv_dir", type=Path, default=PROJECT_ROOT / "datasets" / "gesture_csv")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "outputs" / "adann_grl_loso_stability")
    parser.add_argument("--paper_figure", type=Path, default=PROJECT_ROOT / "V6" / "Figure25.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--feature_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--domain_loss_weight", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "outputs" / ".matplotlib").mkdir(parents=True, exist_ok=True)

    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    X, y, subjects = load_data(args.csv_dir)
    handcrafted = extract_handcrafted_features(X)
    print(f"Loaded data: X={X.shape}, feature_dim={handcrafted.shape[1]}, subjects={sorted(np.unique(subjects).tolist())}", flush=True)

    fold_results = []
    for fold_subject in sorted(np.unique(subjects)):
        set_all_seeds(args.seed + int(fold_subject))
        print(f"Running LOSO fold: held-out subject {int(fold_subject)}", flush=True)
        result = train_loso_fold(
            int(fold_subject),
            handcrafted,
            y,
            subjects,
            seed=args.seed,
            val_size=args.val_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            feature_size=args.feature_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            domain_loss_weight=args.domain_loss_weight,
            patience=args.patience,
            device=device,
        )
        final_f1 = result["history"]["heldout_macro_f1"][-1]
        final_domain_acc = result["history"]["source_domain_accuracy"][-1]
        print(
            f"  final held-out Macro-F1={final_f1:.4f}, "
            f"selected={result['selected_heldout_macro_f1']:.4f} "
            f"(epoch {result['selected_epoch']}), "
            f"source-domain acc={final_domain_acc:.4f}",
            flush=True,
        )
        fold_results.append(result)

    csv_path = args.output_dir / "adann_grl_loso_stability_history.csv"
    save_history_csv(fold_results, csv_path)
    figure_path = args.output_dir / "adann_grl_loso_stability_panel.png"
    save_loso_stability_plot(fold_results, figure_path)

    summary = {
        "seed": args.seed,
        "csv_dir": str(args.csv_dir),
        "n_samples": int(len(X)),
        "n_subjects": int(len(np.unique(subjects))),
        "n_classes": int(len(np.unique(y))),
        "epochs": int(args.epochs),
        "early_stopping_patience": int(args.patience),
        "source_validation_size": float(args.val_size),
        "fold_results": fold_results,
        "outputs": {
            "history_csv": str(csv_path),
            "figure": str(figure_path),
            "paper_figure": str(args.paper_figure),
        },
    }
    summary_path = args.output_dir / "adann_grl_loso_stability_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    if args.paper_figure:
        args.paper_figure.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(figure_path, args.paper_figure)

    print(f"Saved history: {csv_path}", flush=True)
    print(f"Saved figure: {figure_path}", flush=True)
    print(f"Copied paper figure: {args.paper_figure}", flush=True)
    print(f"Saved summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
