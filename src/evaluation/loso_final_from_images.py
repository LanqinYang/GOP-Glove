import os
import math
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class MetricStats:
    mean: float
    sd: float
    n: int

    def ci95_halfwidth(self) -> float:
        """Compute 95% CI half-width using t_{0.975,5} and n=6 (df=5)."""
        # The task specifies n=6, df=5, t ≈ 2.571
        t_975_df5 = 2.571
        return t_975_df5 * (self.sd / math.sqrt(self.n))


# Mapping to unified model naming used in figures
MODEL_NAME_MAP: Dict[str, str] = {
    "Transformer_Encoder": "Transformer Encoder",
    "1D_CNN": "1D-CNN",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "ADANN": "ADANN",
    "ADANN_LightGBM": "DA-LGBM",
}


def get_image_extracted_numbers() -> Dict[str, Dict[str, MetricStats]]:
    """
    Numbers transcribed from the six provided LOSO-final summary images.
    All values are means in [0,1] and fold SDs (across n=6 folds).

    Keys:
      model -> metric -> MetricStats
    Metrics supported: "macro_f1", "accuracy"
    """
    n = 6

    # Values read from the screenshots (mean ± SD)
    return {
        "1D_CNN": {
            "macro_f1": MetricStats(mean=0.7345, sd=0.1710, n=n),
            "accuracy": MetricStats(mean=0.7591, sd=0.1532, n=n),
        },
        "Transformer_Encoder": {
            "macro_f1": MetricStats(mean=0.7500, sd=0.1410, n=n),
            "accuracy": MetricStats(mean=0.7758, sd=0.1219, n=n),
        },
        "XGBoost": {
            "macro_f1": MetricStats(mean=0.7479, sd=0.1516, n=n),
            "accuracy": MetricStats(mean=0.7682, sd=0.1399, n=n),
        },
        "ADANN": {
            "macro_f1": MetricStats(mean=0.7712, sd=0.1151, n=n),
            "accuracy": MetricStats(mean=0.8030, sd=0.0944, n=n),
        },
        "LightGBM": {
            "macro_f1": MetricStats(mean=0.7417, sd=0.0871, n=n),
            "accuracy": MetricStats(mean=0.7712, sd=0.0736, n=n),
        },
        "ADANN_LightGBM": {
            "macro_f1": MetricStats(mean=0.8046, sd=0.1067, n=n),
            "accuracy": MetricStats(mean=0.8258, sd=0.0944, n=n),
        },
    }


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(data: Dict[str, Dict[str, MetricStats]], out_csv: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model_key, metrics in data.items():
        for metric_name, stats in metrics.items():
            rows.append(
                {
                    "model": MODEL_NAME_MAP.get(model_key, model_key),
                    "metric": metric_name,
                    "mean": float(stats.mean),
                    "sd": float(stats.sd),
                    "n": int(stats.n),
                }
            )
    df = pd.DataFrame(rows)
    # Stable sort by model name for CSV readability
    df = df.sort_values(by=["model", "metric"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    return df


def plot_metric_bar(
    df: pd.DataFrame,
    metric: str,
    out_pdf: str,
    out_svg: str,
    caption_txt_path: str = None,
    order_by: List[str] = None,
    title: str = None,
    label_offset: float = 0.015,
) -> List[str]:
    """Create bar chart with 95% CI error bars in grayscale and export to PDF/SVG.

    Returns the model order used in the plot (list of model names).
    """
    metric_df = df[df["metric"] == metric].copy()

    # Compute ci95 half-widths
    ci_hw = []
    for _, row in metric_df.iterrows():
        stats = MetricStats(mean=row["mean"], sd=row["sd"], n=int(row["n"]))
        ci_hw.append(stats.ci95_halfwidth())
    metric_df["ci_hw"] = ci_hw

    # Determine plotting order
    if order_by is None:
        metric_df = metric_df.sort_values(by="mean", ascending=False)
        model_order = metric_df["model"].tolist()
    else:
        # Reindex to provided order
        metric_df = metric_df.set_index("model").loc[order_by].reset_index()
        model_order = order_by

    # Grayscale colors for bars
    gray_levels = np.linspace(0.25, 0.75, len(metric_df))  # darker to lighter
    bar_colors = [str(gl) for gl in gray_levels]  # Matplotlib grayscale: string in [0,1]

    # Plot
    plt.figure(figsize=(7, 4.5))
    x = np.arange(len(metric_df))
    means = metric_df["mean"].to_numpy()
    yerr = metric_df["ci_hw"].to_numpy()

    bars = plt.bar(x, means, color=bar_colors, edgecolor="black", linewidth=0.8)
    plt.errorbar(x, means, yerr=yerr, fmt="none", ecolor="black", elinewidth=1, capsize=3)

    # Optional title
    if title:
        plt.title(title)

    # Value labels (two decimals) placed above error bar
    for xi, m, err in zip(x, means, yerr):
        y_pos = min(1.0, m + err + label_offset)
        plt.text(
            xi,
            y_pos,
            f"{m:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(x, model_order, rotation=0)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Macro-F1" if metric == "macro_f1" else "Accuracy")
    plt.grid(axis="y", linestyle="--", alpha=0.6, color="#cccccc")
    plt.tight_layout()

    # Save figures
    plt.savefig(out_pdf, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(out_svg, format="svg", dpi=300, bbox_inches="tight")
    plt.close()

    # Optional: caption text
    if caption_txt_path is not None:
        caption = (
            "Figure X. LOSO (mean ± 95% CI) — Macro-F1. "
            "Error bars are 95% CIs converted from per-fold SD using t_{0.975,5} · SD/√6 (n=6). "
            "Models are ordered by mean macro-F1."
        )
        with open(caption_txt_path, "w") as f:
            f.write(caption)

    return model_order


def main(output_dir: str = None, also_plot_accuracy: bool = True) -> None:
    data = get_image_extracted_numbers()

    # Validate ranges and print table for verification
    header = ["model", "metric", "mean", "sd", "n"]
    rows: List[List[object]] = []
    for model, metrics in data.items():
        for metric, stats in metrics.items():
            if not (0.0 <= stats.mean <= 1.0):
                raise ValueError(f"Mean out of bounds for {model}/{metric}: {stats.mean}")
            rows.append([MODEL_NAME_MAP.get(model, model), metric, stats.mean, stats.sd, stats.n])

    # Determine output directory
    if output_dir is None:
        output_dir = os.path.join("outputs", "comprehensive_summary")
    ensure_dir(output_dir)

    # Write CSV
    out_csv = os.path.join(output_dir, "loso_full_model_summary.csv")
    df = write_csv(data, out_csv)

    # Plot macro-f1 primary figure (determine canonical order)
    macro_pdf = os.path.join(output_dir, "fig_model_macroF1.pdf")
    macro_svg = os.path.join(output_dir, "fig_model_macroF1.svg")
    caption_txt = os.path.join(output_dir, "fig_model_macroF1_caption.txt")
    model_order = plot_metric_bar(
        df,
        metric="macro_f1",
        out_pdf=macro_pdf,
        out_svg=macro_svg,
        caption_txt_path=caption_txt,
        order_by=None,
        title="LOSO (mean ± 95% CI) — Macro-F1",
        label_offset=0.015,
    )

    # Optional: accuracy figure
    if also_plot_accuracy:
        acc_pdf = os.path.join(output_dir, "fig_model_accuracy.pdf")
        acc_svg = os.path.join(output_dir, "fig_model_accuracy.svg")
        plot_metric_bar(
            df,
            metric="accuracy",
            out_pdf=acc_pdf,
            out_svg=acc_svg,
            caption_txt_path=None,
            order_by=model_order,  # keep same order as macro-f1
            title="LOSO (mean ± 95% CI) — Accuracy",
            label_offset=0.015,
        )

    # Console print for OCR verification
    table_df = df.pivot_table(index="model", columns="metric", values=["mean", "sd"], aggfunc="first")
    # Flatten multiindex columns
    table_df.columns = [f"{lvl0}_{lvl1}" for lvl0, lvl1 in table_df.columns]
    print("\nOCR verification table (means and SDs):")
    print(table_df.reset_index().to_string(index=False))
    print(f"\nArtifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
