#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 纯净版：从一组(6折)混淆矩阵CSV -> 计算每类F1 -> 画 mean±95%CI 水平误差线
# 口径：NaN 统一当作 0（该折该类全错），y轴标签固定 n=6，手势10标为“Gesture 10 (Static)”

import os, re, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

plt.rcParams.update({
    "figure.dpi": 170,
    "font.size": 11,
    "axes.edgecolor": "0.2",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def _strip_gesture(s: str) -> str:
    s = str(s)
    s = re.sub(r'(?i)^gesture[_-]?', '', s)  # Remove leading Gesture_/gesture_
    return s

def _read_cm(path: str) -> pd.DataFrame:
    """Read a confusion matrix CSV robustly as a square numeric matrix.
    Supports files without headers and without index labels.
    If an extra index/column is present, try to drop it to enforce square shape.
    """
    df = pd.read_csv(path, header=None)
    r, c = df.shape
    if r != c:
        if c == r + 1:
            # extra first column (index-like)
            df = df.iloc[:, 1:]
        elif r == c + 1:
            # extra first row (header-like)
            df = df.iloc[1:, :]
        else:
            raise ValueError(f"Confusion matrix not square: {df.shape} at {path}")
    n = df.shape[0]
    df.index = [str(i) for i in range(n)]
    df.columns = [str(i) for i in range(n)]
    return df

def _per_class_f1(cm_df: pd.DataFrame) -> pd.DataFrame:
    """行真值、列预测；返回该折各类 precision/recall/f1。"""
    cm = cm_df.values.astype(float)
    labels = list(cm_df.index)
    tp = np.diag(cm)
    col_sum = cm.sum(axis=0)
    row_sum = cm.sum(axis=1)
    fp = col_sum - tp
    fn = row_sum - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall    = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1        = np.divide(2*precision*recall, precision+recall,
                          out=np.zeros_like(precision), where=(precision+recall)!=0)
    return pd.DataFrame({"gesture": labels, "precision": precision, "recall": recall, "f1": f1})

def _mean_sd_ci(x: np.ndarray):
    x = np.asarray(x, float)
    m = float(np.mean(x)) if len(x) else np.nan
    if len(x) <= 1:
        return m, 0.0, 0.0
    sd = float(np.std(x, ddof=1))
    ci = float(t.ppf(0.975, len(x)-1) * sd / np.sqrt(len(x)))
    return m, sd, ci

def plot_from_confusions(files, out_prefix, title="Per-class performance (LOSO) – DA-LGBM",
                         n_text=6, macro_override=None):
    # 读取 6 折并拼成长表
    long_rows = []
    for i, fp in enumerate(sorted(files), 1):
        cm = _read_cm(fp)
        met = _per_class_f1(cm)
        met["fold"] = i
        long_rows.append(met)
    df = pd.concat(long_rows, ignore_index=True)

    # NaN -> 0（你的口径）
    for k in ("precision","recall","f1"):
        df[k] = df[k].astype(float).fillna(0.0)

    # 聚合到每类
    stats = []
    for g, sub in df.groupby("gesture", dropna=False):
        vals = sub["f1"].values
        m, sd, ci = _mean_sd_ci(vals)
        stats.append({"gesture": str(g), "mean": m, "sd": sd, "ci": ci, "n": len(vals)})
    stat = pd.DataFrame(stats)

    # 手势10重命名
    stat["yname"] = stat["gesture"].apply(
        lambda s: "Gesture 10 (Static)" if s == "10" else f"Gesture {s}"
    )
    stat["yname"] = stat["yname"].astype(str) + "\n" + f"n={n_text}"

    # 按均值升序（最难在上面）
    stat = stat.sort_values("mean", ascending=True).reset_index(drop=True)

    # 计算整体 Macro-F1（按“每折取各类F1均值，再在折上取平均”的定义）
    per_fold_macro = df.groupby("fold")["f1"].mean()
    macro_f1 = float(per_fold_macro.mean())
    if macro_override is not None:
        macro_f1 = float(macro_override)

    # Plot (colored per class): mean ± 95% CI, per-row color using tab20
    y = np.arange(len(stat))
    means = stat["mean"].values
    ci_hw = stat["ci"].values

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    cmap = plt.cm.get_cmap("tab20", len(stat))
    colors = [cmap(i) for i in range(len(stat))]
    for i in range(len(stat)):
        ax.errorbar(
            means[i], y[i], xerr=ci_hw[i],
            fmt="o", ms=6,
            color=colors[i], ecolor=colors[i],
            markeredgecolor="k", elinewidth=1.3, capsize=4, lw=1.3, zorder=3
        )

    ax.set_yticks(y)
    ax.set_yticklabels(stat["yname"].tolist())
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Per-class F1 (LOSO, mean ± 95% CI)")
    ax.grid(axis="x", linestyle=(0, (2, 2)), alpha=0.35)
    ax.set_title(title, pad=14)

    # Right-side mean labels (match point color)
    for yi, m, c in zip(y, means, colors):
        if np.isfinite(m):
            ax.text(1.005, yi, f"{m:.4f}", va="center", ha="left",
                    transform=ax.get_yaxis_transform(), fontsize=10, color=c)

    # 宏F1竖线+文字
    ax.axvline(macro_f1, ls="--", color="tab:red", alpha=0.9)
    ax.annotate(f"Macro-F1 = {macro_f1:.3f}",
                xy=(macro_f1, 1.02), xycoords=("data", "axes fraction"),
                xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", color="tab:red")

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{out_prefix}.{ext}", bbox_inches="tight")
    print("Saved:", f"{out_prefix}.pdf|.svg|.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cm", nargs="*", help="逐个写入CSV路径")
    ap.add_argument("--cm_glob", help="或使用通配，例如 src/.../confusion_matrix_fold*.csv")
    ap.add_argument("--out", required=True, help="输出文件前缀（不带扩展名）")
    ap.add_argument("--title", default="Per-class performance (LOSO) – DA-LGBM")
    ap.add_argument("--macro_f1", type=float, default=None, help="可覆盖宏F1虚线位置")
    args = ap.parse_args()

    files = []
    if args.cm:
        files += args.cm
    if args.cm_glob:
        files += sorted(glob.glob(args.cm_glob))
    if not files:
        raise SystemExit("No confusion-matrix CSV provided. Use --cm ... or --cm_glob ...")

    plot_from_confusions(files, out_prefix=args.out, title=args.title, macro_override=args.macro_f1)

if __name__ == "__main__":
    main()
