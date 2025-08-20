# Enhanced latency vs F1 plot (split 3.2C1), without supported/unsupported,
# clean journal style, colorful, single chart for latency vs F1 plus a separate
# colorful memory bars chart (Flash & RAM).
# Saves to /mnt/data/output/plots/Fig_3_2C1_latency_vs_f1_enhanced and a memory folder.

import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----- Data (hard-coded) -----
rows = [
    # model,                 mean,    sd,     ci_hw,  latency_ms, flash_kb, ram_kb
    ("DA-LGBM",             0.7956,  0.1100, 0.1155,  16.49,      553.40,   47.51),
    ("1D-CNN",              0.7677,  0.2022, 0.2123,  16.28,      350.75,   88.88),
    ("ADANN",               0.7582,  0.1004, 0.1054,  11.17,      384.08,   47.51),
    ("LightGBM",            0.7195,  0.1592, 0.1672,   4.66,      305.73,   51.66),
    ("XGBoost",             0.7029,  0.1710, 0.1796,   9.00,       89.43,   51.59),
    # Transformer has no latency_ms -> exclude from this figure
]
df = pd.DataFrame(rows, columns=[
    "model","mean","sd","ci_hw","latency_ms","flash_kb","ram_kb"
])

# If ci_hw is missing, compute as 2.571*sd/sqrt(6)
mask = df["ci_hw"].isna() if "ci_hw" in df.columns else None
if mask is not None and mask.any():
    df.loc[mask, "ci_hw"] = df.loc[mask, "sd"] * 2.571 / math.sqrt(6.0)

# Preserve original model order for consistent colors
original_order = [r[0] for r in rows]

# Sort by latency asc (fastest at left), then by mean desc for ties
df = df.sort_values(["latency_ms","mean"], ascending=[True, False]).reset_index(drop=True)

# ----- Figure: Latency vs F1 (colorful) -----
plt.rcParams.update({
    "figure.dpi": 180,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(9.6, 5.6))

# Color mapping per model (kept for the memory bars figure below)
palette = plt.get_cmap('tab10').colors
model_to_color = {m: palette[i % len(palette)] for i, m in enumerate(original_order)}
colors = [model_to_color[m] for m in df["model"].tolist()]

# For the F1 vs latency figure, use the explicit sequence matching single-window latency:
# red → purple → green → yellow → blue
custom_colors = ['tab:red', 'tab:purple', 'tab:green', 'gold', 'tab:blue']
num_points = len(df)
color_seq = (custom_colors * ((num_points + len(custom_colors) - 1) // len(custom_colors)))[:num_points]

# Error bars + points (per-point colors)
for i, r in enumerate(df.itertuples()):
    c = color_seq[i]
    ax.errorbar(r.latency_ms, r.mean, yerr=r.ci_hw,
                fmt='o', ms=6, color=c, ecolor=c, capsize=3.5,
                elinewidth=1.2, lw=1.2, alpha=0.95)
    ax.scatter([r.latency_ms], [r.mean], s=40, facecolor=c, edgecolor='black', linewidth=0.6, zorder=3)

# Axis formatting
ax.set_xlabel("Single-window latency on Arduino (ms)")
ax.set_ylabel("Macro-F1 (LOSO–Arduino, mean ± 95% CI)")
ax.set_ylim(0.0, 1.0)
ax.grid(True, axis="both", linestyle="--", alpha=0.3)

# Expand x limits with padding
x = df["latency_ms"].values
xmin, xmax = x.min(), x.max()
pad = max(0.5, 0.06*(xmax - xmin if xmax > xmin else 10.0))
ax.set_xlim(xmin - pad, xmax + 8*pad/4)

# Improved non-overlapping colored labels with left/right placement and connectors
y = df["mean"].values
xrange = (xmax - xmin) if xmax > xmin else 10.0
dx = 0.04 * xrange
x_med = float(np.median(x))

# Sort labels by y to place with vertical separation
order_idx = np.argsort(y)
placed_y = []
min_sep_y = 0.035  # minimal vertical separation
for idx in order_idx:
    xi = float(x[idx]); yi = float(y[idx]); ci = float(df.loc[idx, "ci_hw"]) if not np.isnan(df.loc[idx, "ci_hw"]) else 0.0
    c = color_seq[idx]
    label = f"{df.loc[idx,'model']} — F1={yi:.4f}, {xi:.2f} ms"
    # Decide side: left if on the right half to reduce clutter, else right
    place_right = xi <= x_med
    tx = xi + dx if place_right else xi - dx
    # Vertical placement with simple repel
    ty = yi
    for py in placed_y:
        if abs(ty - py) < min_sep_y:
            ty = py + min_sep_y if ty <= py else py - min_sep_y
    ty = max(0.02, min(0.98, ty))
    placed_y.append(ty)
    # Connector
    ax.plot([xi, tx], [yi, ty], color=c, alpha=0.7, linewidth=0.8, zorder=2)
    # Colored label
    ax.text(tx, ty, label, va="center", ha=("left" if place_right else "right"), fontsize=9, color=c,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=c, alpha=0.85), clip_on=False)

# Title and save
ax.set_title("LOSO–Arduino — Macro-F1 vs single-window latency")
fig.tight_layout()

outdir = Path("/Users/aqin/Documents/Project/src/evaluation/output/plots/Fig_3_2C1_latency_vs_f1_enhanced")
outdir.mkdir(parents=True, exist_ok=True)
for ext in ("pdf","svg","png"):
    fig.savefig(outdir / f"Fig_3_2C1_latency_vs_f1_enhanced.{ext}", bbox_inches="tight")

# ----- Second figure: colorful memory bars (Flash & RAM) -----
order_by_f1 = df.sort_values('mean', ascending=False)['model'].tolist()
mem_df = df.set_index('model').loc[order_by_f1].reset_index()
labels = mem_df['model'].tolist()
flash = mem_df['flash_kb'].values.astype(float)
ram = mem_df['ram_kb'].values.astype(float)
colors_mem = [model_to_color[m] for m in labels]

fig2, ax2 = plt.subplots(figsize=(9.2, 5.0))
ypos = np.arange(len(labels))
h = 0.36
# Flash bars: solid color
ax2.barh(ypos - h/2, flash, height=h, color=colors_mem, edgecolor='black', linewidth=0.6)
# RAM bars: same hue but hatched + lighter alpha
ax2.barh(ypos + h/2, ram,   height=h, color=colors_mem, edgecolor='black', linewidth=0.6, hatch='///', alpha=0.6)

# Annotate values
for yi, v, c in zip(ypos - h/2, flash, colors_mem):
    ax2.text(v + max(2, 0.02*v), yi, f"{v:.0f}", va='center', ha='left', fontsize=9, color=c)
for yi, v, c in zip(ypos + h/2, ram, colors_mem):
    ax2.text(v + max(2, 0.02*v), yi, f"{v:.0f}", va='center', ha='left', fontsize=9, color=c)

ax2.set_yticks(ypos); ax2.set_yticklabels(labels)
ax2.invert_yaxis()
ax2.set_xlabel('KB')
ax2.set_title('LOSO–Arduino — Memory Footprint (Flash & RAM, compile-time)')
ax2.grid(axis='x', linestyle='--', color='0.85')
from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor='0.3', edgecolor='black', label='Flash (KB)'),
    Patch(facecolor='white', edgecolor='black', hatch='///', label='RAM (KB)')
]
ax2.legend(handles=legend_elems, loc='lower right', frameon=False)


outdir2 = Path("/Users/aqin/Documents/Project/src/evaluation/output/plots/Fig_3_2C2_memory_bars")
outdir2.mkdir(parents=True, exist_ok=True)
fig2.tight_layout()
for ext in ("pdf","svg","png"):
    fig2.savefig(outdir2 / f"Fig_3_2C2_memory_bars.{ext}", bbox_inches="tight")
