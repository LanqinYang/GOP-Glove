# === Publication-style pooled confusion matrix (LOSO-Full) ===
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patheffects as pe
import os, glob
import seaborn as sns

# 1) Input: 6-fold confusion matrices (rows=true class, cols=predicted class)
# Load from CSVs automatically
base_dir = os.path.dirname(__file__)
cm_glob = os.path.join(base_dir, "output", "confusion_matrix_fold*.csv")
cm_files = sorted(glob.glob(cm_glob))
if not cm_files:
	raise FileNotFoundError(f"No confusion matrices found at pattern: {cm_glob}")
confusions = [np.loadtxt(fp, delimiter=",") for fp in cm_files]
for i, arr in enumerate(confusions):
	if arr.shape != (11, 11):
		raise ValueError(f"{cm_files[i]} has shape {arr.shape}, expected (11, 11)")

# Customize labels and output naming here
labels = ["G0","G1","G2","G3","G4","G5","G6","G7","G8","G9","Static"]
best_model = "BestModel"  # change to your model name, e.g., "DA-LGBM"
topk = 6
out_dir = os.path.join(base_dir, "output", "plots")
os.makedirs(out_dir, exist_ok=True)
outfile = os.path.join(out_dir, f"fig_confusion_LOSO_{best_model}")

# 2) Aggregate and row-normalize to percentages within each true class
C = np.sum(confusions, axis=0).astype(float)
# supports per true class (before zero-fix for normalization)
supports = C.sum(axis=1).astype(int)
row_sum = supports.reshape(-1, 1).astype(float)
row_sum[row_sum == 0] = 1.0
P = C / row_sum  # in [0,1]

# 3) Select Top-K off-diagonal misclassifications
n = P.shape[0]
mask = np.ones_like(P, dtype=bool)
np.fill_diagonal(mask, False)
pairs = np.stack(np.where(mask), axis=1)  # (i, j)
vals  = P[mask]
order = np.argsort(vals)[::-1]
pairs_top = pairs[order][:topk]
vals_top  = vals[order][:topk]

# 4) Plot: main heatmap + right bar chart for Top misclassifications
plt.rcParams.update({
	"figure.dpi": 300,
	"font.size": 12,  # base font size
	"axes.linewidth": 0.9,
})
# Larger figure; left panel made wider to match visual weight of right panel
fig = plt.figure(figsize=(11.2, 7.6))
gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1.8, 1.0], wspace=0.32)

# Main heatmap (seaborn with soft colormap and cell borders)
ax = fig.add_subplot(gs[0,0])
sns.set_style("white")
hm = sns.heatmap(
	P,
	ax=ax,
	cmap="Blues",
	vmin=0.0,
	vmax=1.0,
	square=True,
	linewidths=0.6,
	linecolor="white",
	cbar=True,
	cbar_kws=dict(orientation="horizontal", fraction=0.08, pad=0.18),
	xticklabels=labels,
	yticklabels=[f"{labels[i]} (n={supports[i]})" for i in range(n)],
)
ax.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor")
ax.set_xlabel("Predicted", fontsize=14, labelpad=8)
ax.set_ylabel("True", fontsize=14, labelpad=8)
# Tick label size
ax.tick_params(axis="both", labelsize=12, pad=4)
# Borders
for spine in ax.spines.values():
	spine.set_linewidth(0.9)
ax.set_aspect("equal")

# Annotate only diagonal and Top-K off-diagonal cells with subtle stroke
fmt = lambda p: f"{p*100:.1f}%"
anno_fs = 9  # unified 9pt
stroke_w = 1.4
for i in range(n):
	v = P[i,i]
	fg = "black" if v < 0.55 else "white"
	bg = "white" if fg == "black" else "black"
	txt = ax.text(i + 0.5, i + 0.5, fmt(v), ha="center", va="center",
				 fontsize=anno_fs, color=fg, fontweight="bold")
	txt.set_path_effects([pe.withStroke(linewidth=stroke_w, foreground=bg)])
for (i,j), v in zip(pairs_top, vals_top):
	fg = "black" if P[i,j] < 0.55 else "white"
	bg = "white" if fg == "black" else "black"
	txt = ax.text(j + 0.5, i + 0.5, fmt(v), ha="center", va="center",
				 fontsize=anno_fs, color=fg)  # regular (non-bold)
	txt.set_path_effects([pe.withStroke(linewidth=stroke_w, foreground=bg)])

# Configure seaborn colorbar
cbar = hm.collections[0].colorbar
cbar.set_label("Row-normalized (%)", fontsize=12, labelpad=6)
cbar.ax.tick_params(labelsize=11)
cbar.set_ticks([0,0.25,0.5,0.75,1.0])
cbar.set_ticklabels([0,25,50,75,100])

# Right: Top misclassification pairs (journal-style palette)
axr = fig.add_subplot(gs[0,1])
show = list(zip([f"{labels[i]} → {labels[j]}" for i,j in pairs_top], (vals_top*100)))
y = np.arange(len(show))[::-1]
values = [v for _, v in show]
# ColorBrewer Set2 palette (soft, journal-style)
palette = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3"]
bar_colors = [palette[i % len(palette)] for i in range(len(values))]
axr.barh(y, values, height=0.6, color=bar_colors, edgecolor="#4D4D4D")
axr.set_yticks(y); axr.set_yticklabels([s for s,_ in show], fontsize=12)
axr.set_xlabel("Percent of true class (%)", fontsize=13, labelpad=6)
axr.tick_params(axis="x", labelsize=12)
# leave extra right margin for value text
xmax = max(5, float(np.ceil(max(values)/5)*5))
axr.set_xlim(0, xmax + 8)
for yi, v in zip(y, values):
	axr.text(v + 0.8, yi, f"{v:.1f}%", va="center", ha="left", fontsize=12)
axr.set_title("Top misclassifications (within-row %)", fontsize=13, pad=8)
for spine in axr.spines.values():
	spine.set_linewidth(0.9)

# Final layout tweaks
fig.subplots_adjust(left=0.07, right=0.985, bottom=0.10, top=0.96)
for ext in ("pdf","svg"):
	plt.savefig(f"{outfile}.{ext}", bbox_inches="tight")
plt.close()
