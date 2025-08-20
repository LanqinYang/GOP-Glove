# Requirements: numpy pandas matplotlib scipy
# pip install numpy pandas matplotlib scipy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t, wilcoxon
from matplotlib.transforms import blended_transform_factory

np.random.seed(7)
plt.rcParams.update({"figure.dpi": 160, "font.size": 11})

# ===== 你的每模型6折分数（已填入） =====
scores = {
    "1D-CNN": {
        "macro_f1": [0.9530, 0.4100, 0.8475, 0.7854, 0.6539, 0.7571],
        "accuracy": [0.9545, 0.4727, 0.8545, 0.8182, 0.6727, 0.7818],
    },
    "Transformer Encoder": {
        "macro_f1": [0.8895, 0.4605, 0.7752, 0.8668, 0.7253, 0.7823],
        "accuracy": [0.8909, 0.5273, 0.7909, 0.8818, 0.7455, 0.8182],
    },
    "XGBoost": {
        "macro_f1": [0.9434, 0.4491, 0.8259, 0.7433, 0.7194, 0.8065],
        "accuracy": [0.9455, 0.4909, 0.8364, 0.7727, 0.7364, 0.8273],
    },
    "LightGBM": {
        "macro_f1": [0.8808, 0.5908, 0.7948, 0.7294, 0.7240, 0.7303],
        "accuracy": [0.8909, 0.6545, 0.8273, 0.7636, 0.7455, 0.7455],
    },
    "ADANN": {
        "macro_f1": [0.8883, 0.5551, 0.8475, 0.8675, 0.7100, 0.7587],
        "accuracy": [0.9091, 0.6273, 0.8545, 0.8818, 0.7545, 0.7909],
    },
    "DA-LGBM": {
        "macro_f1": [0.8782, 0.5877, 0.9266, 0.8331, 0.8018, 0.7995],
        "accuracy": [0.9000, 0.6364, 0.9273, 0.8636, 0.8091, 0.8182],
    },
}

# ========== 统计汇总 ==========
def summarize(a):
    a = np.asarray(a, float); n=len(a)
    mean=a.mean(); sd=a.std(ddof=1)
    tval=t.ppf(0.975, n-1)  # df=5
    ci_hw=tval*sd/np.sqrt(n)
    return mean, sd, mean-ci_hw, mean+ci_hw, ci_hw

def table(metric):
    rows=[]
    for m in scores:
        vals=scores[m].get(metric, [])
        if len(vals)==6:
            mean,sd,lo,hi,ci=summarize(vals)
            rows.append(dict(model=m, metric=metric, mean=mean, sd=sd, ci_low=lo, ci_high=hi, n=6))
    return pd.DataFrame(rows)

tbl_f1 = table("macro_f1")
tbl_acc = table("accuracy")

# 顺序：按 Macro-F1 均值降序；Accuracy 复用同序
order = tbl_f1.sort_values("mean", ascending=False)["model"].tolist()
tbl_f1 = tbl_f1.set_index("model").loc[order].reset_index()
tbl_acc = tbl_acc.set_index("model").reindex(order).reset_index()

# ========== Wilcoxon + Holm（结果写入CSV；默认不在图上显示星号） ==========
def holm(pvals):
    pvals=np.array(pvals,float); m=len(pvals)
    ord_idx=np.argsort(pvals); adj=np.zeros(m)
    running=0.0
    for rank, k in enumerate(ord_idx):
        adj_val=min(pvals[k]*(m-rank),1.0)
        running=max(running, adj_val)
        adj[k]=running
    return adj

def signif_table(base="DA-LGBM"):
    rows=[]
    base_f1=np.asarray(scores[base]["macro_f1"], float)
    for m in order:
        if m==base: continue
        x=np.asarray(scores[m]["macro_f1"], float)
        try:
            stat,p=wilcoxon(base_f1, x, alternative="greater", zero_method="wilcox")
        except ValueError:
            p=1.0
        rows.append(dict(model=m, p=p))
    if not rows: return pd.DataFrame()
    df=pd.DataFrame(rows)
    df["p_adj"]=holm(df["p"].values)
    def star(p):
        return "***" if p<1e-3 else ("**" if p<1e-2 else ("*" if p<0.05 else ""))
    df["stars"]=df["p_adj"].apply(star)
    return df

sig_df = signif_table(base="DA-LGBM")
sig_df.to_csv("pairwise_significance.csv", index=False)
print("Pairwise vs DA-LGBM (Wilcoxon, Holm-adjusted):\n", sig_df, "\n")


def point_ci_plot(tbl, metric_key, filename,
                  order=None,
                  title="LOSO-Full (CPU, non-Arduino)",
                  xlabel=None,
                  meta="Protocol: LOSO (n=6); window=100×5@50 Hz; resample→standardize; error bars=95% CI from per-fold SD (df=5, t=2.571)",
                  add_stars=False,  # 如需在y轴标签加星号（p_adj<0.05），设True并传入 sig_df
                  sig_df=None,
                  show_values=True, decimals=4):
    if xlabel is None:
        xlabel = metric_key.replace("_"," ").title() + " (mean ± 95% CI)"

    # 留足右侧空间专门放数值列（0.82可按需要微调）
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    plt.subplots_adjust(right=0.82)

    # 排序
    if order is None:
        order = tbl.sort_values("mean", ascending=False)["model"].tolist()
    tbl = tbl.set_index("model").loc[order].reset_index()

    # 反向Y：最优在上
    y_pos = np.arange(len(tbl))[::-1]
    means = tbl["mean"].values[::-1]
    ci_lo = tbl["ci_low"].values[::-1]
    ci_hi = tbl["ci_high"].values[::-1]
    models = tbl["model"].values[::-1]

    # 彩色调色板（tab10），每个模型独立着色
    cmap = plt.get_cmap('tab10')
    palette = [cmap(i % 10) for i in range(len(models))]

    # 原始点（轻抖动，使用各自颜色）
    for i, m in enumerate(models):
        vals = np.asarray(scores[m][metric_key], float)
        jitter = (np.random.rand(len(vals))-0.5)*0.12
        ax.scatter(vals, np.full_like(vals, y_pos[i])+jitter,
                   s=30, alpha=0.75,
                   facecolors=palette[i], edgecolors="#333333",
                   linewidths=0.35, zorder=3)

    # 均值 + 95%CI（逐行绘制以套用各自颜色）
    for i in range(len(models)):
        mu = means[i]
        lo = mu - (mu - ci_lo[i])
        hi = (ci_hi[i] - mu)
        ax.errorbar(mu, y_pos[i],
                    xerr=[[mu - ci_lo[i]], [ci_hi[i] - mu]],
                    fmt='o', ms=6, capsize=4, elinewidth=1.2, lw=1.2,
                    color=palette[i], ecolor=palette[i],
                    markerfacecolor=palette[i], markeredgecolor=palette[i],
                    zorder=4)

    # 右侧数值：只标一个均值（4位小数），固定在坐标系右侧 1.01 位置
    if show_values:
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        for i, mu in enumerate(means):
            ax.text(1.01, y_pos[i], f"{mu:.{decimals}f}",
                    transform=trans, ha="left", va="center",
                    fontsize=10, color="0.25", clip_on=False)

    # y 轴标签（如需方案B：仅对显著的模型加星）
    labels = list(models)
    if add_stars and (sig_df is not None) and not sig_df.empty:
        star_map = {r.model: r.stars for r in sig_df.itertuples()}
        star_map.setdefault("DA-LGBM","")
        labels = [m + (f" {star_map.get(m,'')}" if star_map.get(m,"") else "") for m in models]

    ax.set_yticks(y_pos); ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.0)  # 主图范围仍是0..1；右侧数值用Axes坐标放，不受xlim影响
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # 标题 + meta 行
    ax.set_title(f"{title} — {xlabel}", loc="left", fontsize=12, pad=10)
    if meta:
        ax.text(0.0, 1.02, meta, transform=ax.transAxes, fontsize=9, color="0.35")

    plt.tight_layout()
    plt.savefig(filename + ".pdf", bbox_inches="tight")
    plt.savefig(filename + ".svg", bbox_inches="tight")
    print("Saved:", filename + ".pdf", "and", filename + ".svg")


# —— 绘制 —— 
# (a) Macro-F1：决定排序
order_macro = tbl_f1.sort_values("mean", ascending=False)["model"].tolist()
point_ci_plot(
    tbl_f1, metric_key="macro_f1",
    filename="fig_LOSO-Full_macroF1_pointCI",
    order=order_macro,
    title="LOSO-Full (CPU, non-Arduino)",
    xlabel="Macro-F1 (mean ± 95% CI)",
    meta=None,
    add_stars=False,  # 若要在标签加星，改True并传sig_df
    sig_df=sig_df,
    show_values=True, decimals=4
)

# (b) Accuracy：沿用同序
point_ci_plot(
    tbl_acc, metric_key="accuracy",
    filename="fig_LOSO-Full_accuracy_pointCI",
    order=order_macro,
    title="LOSO-Full (CPU, non-Arduino)",
    xlabel="Accuracy (mean ± 95% CI)",
    meta=None,
    add_stars=False,
    sig_df=sig_df,
    show_values=True, decimals=4
)

# —— 导出主汇总表 —— 
pd.concat([
    tbl_f1.assign(metric="macro_f1"),
    tbl_acc.assign(metric="accuracy")
]).reset_index(drop=True).to_csv("loso_fold_summary.csv", index=False)
print("Saved: loso_fold_summary.csv")
