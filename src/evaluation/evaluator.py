"""
通用评估工具
=================

为 `src.training.pipeline` 提供统一的评估入口，支持：
- 计算并保存分类报告、准确率等数值指标
- 绘制并保存混淆矩阵
- 根据训练历史绘制 loss / accuracy 曲线（如果可用）
- 为 LOSO 运行生成汇总可视化
"""

from __future__ import annotations

import os
import csv
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def _ensure_dir(path: str) -> None:
    """确保目录存在。"""
    os.makedirs(path, exist_ok=True)


def _to_numpy(x: Any) -> np.ndarray:
    """将张量/列表等安全转换为 NumPy 数组。"""
    if isinstance(x, np.ndarray):
        return x
    try:
        # 兼容 PyTorch / TensorFlow 张量
        return x.detach().cpu().numpy()  # type: ignore[attr-defined]
    except Exception:
        try:
            return np.asarray(x)
        except Exception:
            raise TypeError(f"无法将对象类型 {type(x)} 转为 numpy 数组")


def _infer_prob_and_labels(
    model: Any, X: Any
) -> (Optional[np.ndarray], np.ndarray):
    """
    根据模型类型自动获得预测概率和标签。

    返回:
        y_proba: (n_samples, n_classes) 或 None
        y_pred:  (n_samples,)
    """
    # 优先使用 sklearn 风格的 predict_proba
    if hasattr(model, "predict_proba"):
        try:
            y_proba = _to_numpy(model.predict_proba(X))
            if y_proba.ndim == 2 and y_proba.shape[1] > 1:
                y_pred = np.argmax(y_proba, axis=1)
                return y_proba, y_pred
        except Exception:
            # 回退到通用 predict
            pass

    # 通用 predict 接口（Keras、LightGBM、包装器等）
    y_raw = _to_numpy(model.predict(X))

    if y_raw.ndim == 2 and y_raw.shape[1] > 1:
        # 视为 [N, num_classes] 的概率或 logits
        y_proba = y_raw
        y_pred = np.argmax(y_raw, axis=1)
        return y_proba, y_pred

    # 视为已经是类别标签
    y_pred = y_raw.astype(int).ravel()
    return None, y_pred


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str,
) -> None:
    """绘制并保存混淆矩阵。"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    # 在每个格子里写数值
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _plot_history_curves(
    history: Any,
    save_dir: str,
    prefix: str,
) -> None:
    """
    根据 Keras/自定义 History 对象绘制训练曲线。

    要求 history.history 为 dict，包含若干 list 序列。
    """
    if history is None or not hasattr(history, "history"):
        return

    hist_dict = getattr(history, "history", None)
    if not isinstance(hist_dict, dict):
        return

    # loss / val_loss 曲线
    if "loss" in hist_dict:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(hist_dict["loss"], label="loss")
        if "val_loss" in hist_dict:
            ax.plot(hist_dict["val_loss"], label="val_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{prefix} - Loss Curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{prefix}_loss_curve.png"))
        plt.close(fig)

    # accuracy / val_accuracy 曲线
    if "accuracy" in hist_dict:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(hist_dict["accuracy"], label="accuracy")
        if "val_accuracy" in hist_dict:
            ax.plot(hist_dict["val_accuracy"], label="val_accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{prefix} - Accuracy Curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{prefix}_accuracy_curve.png"))
        plt.close(fig)


def comprehensive_evaluation(
    model: Any,
    X_test: Any,
    y_test: Any,
    scaler: Any,
    output_dir: str,
    run_name: str,
    history: Optional[Any] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    统一评估接口，供标准训练与 LOSO 训练调用。

    参数
    ----
    model: 训练好的模型（Keras / XGBoost / LightGBM / 自定义包装器均可）
    X_test, y_test: 测试集数据与标签（假定已完成必要的预处理/归一化）
    scaler: 保持接口兼容，目前不在此处再次缩放，避免重复归一化
    output_dir: 评估结果保存目录
    run_name: 本次运行/折次的名称，用于文件前缀
    history: 可选训练历史，用于绘制曲线
    class_names: 类别名称列表；如为 None，则根据标签自动生成
    """
    del scaler  # 避免未使用的参数告警，预处理应在 pipeline 中完成

    _ensure_dir(output_dir)

    X_test_np = _to_numpy(X_test)
    y_test_np = _to_numpy(y_test).astype(int).ravel()

    # 模型预测
    y_proba, y_pred = _infer_prob_and_labels(model, X_test_np)

    # 推断类别数与类别名称
    unique_labels = np.unique(y_test_np)
    n_classes = unique_labels.max() + 1 if unique_labels.size > 0 else 0
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(n_classes)]

    # 计算指标
    acc = float(accuracy_score(y_test_np, y_pred))
    report_dict = classification_report(
        y_test_np,
        y_pred,
        target_names=class_names if len(class_names) >= n_classes else None,
        output_dict=True,
        zero_division=0,
    )
    # 直接取出 macro-F1，方便后续消融/图表使用
    try:
        macro_f1 = float(report_dict.get("macro avg", {}).get("f1-score", 0.0))
    except Exception:
        macro_f1 = 0.0
    cm = confusion_matrix(y_test_np, y_pred, labels=range(n_classes))

    # 保存文本报告
    txt_report_path = os.path.join(output_dir, f"{run_name}_classification_report.txt")
    with open(txt_report_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro-F1: {macro_f1:.4f}\n\n")
        f.write(
            classification_report(
                y_test_np,
                y_pred,
                target_names=class_names if len(class_names) >= n_classes else None,
                zero_division=0,
            )
        )

    # 额外保存一个简洁的 CSV，专门记录 Accuracy 和 Macro-F1（方便做消融图）
    metrics_csv_path = os.path.join(output_dir, f"{run_name}_metrics.csv")
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["run_name", "accuracy", "macro_f1"])
        writer.writerow([run_name, f"{acc:.6f}", f"{macro_f1:.6f}"])

    # 保存混淆矩阵图
    cm_fig_path = os.path.join(output_dir, f"{run_name}_confusion_matrix.png")
    _plot_confusion_matrix(
        cm,
        class_names=class_names,
        save_path=cm_fig_path,
        title=f"{run_name} - Confusion Matrix",
    )

    # 保存训练曲线（如有）
    _plot_history_curves(history, output_dir, prefix=run_name)

    eval_report: Dict[str, Any] = {
        "run_name": run_name,
        "test_accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }

    # 如有概率输出，保存一份 Numpy 文件（便于后续分析）
    if y_proba is not None:
        proba_path = os.path.join(output_dir, f"{run_name}_proba.npy")
        np.save(proba_path, y_proba)
        eval_report["proba_path"] = proba_path

    return eval_report


def generate_loso_summary_plots(
    history: Optional[Any],
    all_fold_evaluations: List[Dict[str, Any]],
    output_dir: str,
    run_name: str,
) -> None:
    """
    为 LOSO 训练生成简单的可视化汇总图：
    - 各 fold 准确率柱状图
    -（可选）总体训练曲线（如果传入了 history）
    """
    _ensure_dir(output_dir)

    if not all_fold_evaluations:
        return

    fold_accuracies = [
        float(ev.get("test_accuracy", 0.0)) for ev in all_fold_evaluations
    ]
    fold_ids = list(range(1, len(fold_accuracies) + 1))

    # 1) 各 fold 准确率柱状图
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(fold_ids, fold_accuracies, color="#4C72B0")
    ax.set_xlabel("LOSO Fold Index")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{run_name} - LOSO Fold Accuracies")
    ax.set_xticks(fold_ids)
    ax.set_ylim(0.0, 1.0)
    for i, acc in enumerate(fold_accuracies, start=1):
        ax.text(i, acc + 0.01, f"{acc:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{run_name}_loso_fold_accuracies.png"))
    plt.close(fig)

    # 2) 如有总体 history，则再绘制一份总的训练曲线
    _plot_history_curves(history, output_dir, prefix=f"{run_name}_overall")

