from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix
from sklearn.preprocessing import normalize


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_confusion_normalized(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
    cmn = normalize(cm, norm="l1", axis=1)
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cmn,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=["Benign", "Attack"],
        yticklabels=["Benign", "Attack"],
        vmin=0.0,
        vmax=1.0,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(4, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_pr(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(4, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, n_bins: int = 10) -> None:
    from metrics import reliability_bins

    rb = reliability_bins(np.asarray(y_true), np.asarray(y_prob), n_bins)
    conf = [c for c, n in zip(rb["confidence"], rb["count"]) if n]
    acc = [a for a, n in zip(rb["accuracy"], rb["count"]) if n]
    cnt = [n for n in rb["count"] if n]
    plt.figure(figsize=(4, 4))
    # Empty bins are omitted rather than drawn at (midpoint, 0). Marker size shows bin count.
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.plot(conf, acc, color="#2b6cb0", alpha=0.6)
    sizes = 20 + 80 * np.log10(np.asarray(cnt, dtype=float) + 1)
    plt.scatter(conf, acc, s=sizes, color="#2b6cb0", zorder=3)
    for c, a, n in zip(conf, acc, cnt):
        plt.annotate(str(n), (c, a), textcoords="offset points", xytext=(4, 4), fontsize=6)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed attack rate")
    plt.title("Reliability Diagram (n per bin)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_score_histogram(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    plt.figure(figsize=(6, 4))
    sns.histplot(y_prob[y_true == 0], bins=30, stat="density", color="#2b6cb0", alpha=0.5, label="Benign")
    sns.histplot(y_prob[y_true == 1], bins=30, stat="density", color="#c53030", alpha=0.5, label="Attack")
    plt.xlabel("Predicted Probability (Attack)")
    plt.ylabel("Density")
    plt.title("Score Distribution by Class")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_feature_importance(feature_names: np.ndarray, importances: np.ndarray, out_path: Path, top_k: int = 30) -> None:
    order = np.argsort(importances)[::-1][:top_k]
    names = [str(feature_names[i]) for i in order]
    vals = importances[order]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=vals, y=names, color="#2b6cb0")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"Top-{top_k} Random Forest Feature Importances")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_tactic_recall(tactic_breakdown: Dict[str, Dict[str, float]], out_path: Path) -> None:
    if not tactic_breakdown:
        return
    items = sorted(tactic_breakdown.items(), key=lambda kv: kv[1].get("support", 0), reverse=True)
    tactics = [k for k, _ in items]
    recall = [float(v.get("recall", 0.0)) for _, v in items]
    support = [int(v.get("support", 0)) for _, v in items]
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(x=recall, y=tactics, color="#2f855a")
    for i, (r, s) in enumerate(zip(recall, support)):
        ax.text(min(r + 0.02, 0.98), i, f"n={s}", va="center", fontsize=9)
    plt.xlim(0, 1)
    plt.xlabel("Recall")
    plt.ylabel("ATT&CK Tactic (subLabel)")
    plt.title("Per-Tactic Recall (Test Split)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
