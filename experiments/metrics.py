from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass(frozen=True)
class MetricBundle:
    precision: float
    recall: float
    f1: float
    auroc: float
    auprc: float
    brier: float
    ece: float
    confusion: Tuple[int, int, int, int]  # tn, fp, fn, tp
    n: int
    n_pos: int


def _bin_ids(y_prob: np.ndarray, n_bins: int) -> np.ndarray:
    # np.digitize puts p == 1.0 into bin n_bins, which the old code silently dropped.
    return np.clip(np.digitize(y_prob, np.linspace(0.0, 1.0, n_bins + 1)) - 1, 0, n_bins - 1)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    ids = _bin_ids(y_prob, n_bins)
    ece = 0.0
    for b in range(n_bins):
        mask = ids == b
        if not np.any(mask):
            continue
        ece += (np.sum(mask) / len(y_true)) * abs(np.mean(y_true[mask]) - np.mean(y_prob[mask]))
    return float(ece)


def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, list]:
    """Per-bin accuracy/confidence/count. Empty bins are reported with count 0, not invented."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    ids = _bin_ids(y_prob, n_bins)
    acc, conf, cnt = [], [], []
    for b in range(n_bins):
        mask = ids == b
        n = int(np.sum(mask))
        cnt.append(n)
        acc.append(float(np.mean(y_true[mask])) if n else None)
        conf.append(float(np.mean(y_prob[mask])) if n else None)
    return {"accuracy": acc, "confidence": conf, "count": cnt}


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_prob, dtype=float) - np.asarray(y_true, dtype=float)) ** 2))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> MetricBundle:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    two_class = len(np.unique(y_true)) > 1
    auroc = roc_auc_score(y_true, y_prob) if two_class else float("nan")
    auprc = average_precision_score(y_true, y_prob) if two_class else float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return MetricBundle(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        auroc=float(auroc),
        auprc=float(auprc),
        brier=brier_score(y_true, y_prob),
        ece=expected_calibration_error(y_true, y_prob, n_bins=10),
        confusion=(int(tn), int(fp), int(fn), int(tp)),
        n=int(len(y_true)),
        n_pos=int(np.sum(y_true == 1)),
    )


def recall_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, fpr_targets=(0.001, 0.01)) -> Dict[str, float]:
    """Attack recall at fixed false-positive rates (threshold chosen on the same scores)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    neg = np.sort(y_prob[y_true == 0])
    pos = y_prob[y_true == 1]
    out = {}
    for f in fpr_targets:
        if len(neg) == 0 or len(pos) == 0:
            out[f"recall_at_fpr_{f}"] = float("nan")
            continue
        k = int(np.floor(f * len(neg)))
        thr = neg[len(neg) - k - 1] if k < len(neg) else neg[0]
        out[f"recall_at_fpr_{f}"] = float(np.mean(pos > thr))
    return out


def bootstrap_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 500,
    seed: int = 0,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Percentile bootstrap CIs (row resampling, stratified by class)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return {}
    keys = ["precision", "recall", "f1", "auprc", "ece", "brier"]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos_idx, len(pos_idx)), rng.choice(neg_idx, len(neg_idx))])
        m = compute_metrics(y_true[idx], y_prob[idx], threshold)
        for k in keys:
            samples[k].append(getattr(m, k))
    return {
        k: {
            "lo": float(np.nanpercentile(v, 2.5)),
            "hi": float(np.nanpercentile(v, 97.5)),
            "mean": float(np.nanmean(v)),
        }
        for k, v in samples.items()
    }


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    return {
        "benign": {"precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0]), "support": int(support[0])},
        "attack": {"precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1]), "support": int(support[1])},
    }
