from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from metrics import expected_calibration_error


@dataclass(frozen=True)
class FusionResult:
    alpha: float
    beta: float
    fused_scores: np.ndarray


def fuse_logit(
    p_stat: np.ndarray, p_sem: np.ndarray, alpha: float, beta: float
) -> np.ndarray:
    eps = 1e-9
    p_stat = np.clip(p_stat, eps, 1 - eps)
    p_sem = np.clip(p_sem, eps, 1 - eps)
    logit_stat = np.log(p_stat / (1 - p_stat))
    logit_sem = np.log(p_sem / (1 - p_sem))
    fused_logit = alpha * logit_stat + beta * logit_sem
    return 1.0 / (1.0 + np.exp(-fused_logit))


def grid_search_alpha_beta(
    y_true: np.ndarray,
    p_stat: np.ndarray,
    p_sem: np.ndarray,
    grid: Tuple[float, ...] = tuple([i / 10 for i in range(1, 10)]),
) -> FusionResult:
    best_alpha = 0.5
    best_beta = 0.5
    best_ece = float("inf")
    best_scores = None
    for alpha in grid:
        for beta in grid:
            fused = fuse_logit(p_stat, p_sem, alpha, beta)
            ece = expected_calibration_error(y_true, fused)
            if ece < best_ece:
                best_ece = ece
                best_alpha = alpha
                best_beta = beta
                best_scores = fused
    return FusionResult(alpha=best_alpha, beta=best_beta, fused_scores=best_scores)
