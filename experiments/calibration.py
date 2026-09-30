from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass(frozen=True)
class IsotonicCalibrator:
    model: IsotonicRegression

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return self.model.transform(scores)


def fit_isotonic(y_true: np.ndarray, y_prob: np.ndarray) -> IsotonicCalibrator:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_prob, y_true)
    return IsotonicCalibrator(model=iso)


def _nll_from_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
    probs = 1.0 / (1.0 + np.exp(-logits))
    eps = 1e-9
    probs = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs)))


def temperature_scale(logits: np.ndarray, y_true: np.ndarray) -> Tuple[float, np.ndarray]:
    temps = np.linspace(0.5, 5.0, 91)
    best_t = 1.0
    best_nll = float("inf")
    for t in temps:
        nll = _nll_from_logits(logits / t, y_true)
        if nll < best_nll:
            best_nll = nll
            best_t = t
    calibrated = 1.0 / (1.0 + np.exp(-(logits / best_t)))
    return best_t, calibrated
