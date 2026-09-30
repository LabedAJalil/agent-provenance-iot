from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class RandomForestConfig:
    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 2
    class_weight: str | None = "balanced"
    random_state: int = 42
    oob_score: bool = True


def train_random_forest(X, y, cfg: RandomForestConfig) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        class_weight=cfg.class_weight,
        random_state=cfg.random_state,
        oob_score=cfg.oob_score,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def predict_proba(model: RandomForestClassifier, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def oob_proba(model: RandomForestClassifier, X) -> np.ndarray:
    """Out-of-bag attack probability for the training rows. Rows that were never
    out-of-bag (rare with 300 trees) fall back to the in-sample prediction.
    Used to fit calibrators without a separate validation slice."""
    oob = model.oob_decision_function_[:, 1]
    missing = np.isnan(oob)
    if missing.any():
        oob = oob.copy()
        oob[missing] = predict_proba(model, X[missing])
    return oob
