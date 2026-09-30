from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    time_col: str = "time",
    train_ratio: float = 0.2,
    val_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if time_col not in df.columns:
        raise ValueError(f"time column '{time_col}' not found in dataframe")
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = df_sorted.iloc[:n_train]
    val = df_sorted.iloc[n_train : n_train + n_val]
    test = df_sorted.iloc[n_train + n_val :]
    return train, val, test


def rolling_folds(df: pd.DataFrame, time_col: str, n_folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Rolling-origin chronological folds.

    ``df`` is cut into ``n_folds + 1`` consecutive time chunks of equal size.
    Fold k (k = 1..n_folds) trains on chunks 0..k-1 and tests on chunk k, so every
    test example is scored by a model that only saw earlier data. Returns
    positional index arrays into ``df`` sorted by ``time_col``.
    """
    if time_col not in df.columns:
        raise ValueError(f"time column '{time_col}' not found in dataframe")
    order = np.argsort(df[time_col].to_numpy(), kind="stable")
    chunks = np.array_split(order, n_folds + 1)
    folds = []
    for k in range(1, n_folds + 1):
        train_idx = np.concatenate(chunks[:k])
        test_idx = chunks[k]
        folds.append((train_idx, test_idx))
    return folds


def rolling_folds_time(df: pd.DataFrame, time_col: str, n_folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Rolling-origin folds with chunks of equal DURATION rather than equal count.

    The observed span of ``time_col`` is cut into ``n_folds + 1`` equal intervals.
    Chunk boundaries therefore depend only on the clock, not on how many units a
    relabelling keeps, so every label view shares the same boundaries and no chunk
    can collapse to a handful of units because a dense burst was removed.
    """
    if time_col not in df.columns:
        raise ValueError(f"time column '{time_col}' not found in dataframe")
    t = df[time_col].to_numpy(dtype=float)
    edges = np.linspace(t.min(), t.max(), n_folds + 2)
    chunk = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, n_folds)
    folds = []
    for k in range(1, n_folds + 1):
        folds.append((np.flatnonzero(chunk < k), np.flatnonzero(chunk == k)))
    return folds
