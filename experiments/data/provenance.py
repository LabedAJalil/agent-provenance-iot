from __future__ import annotations

from bisect import bisect_left
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .attack_info import load_attack_info


PROVENANCE_COLUMNS = [
    "id",
    "type",
    "from",
    "to",
    "uid",
    "egid",
    "exe",
    "gid",
    "euid",
    "name",
    "pid",
    "seen time",
    "source",
    "ppid",
    "command line",
    "start time",
    "event id",
    "time",
    "operation",
    "path",
    "subtype",
    "permissions",
    "epoch",
    "version",
    "flags",
    "local address",
    "remote port",
    "protocol",
    "remote address",
    "local port",
    "tgid",
    "fd",
    "mode",
    "label",
    "subLabel",
]


def load_provenance(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in ["pid", "ppid", "uid", "euid", "gid", "egid", "tgid", "fd"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["time", "seen time", "start time", "epoch"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["type", "operation", "protocol", "subtype", "source", "exe", "name"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Many rows (notably Process nodes) do not populate the "time" column.
    # For temporal splits and attack labeling we use a unified timestamp.
    time_candidates = [c for c in ["time", "seen time", "start time", "epoch"] if c in df.columns]
    if time_candidates:
        event_time = None
        for c in time_candidates:
            if event_time is None:
                event_time = df[c]
            else:
                event_time = event_time.combine_first(df[c])
        # Filter out clearly invalid timestamps (0/1/etc.). Dataset times are epoch seconds ~1.7e9.
        event_time = pd.to_numeric(event_time, errors="coerce")
        event_time = event_time.mask(event_time < 1_000_000_000, np.nan)
        df["event_time"] = event_time
    else:
        df["event_time"] = np.nan
    return df


def _build_attack_index(attacks: pd.DataFrame) -> Dict[int, List[Tuple[float, str, str]]]:
    index: Dict[int, List[Tuple[float, str, str]]] = {}
    for _, row in attacks.iterrows():
        pid = int(row["pid"])
        entry = (float(row["time"]), str(row["tactic"]), str(row["technique"]))
        index.setdefault(pid, []).append(entry)
    for pid, entries in index.items():
        entries.sort(key=lambda x: x[0])
        index[pid] = entries
    return index


def _nearest_attack(
    times: List[Tuple[float, str, str]], t: float, window: float
) -> Tuple[bool, str, str]:
    ts = [x[0] for x in times]
    idx = bisect_left(ts, t)
    candidates = []
    if idx < len(ts):
        candidates.append(times[idx])
    if idx > 0:
        candidates.append(times[idx - 1])
    for at, tactic, tech in candidates:
        if abs(at - t) <= window:
            return True, tactic, tech
    return False, "", ""


def label_with_attack_info(
    df: pd.DataFrame,
    attack_info_path: Path,
    window_seconds: float = 5.0,
    time_col: str = "event_time",
    pid_col: str = "pid",
) -> pd.DataFrame:
    attacks = load_attack_info(attack_info_path)
    index = _build_attack_index(attacks)

    labels = np.zeros(len(df), dtype=int)
    tactics = np.array([""] * len(df), dtype=object)
    techniques = np.array([""] * len(df), dtype=object)

    pid_vals = df[pid_col].fillna(-1).astype(int).to_numpy()
    time_vals = df[time_col].fillna(-1).to_numpy()

    for i, (pid, t) in enumerate(zip(pid_vals, time_vals)):
        if pid in index and t >= 0:
            is_attack, tactic, tech = _nearest_attack(index[pid], float(t), window_seconds)
            if is_attack:
                labels[i] = 1
                tactics[i] = tactic
                techniques[i] = tech

    out = df.copy()
    out["is_attack"] = labels
    out["attack_tactic"] = tactics
    out["attack_technique"] = techniques
    return out
