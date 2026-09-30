from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class AttackEvent:
    time: float
    tactic: str
    technique: str
    pid: int


def load_attack_info(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "Time of Attack": "time",
            "Tactic Name": "tactic",
            "Technique Name": "technique",
            "PID": "pid",
        }
    )
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["pid"] = pd.to_numeric(df["pid"], errors="coerce").astype("Int64")
    df["tactic"] = df["tactic"].astype(str).str.lower()
    df["technique"] = df["technique"].astype(str)
    return df.dropna(subset=["time", "pid"])


def to_events(df: pd.DataFrame) -> Iterable[AttackEvent]:
    for _, row in df.iterrows():
        yield AttackEvent(
            time=float(row["time"]),
            tactic=str(row["tactic"]),
            technique=str(row["technique"]),
            pid=int(row["pid"]),
        )
