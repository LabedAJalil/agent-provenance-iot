from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import List

import pandas as pd


def _pcap2csv_dir(root: Path) -> Path:
    return root / "Dataset" / "Supplementary Material" / "pcap2csv"


def pcap_to_csv(pcap_path: Path, output_csv: Path, root: Path) -> None:
    pcap2csv_path = _pcap2csv_dir(root)
    if not pcap2csv_path.exists():
        raise FileNotFoundError(f"pcap2csv scripts not found at {pcap2csv_path}")
    sys.path.insert(0, str(pcap2csv_path))
    try:
        fe_module = importlib.import_module("Feature_extraction")
        fe = fe_module.Feature_extraction()
        fe.pcap_evaluation(str(pcap_path), str(output_csv.with_suffix("")))
    finally:
        if str(pcap2csv_path) in sys.path:
            sys.path.remove(str(pcap2csv_path))


def ensure_flow_csvs(pcap_dir: Path, out_dir: Path, root: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_paths: List[Path] = []
    for pcap in sorted(pcap_dir.glob("*.pcap")):
        csv_path = out_dir / f"{pcap.stem}.csv"
        if not csv_path.exists():
            pcap_to_csv(pcap, csv_path, root)
        csv_paths.append(csv_path)
    return csv_paths


def load_flow_csvs(csv_paths: List[Path], pcap_time_offset: float = 0.0) -> pd.DataFrame:
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path, low_memory=False)
        df["pcap_file"] = path.stem
        if "ts" in df.columns:
            df["ts"] = pd.to_numeric(df["ts"], errors="coerce") + pcap_time_offset
        frames.append(df)
    flows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    flows["flow_id"] = flows.index.astype(str) if len(flows) else []
    return flows
