"""Cross-event memory: propagate the detector's OWN earlier verdicts through provenance.

Question: which attacks that a per-process detector misses become visible once the
detector remembers what it flagged before? Three label-free, causal mechanisms are
tested on top of the forest's verdicts (``p_rf_rethr >= 0.5`` in ``test_predictions.csv``):

* ``lineage``  : the process's parent or an ancestor (observed at or before t) was flagged
                 before t.
* ``artifact`` : the process names (command line) or touches (file events in its window)
                 a path that a flagged process touched or named before t, excluding paths
                 that exist in the benign Phase 1 baseline and top-level directories.
* ``remote``   : the process connects to a remote address a flagged process connected to
                 before t, excluding addresses seen in Phase 1.

Verdicts are replayed in decision-time order; with ``--chain`` a process flagged by
propagation becomes a seed for later processes (tag propagation), otherwise only the
forest's verdicts seed memory. Nothing here reads a label; labels are used only to score.

Usage: .venv/bin/python memory_propagation.py outputs/rolling5_exclude_staging [--chain]
Writes ``memory_propagation[_chain].json`` in the run directory.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from config import default_paths
from data.provenance import load_provenance
from emulator import Emulator, strip_labels
from metrics import compute_metrics

PATH_RE = re.compile(r"(/[^\s'\";|&<>()$]+)")
GENERIC_PREFIXES = ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/proc", "/sys", "/dev", "/run", "/var/lib", "/var/cache", "/snap")


def cmd_paths(cmd) -> Set[str]:
    if not isinstance(cmd, str):
        return set()
    return {p.rstrip("/") for p in PATH_RE.findall(cmd) if p.count("/") >= 2}


def interesting(path: str, baseline: Set[str]) -> bool:
    """A path worth remembering: absolute, at least two levels deep, not a system prefix,
    not present in the benign Phase 1 baseline."""
    if not path or path.count("/") < 2 or path in baseline:
        return False
    return not path.startswith(GENERIC_PREFIXES)


def matches(paths: Set[str], memory: Set[str]) -> List[str]:
    hits = []
    for p in paths:
        if p in memory:
            hits.append(p)
            continue
        for m in memory:
            if p.startswith(m + "/") or m.startswith(p + "/"):
                hits.append(p)
                break
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str)
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--chain", action="store_true", help="propagated flags become seeds for later processes")
    ap.add_argument("--seed-col", type=str, default="p_rf_rethr", help="column whose verdicts seed memory")
    ap.add_argument("--seed-threshold", type=float, default=0.5, help="seed = seed_col >= threshold (memory writes); the rf variant is always scored at 0.5")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    root = Path(__file__).resolve().parents[1]
    paths = default_paths(root)

    pred = pd.read_csv(run_dir / "test_predictions.csv").sort_values("decision_time").reset_index(drop=True)
    ph1 = load_provenance(paths.provenance_phase1)
    ph2 = load_provenance(paths.provenance_phase2)
    emu = Emulator(provenance=strip_labels(ph2))
    baseline_paths = {str(p).rstrip("/") for p in ph1["path"].dropna().astype(str)} | {p for c in ph1["command line"].dropna().astype(str) for p in cmd_paths(c)}
    baseline_remote = set(ph1["remote address"].dropna().astype(str))

    seed = pred[args.seed_col].to_numpy() >= args.seed_threshold  # what gets written to memory
    rf05 = pred[args.seed_col].to_numpy() >= 0.5  # what the forest alone reports
    y = pred["is_attack"].to_numpy().astype(int)
    tactics = pred["attack_tactic"].fillna("").astype(str).to_numpy()

    flagged_pids: Dict[int, float] = {}  # pid -> decision time at which it was flagged
    mem_paths: Set[str] = set()
    mem_remote: Set[str] = set()
    lineage = np.zeros(len(pred), bool)
    artifact = np.zeros(len(pred), bool)
    remote = np.zeros(len(pred), bool)
    why: List[str] = [""] * len(pred)

    for i, r in pred.iterrows():
        pid, t = int(r["pid"]), float(r["decision_time"])
        # --- read memory (everything in memory was flagged strictly before t)
        tree = emu.get_process_tree(pid, t, depth=6)
        anc = [int(a["pid"]) for a in tree[1:] if a.get("status") == "OK"]
        if int(r["ppid"]) > 0:
            anc = [int(r["ppid"])] + [a for a in anc if a != int(r["ppid"])]
        hit_anc = [a for a in anc if a in flagged_pids and flagged_pids[a] < t]
        own_paths = cmd_paths(r["command line"]) | {str(f["path"]).rstrip("/") for f in emu.get_file_events(pid, t, window=args.window) if f.get("path")}
        own_paths = {p for p in own_paths if interesting(p, baseline_paths)}
        hit_paths = matches(own_paths, mem_paths)
        own_remote = {str(s["remote_address"]) for s in emu.get_socket_events(pid, t, window=args.window) if s.get("remote_address")} - baseline_remote
        hit_remote = sorted(own_remote & mem_remote)
        lineage[i], artifact[i], remote[i] = bool(hit_anc), bool(hit_paths), bool(hit_remote)
        why[i] = ";".join(filter(None, [f"ancestor_flagged={hit_anc[:2]}" if hit_anc else "", f"path={hit_paths[:2]}" if hit_paths else "", f"remote={hit_remote[:2]}" if hit_remote else ""]))
        # --- write memory: at decision time t the verdict for this process is known
        flag_now = bool(seed[i]) or (args.chain and (lineage[i] or artifact[i] or remote[i]))
        if flag_now:
            flagged_pids.setdefault(pid, t)
            touched = cmd_paths(r["command line"]) | {str(f["path"]).rstrip("/") for f in emu.get_file_events(pid, t, window=None) if f.get("path")}
            mem_paths |= {p for p in touched if interesting(p, baseline_paths)}
            mem_remote |= {str(s["remote_address"]) for s in emu.get_socket_events(pid, t, window=None) if s.get("remote_address")} - baseline_remote
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(pred)} replayed, memory: {len(flagged_pids)} pids, {len(mem_paths)} paths, {len(mem_remote)} remotes", flush=True)

    def block(name: str, pos: np.ndarray) -> Dict:
        m = compute_metrics(y, pos.astype(float))
        tn, fp, fn, tp = m.confusion
        per_t = {}
        for tac in sorted(set(tactics[y == 1])):
            mm = (tactics == tac) & (y == 1)
            per_t[tac] = {"support": int(mm.sum()), "recall": float(pos[mm].mean())}
        return {"name": name, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": m.precision, "recall": m.recall, "f1": m.f1,
                "fp_per_1000_benign": 1000.0 * fp / max(1, fp + tn), "per_tactic": per_t}

    seed = rf05  # score unions against the forest's ordinary verdicts
    variants = {
        "rf": seed,
        "lineage_only": lineage, "artifact_only": artifact, "remote_only": remote,
        "rf|lineage": seed | lineage, "rf|artifact": seed | artifact, "rf|remote": seed | remote,
        "rf|lineage|artifact": seed | lineage | artifact, "rf|lineage|artifact|remote": seed | lineage | artifact | remote,
    }
    out = {"run": run_dir.name, "chain": args.chain, "seed_col": args.seed_col, "seed_threshold": args.seed_threshold, "n": int(len(pred)), "n_attack": int(y.sum()),
           "memory_size": {"pids": len(flagged_pids), "paths": len(mem_paths), "remotes": len(mem_remote)},
           "baseline_size": {"paths": len(baseline_paths), "remotes": len(baseline_remote)},
           "variants": {k: block(k, v) for k, v in variants.items()}}
    # What propagation adds and what it costs, itemised.
    added = (~seed) & (lineage | artifact | remote)
    out["added_by_propagation"] = [{"pid": int(pred.pid[i]), "name": str(pred.name[i]), "cmd": str(pred["command line"][i])[:100], "is_attack": int(y[i]),
                                    "tactic": tactics[i], "why": why[i]} for i in np.flatnonzero(added)]
    suffix = ("_chain" if args.chain else "") + (f"_{args.tag}" if args.tag else "")
    (run_dir / f"memory_propagation{suffix}.json").write_text(json.dumps(out, indent=1))
    pred.assign(mem_lineage=lineage, mem_artifact=artifact, mem_remote=remote, mem_why=why)[["row_id", "pid", "name", "is_attack", "attack_tactic", args.seed_col, "mem_lineage", "mem_artifact", "mem_remote", "mem_why"]].to_csv(run_dir / f"memory_propagation{suffix}.csv", index=False)

    print(f"[{run_dir.name}] chain={args.chain} n={len(pred)} attacks={int(y.sum())} memory={out['memory_size']}")
    print("variant | TP | FP | recall | precision | F1 | FP/1000 | defenceEvasion R | collection R")
    for k, b in out["variants"].items():
        de = b["per_tactic"].get("defenceEvasion", {}).get("recall", float("nan"))
        co = b["per_tactic"].get("collection", {}).get("recall", float("nan"))
        print(f"{k} | {b['tp']} | {b['fp']} | {b['recall']:.3f} | {b['precision']:.3f} | {b['f1']:.3f} | {b['fp_per_1000_benign']:.1f} | {de:.2f} | {co:.2f}")
    print(f"\nadded by propagation: {len(out['added_by_propagation'])} (attacks: {sum(a['is_attack'] for a in out['added_by_propagation'])})")
    for a in out["added_by_propagation"][:40]:
        print(f"  {'ATTACK' if a['is_attack'] else 'benign'} {a['tactic']:16s} pid={a['pid']} {a['name']:12s} {a['cmd'][:60]!r}  <- {a['why'][:90]}")


if __name__ == "__main__":
    main()
