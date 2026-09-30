"""Post hoc evidence tables read from existing run directories (no model call, no retraining).

Writes ``outputs/posthoc/<name>.json`` for every analysis the manuscript quotes that is not already a
field of a run's ``metrics.json``:

  signature_rule      training-free command-line rules scored on every label view
  llm_false_positives false positives of each language-model condition by process name
  view0_logistic_fps  composition of the forest's false positives under shipped labels + logistic rule
  staging_recall      recall of each language-model condition on the staging copies (view C)
  tool_calls          tool-call histogram and number of distinct lookups per record
  score_values        distinct elicited risk-score values per condition
  cache_snapshots     cached responses by creation day and recorded model snapshot
  agent_coverage      share of each test chunk scored by the tool-using condition in the all-benign runs
  attack_info_check   audit rules cross-checked against the Caldera operation report shipped with the dataset
  label_closure       test of how the shipped labels were assigned (report pids + recorded descendants)
  suspects_*.csv      the staging and helper suspect lists (row ids), for release

Usage: .venv/bin/python posthoc_tables.py
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "outputs"
POST = OUT / "posthoc"
SIG_RE = r"base64|eval"
STAGING_RE = r"/tmp/\.s(/|\b)|/home/erfan/staged|(^|\s)staged/"
VIEWS = {"view0_shipped": "rolling5", "viewA_staging_excluded": "rolling5_exclude_staging",
         "viewB_staging_helpers_excluded": "rolling5_exclude_staging_helpers", "viewC_staging_positive": "rolling5_positive_staging",
         "viewA_time_folds": "rolling5_time_exclude_staging"}
CONDS = ("narrative", "context", "agent")


def conf(y: np.ndarray, d: np.ndarray) -> dict:
    tp, fp = int(((d == 1) & (y == 1)).sum()), int(((d == 1) & (y == 0)).sum())
    fn, tn = int(((d == 0) & (y == 1)).sum()), int(((d == 0) & (y == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"n": int(len(y)), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0, "fp_per_1000_benign": 1000.0 * fp / max(1, fp + tn)}


def save(name: str, obj) -> None:
    POST.mkdir(parents=True, exist_ok=True)
    (POST / f"{name}.json").write_text(json.dumps(obj, indent=1, default=str))
    print(f"wrote posthoc/{name}.json")


def signature_rule() -> None:
    out = {"rules": {"sig_own_or_ancestor": f"own or ancestor command line matches /{SIG_RE}/",
                     "sig_or_staging_path": f"sig_own_or_ancestor OR own/ancestor command line matches /{STAGING_RE}/ (path list taken from the label audit: post hoc)"},
           "note": "No training. Command lines and ancestor command lines are the causal ones stored in test_predictions.csv.", "views": {}}
    for view, run in VIEWS.items():
        f = OUT / run / "test_predictions.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        y = df["is_attack"].to_numpy().astype(int)
        own, anc = df["command line"].fillna("").astype(str), df["ancestor_cmdline"].fillna("").astype(str)
        sig = (own.str.contains(SIG_RE, regex=True) | anc.str.contains(SIG_RE, regex=True)).to_numpy()
        stg = (own.str.contains(STAGING_RE, regex=True) | anc.str.contains(STAGING_RE, regex=True)).to_numpy()
        out["views"][view] = {"run": run, "sig_own_or_ancestor": conf(y, sig.astype(int)), "sig_or_staging_path": conf(y, (sig | stg).astype(int)),
                              "attacks_with_signature": int((sig & (y == 1)).sum()), "attacks_total": int(y.sum())}
    save("signature_rule", out)


def llm_false_positives() -> None:
    out = {}
    for run in ("rolling5_exclude_staging", "rolling5_allbenign_exclude_staging", "rolling5_allbenign_exclude_staging_helpers", "rolling5_masked_exclude_staging"):
        out[run] = {}
        for c in CONDS:
            f = OUT / run / f"llm_{c}_predictions.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            d = d[d["fold"] > 0] if "fold" in d.columns else d
            fp = d[(d["y"] == 0) & (d["s_llm_raw"] >= 0.5)]
            out[run][c] = {"n_scored": int(len(d)), "n_benign": int((d["y"] == 0).sum()), "fp": int(len(fp)), "fp_by_name": dict(Counter(fp["name"].astype(str)).most_common()),
                           "examples": fp.groupby("name")["command line"].first().astype(str).str[:120].to_dict()}
    save("llm_false_positives", out)


def view0_logistic_fps() -> None:
    df = pd.read_csv(OUT / "rolling5" / "test_predictions.csv")
    fp = df[(df["is_attack"] == 0) & (df["p_rf_rethr"] >= 0.5)]
    stg = fp["command line"].fillna("").astype(str).str.contains(STAGING_RE, regex=True)
    hlp = (fp["name"].astype(str) == "base64")
    save("view0_logistic_fps", {"run": "rolling5", "rule": "rf_rethresholded_logistic", "fp": int(len(fp)), "fp_staging_suspects": int(stg.sum()),
                                "fp_named_base64": int((hlp & ~stg).sum()), "fp_other": int((~stg & ~hlp).sum()),
                                "fp_other_by_name": dict(Counter(fp[~stg & ~hlp]["name"].astype(str)).most_common(10)),
                                "benign_staging_suspects_in_pool": int(((df["is_attack"] == 0) & df["command line"].fillna("").astype(str).str.contains(STAGING_RE, regex=True)).sum())})


def staging_recall() -> None:
    run, out = "rolling5_positive_staging", {}
    for c in CONDS:
        d = pd.read_csv(OUT / run / f"llm_{c}_predictions.csv")
        d = d[d["fold"] > 0]
        m = d["attack_tactic"].fillna("").astype(str).str.contains("staging-relabelled")
        out[c] = {"n_staging_copies_scored": int(m.sum()), "flagged": int((d.loc[m, "s_llm_raw"] >= 0.5).sum()), "recall": float((d.loc[m, "s_llm_raw"] >= 0.5).mean()),
                  "forest_logistic_flagged": int((d.loc[m, "p_rf_rethr"] >= 0.5).sum()) if "p_rf_rethr" in d.columns else None}
    pooled = pd.read_csv(OUT / run / "test_predictions.csv")
    m = pooled["attack_tactic"].fillna("").astype(str).str.contains("staging-relabelled")
    out["forest_pooled"] = {"n_staging_copies": int(m.sum()), "flagged_logistic": int((pooled.loc[m, "p_rf_rethr"] >= 0.5).sum()), "flagged_0.5": int((pooled.loc[m, "p_rf_raw"] >= 0.5).sum()),
                            "by_chunk_recall_logistic": {int(k): float(v) for k, v in (pooled[m].assign(h=pooled.loc[m, "p_rf_rethr"] >= 0.5).groupby("chunk")["h"].mean()).items()}}
    save("staging_recall", out)


def tool_calls() -> None:
    out = {}
    for run in ("rolling5_exclude_staging", "rolling5_allbenign_exclude_staging"):
        d = pd.read_csv(OUT / run / "llm_agent_predictions.csv")
        d = d[d["fold"] > 0]
        n_distinct = d["tools"].fillna("").astype(str).map(lambda s: len({t for t in s.split(",") if t}))
        out[run] = {"n": int(len(d)), "n_tool_calls_hist": {int(k): int(v) for k, v in sorted(Counter(d["n_tool_calls"]).items())},
                    "distinct_tools_hist": {int(k): int(v) for k, v in sorted(Counter(n_distinct).items())},
                    "mean_input_tokens": float(d["input_tokens"].mean()), "mean_output_tokens": float(d["output_tokens"].mean()), "mean_api_calls": float(d["n_api_calls"].mean())}
    save("tool_calls", out)


def score_values() -> None:
    out = {}
    for c in CONDS:
        d = pd.read_csv(OUT / "rolling5_exclude_staging" / f"llm_{c}_predictions.csv")
        d = d[d["fold"] > 0]
        out[c] = {str(k): int(v) for k, v in sorted(Counter(d["s_llm_raw"].round(3)).items())}
        out[c + "_verdict_agrees_with_score_ge_0.5"] = int(((d["verdict"] == "MALICIOUS") == (d["s_llm_raw"] >= 0.5)).sum())
        out[c + "_n"] = int(len(d))
    save("score_values", out)


def cache_snapshots() -> None:
    con = sqlite3.connect(OUT / "llm_cache_v3.sqlite")
    rows, tok_in, tok_out = Counter(), 0, 0
    for res, created in con.execute("select result, created from llm_results"):
        r = json.loads(res)
        day = datetime.fromtimestamp(created, tz=timezone.utc).strftime("%Y-%m-%d")
        rows[(day, r.get("model_snapshot") or "(not recorded)")] += 1
        tok_in += int(r.get("input_tokens", 0)); tok_out += int(r.get("output_tokens", 0))
    save("cache_snapshots", {"by_day_and_snapshot": [{"day": d, "snapshot": s, "n": n} for (d, s), n in sorted(rows.items())], "total": int(sum(rows.values())),
                             "input_tokens_total": tok_in, "output_tokens_total": tok_out,
                             "usd_at_list_price_0.15_in_0.60_out_per_million": round(tok_in * 0.15e-6 + tok_out * 0.60e-6, 2)})


def agent_coverage() -> None:
    out = {}
    for run in ("rolling5_allbenign_exclude_staging", "rolling5_allbenign_exclude_staging_helpers"):
        pooled = pd.read_csv(OUT / run / "test_predictions.csv")
        ag = set(pd.read_csv(OUT / run / "llm_agent_predictions.csv")["row_id"])
        pooled["scored"] = pooled["row_id"].isin(ag)
        g = pooled.groupby(["chunk", "is_attack"])["scored"].agg(["sum", "count"])
        out[run] = {f"chunk{int(c)}_{'attack' if a else 'benign'}": {"scored": int(r["sum"]), "total": int(r["count"])} for (c, a), r in g.iterrows()}
        out[run]["all"] = {"scored": int(pooled["scored"].sum()), "total": int(len(pooled)), "benign_scored": int((pooled["scored"] & (pooled.is_attack == 0)).sum()), "benign_total": int((pooled.is_attack == 0).sum())}
    save("agent_coverage", out)


def attack_info_check() -> None:
    """Independent check of the audit rules against the Caldera report shipped with the dataset."""
    info = pd.read_csv(ROOT / "Dataset" / "Supplementary Material" / "attack_info.csv")
    info.columns = ["time", "tactic", "technique", "pid", "readable"]
    prov = pd.read_csv(ROOT / "Dataset" / "Provenance_Logs" / "Phase2_Provenance.csv", usecols=["type", "pid", "ppid", "name", "command line", "start time", "seen time", "label", "subLabel"], low_memory=False)
    proc = prov[prov["type"] == "Process"].copy()
    proc["t"] = proc["start time"].fillna(proc["seen time"])
    proc["label"] = pd.to_numeric(proc["label"], errors="coerce").fillna(0).astype(int)
    attack_pids = set(proc.loc[proc["label"] == 1, "pid"].dropna().astype(int))
    all_pids = set(proc["pid"].dropna().astype(int))
    steps = []
    for _, s in info.iterrows():
        lab = proc[(proc["label"] == 1) & (proc["pid"] >= s.pid) & (proc["pid"] <= s.pid + 40) & ((proc["t"] - s.time).abs() <= 600)]
        steps.append({"technique": s.technique, "tactic": s.tactic, "pid": int(s.pid), "pid_is_recorded_process": bool(int(s.pid) in all_pids), "pid_is_labelled_attack": bool(int(s.pid) in attack_pids),
                      "labelled_attack_pids_within_40_pids_and_10_min": int(lab["pid"].nunique())})
    # staging suspects against the two 'advanced file search and stager' steps
    stagers = info[info["technique"].str.contains("stager") & (info["tactic"] != "cleanup")]
    sus = proc[(proc["label"] == 0) & proc["command line"].fillna("").astype(str).str.contains(STAGING_RE, regex=True)].groupby("pid").agg(t=("t", "min"), ppid=("ppid", "first")).reset_index()
    per_step = []
    nxt = sorted(info.loc[info["tactic"] != "cleanup", "time"].tolist())
    for _, s in stagers.iterrows():
        t_next = min([t for t in nxt if t > s.time], default=s.time + 3600)
        m = (sus["t"] >= s.time) & (sus["t"] < t_next)
        per_step.append({"step_pid": int(s.pid), "step_time": float(s.time), "next_step_time": float(t_next), "suspects_started_between_step_and_next_step": int(m.sum()),
                         "suspect_parent_pids": {int(k): int(v) for k, v in Counter(sus.loc[m, "ppid"].dropna().astype(int)).most_common(4)},
                         "parent_pid_minus_step_pid": {int(k) - int(s.pid): int(v) for k, v in Counter(sus.loc[m, "ppid"].dropna().astype(int)).most_common(4)},
                         "minutes_from_step_to_last_suspect": float((sus.loc[m, "t"].max() - s.time) / 60.0) if m.any() else None})
    in_any = sum(p["suspects_started_between_step_and_next_step"] for p in per_step)
    save("attack_info_check", {"source": "Dataset/Supplementary Material/attack_info.csv (extracted from MITRE Caldera reports by the dataset authors)",
                               "n_steps": int(len(info)), "n_steps_cleanup": int((info.tactic == "cleanup").sum()),
                               "steps_whose_pid_is_a_recorded_process": int(sum(s["pid_is_recorded_process"] for s in steps)),
                               "steps_whose_pid_is_a_labelled_attack": int(sum(s["pid_is_labelled_attack"] for s in steps)),
                               "steps_with_a_labelled_attack_within_40_pids_and_10_min": int(sum(s["labelled_attack_pids_within_40_pids_and_10_min"] > 0 for s in steps)),
                               "staging_suspect_pids_in_raw_csv": int(len(sus)), "staging_suspects_inside_a_stager_step_interval": int(in_any), "stager_steps": per_step, "steps": steps})


def label_closure() -> None:
    """How were the shipped labels assigned? Test: step pids of the Caldera report + their RECORDED descendants."""
    info = pd.read_csv(ROOT / "Dataset" / "Supplementary Material" / "attack_info.csv")
    steps = set(info["PID"].astype(int))
    prov = pd.read_csv(ROOT / "Dataset" / "Provenance_Logs" / "Phase2_Provenance.csv", usecols=["type", "pid", "ppid", "name", "label"], low_memory=False)
    proc = prov[prov["type"] == "Process"].copy()
    proc["label"] = pd.to_numeric(proc["label"], errors="coerce").fillna(0).astype(int)
    g = proc.groupby("pid").agg(label=("label", "max"), ppid=("ppid", "first"), name=("name", "first")).reset_index()
    g["pid"], g["ppid"] = g["pid"].astype(int), g["ppid"].fillna(-1).astype(int)
    children: dict = {}
    for p, pp in zip(g["pid"], g["ppid"]):
        children.setdefault(pp, []).append(p)
    desc, stack = set(), list(steps)
    while stack:
        for c in children.get(stack.pop(), []):
            if c not in desc:
                desc.add(c); stack.append(c)
    desc -= steps
    att = set(g.loc[g["label"] == 1, "pid"])
    d = g[g["pid"].isin(desc)]
    save("label_closure", {"caldera_step_pids": len(steps), "step_pids_labelled_attack": int(len(steps & att)), "step_pid_parents": {int(k): int(v) for k, v in Counter(g[g.pid.isin(steps)]["ppid"]).items()},
                           "labelled_attack_pids": int(len(att)), "labelled_attack_pids_not_in_report": int(len(att - steps)),
                           "recorded_descendants_of_step_pids": int(len(d)), "recorded_descendants_labelled_attack": int(d["label"].sum()), "recorded_descendants_labelled_benign": int((d["label"] == 0).sum()),
                           "labelled_attack_pids_neither_step_nor_descendant": sorted(int(p) for p in att - steps - desc),
                           "reading": "shipped label == Caldera step pid OR recorded descendant of one; a process whose parent chain is not recorded cannot inherit the label"})


def suspects() -> None:
    f = OUT / "suspects"
    if not f.exists():
        print("outputs/suspects/ not present: run run_rolling.py once to export the suspect lists")


if __name__ == "__main__":
    for fn in (signature_rule, llm_false_positives, view0_logistic_fps, staging_recall, tool_calls, score_values, cache_snapshots, agent_coverage, attack_info_check, label_closure, suspects):
        fn()
