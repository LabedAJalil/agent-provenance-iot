"""Verdict stability of the language-model conditions under repeated sampling.

The published runs sample the model once per record (temperature 0.1, no seed). This script re-scores the
enriched sample of a run k more times WITHOUT the response cache and reports, per condition:

* the share of records whose verdict differs from the published one in at least one repetition (flip rate),
  separately for attacks and benign records, and the share of individual repeated verdicts that differ;
* TP / FP / F1 of every repetition, and their range;
* the exact McNemar p and the F1 difference against the reference forest in every repetition.

Repeated responses are stored in ``outputs/repeats/llm_repeats.sqlite`` keyed by (run, condition, row_id, rep), so
an interrupted run (daily request quota) resumes where it stopped. Costs API calls: about 1 request per record
and repetition for narrative/context and about 2.7 for the tool-using condition.

Usage: .venv/bin/python flip_rate.py outputs/rolling5_exclude_staging --conditions context,agent --reps 3
       .venv/bin/python flip_rate.py outputs/rolling5_exclude_staging --report-only
Writes ``<run>/flip_rate.json``.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from config import default_paths, load_runtime_config
from data.provenance import load_provenance
from emulator import Emulator, strip_labels
from llm import LLMConfig, SemanticScorer
from units import build_process_units, narrative_for_unit, narrative_only_for_unit, sweep_for_unit


class RepeatStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self._lock = str(path), threading.Lock()
        with sqlite3.connect(self.path) as con:
            con.execute("CREATE TABLE IF NOT EXISTS repeats (run TEXT, condition TEXT, row_id TEXT, rep INTEGER, result TEXT, created REAL, PRIMARY KEY (run, condition, row_id, rep))")

    def has(self, run, cond, row_id, rep) -> bool:
        with self._lock, sqlite3.connect(self.path) as con:
            return con.execute("SELECT 1 FROM repeats WHERE run=? AND condition=? AND row_id=? AND rep=?", (run, cond, row_id, rep)).fetchone() is not None

    def put(self, run, cond, row_id, rep, result_json: str) -> None:
        with self._lock, sqlite3.connect(self.path) as con:
            con.execute("INSERT OR REPLACE INTO repeats VALUES (?,?,?,?,?,?)", (run, cond, row_id, rep, result_json, time.time()))

    def frame(self, run) -> pd.DataFrame:
        with sqlite3.connect(self.path) as con:
            rows = con.execute("SELECT condition, row_id, rep, result FROM repeats WHERE run=?", (run,)).fetchall()
        out = []
        for cond, row_id, rep, res in rows:
            r = json.loads(res)
            out.append({"condition": cond, "row_id": row_id, "rep": rep, "score": r["score"], "verdict": r["verdict"], "snapshot": r.get("model_snapshot", ""),
                        "n_tool_calls": len(r.get("tool_calls", [])), "input_tokens": r.get("input_tokens", 0)})
        return pd.DataFrame(out)


def prf(y, d):
    tp, fp, fn = int(((d == 1) & (y == 1)).sum()), int(((d == 1) & (y == 0)).sum()), int(((d == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return tp, fp, (2 * p * r / (p + r) if p + r else 0.0)


def report(run_dir: Path, store: RepeatStore) -> None:
    rep_df = store.frame(run_dir.name)
    out = {"run": run_dir.name, "store": "outputs/repeats/llm_repeats.sqlite", "conditions": {}}
    for cond in ("narrative", "context", "sweep", "agent"):
        f = run_dir / f"llm_{cond}_predictions.csv"
        if not f.exists() or rep_df.empty or not (rep_df["condition"] == cond).any():
            continue
        base = pd.read_csv(f)
        base = base[base["fold"] > 0].set_index("row_id")
        y = base["y"].astype(int)
        d0 = (base["s_llm_raw"] >= 0.5).astype(int)
        rf = (base["p_rf_rethr"] >= 0.5).astype(int)
        r = rep_df[rep_df["condition"] == cond]
        wide = r.assign(d=(r["score"] >= 0.5).astype(int)).pivot(index="row_id", columns="rep", values="d").reindex(base.index)
        complete = wide.dropna()
        reps = sorted(int(c) for c in wide.columns)
        per_rep = []
        for k in reps:
            m = wide[k].notna()
            dk = wide.loc[m, k].astype(int)
            tp, fp, f1 = prf(y[m].to_numpy(), dk.to_numpy())
            ca, cb = (rf[m] == y[m]).to_numpy(), (dk == y[m]).to_numpy()
            n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
            per_rep.append({"rep": k, "n": int(m.sum()), "tp": tp, "fp": fp, "f1": f1, "differs_from_published": int((dk != d0[m]).sum()),
                            "forest_minus_llm_f1": prf(y[m].to_numpy(), rf[m].to_numpy())[2] - f1, "mcnemar_vs_forest": [n01, n10, float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0]})
        any_flip = (complete.ne(d0.loc[complete.index], axis=0)).any(axis=1) if len(complete) else pd.Series(dtype=bool)
        unstable = (complete.nunique(axis=1) > 1) if len(complete) else pd.Series(dtype=bool)
        yc = y.loc[complete.index]
        tp0, fp0, f10 = prf(y.to_numpy(), d0.to_numpy())
        out["conditions"][cond] = {"n_records": int(len(base)), "n_records_with_all_reps": int(len(complete)), "reps": reps,
                                   "published": {"tp": tp0, "fp": fp0, "f1": f10},
                                   "records_with_any_verdict_differing_from_published": int(any_flip.sum()),
                                   "record_flip_rate": float(any_flip.mean()) if len(any_flip) else None,
                                   "record_flip_rate_attacks": float(any_flip[yc == 1].mean()) if (yc == 1).any() else None,
                                   "record_flip_rate_benign": float(any_flip[yc == 0].mean()) if (yc == 0).any() else None,
                                   "records_unstable_across_reps": int(unstable.sum()),
                                   "verdict_level_disagreement_with_published": float(np.mean([p["differs_from_published"] / p["n"] for p in per_rep])) if per_rep else None,
                                   "f1_range": [min(p["f1"] for p in per_rep), max(p["f1"] for p in per_rep)] if per_rep else None,
                                   "fp_range": [min(p["fp"] for p in per_rep), max(p["fp"] for p in per_rep)] if per_rep else None,
                                   "tp_range": [min(p["tp"] for p in per_rep), max(p["tp"] for p in per_rep)] if per_rep else None,
                                   "snapshots": {str(k): int(v) for k, v in r["snapshot"].value_counts().items()},
                                   "distinct_scores": sorted(float(s) for s in r["score"].round(3).unique()), "per_rep": per_rep}
    (run_dir / "flip_rate.json").write_text(json.dumps(out, indent=1))
    for cond, c in out["conditions"].items():
        print(f"[{cond}] reps={c['reps']} complete={c['n_records_with_all_reps']}/{c['n_records']} published TP/FP/F1={c['published']['tp']}/{c['published']['fp']}/{c['published']['f1']:.3f} "
              f"| records flipped vs published: {c['records_with_any_verdict_differing_from_published']} ({(c['record_flip_rate'] or 0):.3f}; attacks {(c['record_flip_rate_attacks'] or 0):.3f}, benign {(c['record_flip_rate_benign'] or 0):.3f}) "
              f"| F1 range {c['f1_range']} TP {c['tp_range']} FP {c['fp_range']}")
        for p in c["per_rep"]:
            print(f"    rep {p['rep']}: n={p['n']} TP={p['tp']} FP={p['fp']} F1={p['f1']:.3f} differs={p['differs_from_published']} forest-LLM dF1={p['forest_minus_llm_f1']:+.3f} McNemar={p['mcnemar_vs_forest']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str)
    ap.add_argument("--conditions", type=str, default="narrative,context,agent")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    root = Path(__file__).resolve().parents[1]
    paths = default_paths(root)
    store = RepeatStore(paths.outputs / "repeats" / "llm_repeats.sqlite")
    if args.report_only:
        report(run_dir, store)
        return

    rt = load_runtime_config(root)
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    wanted = {}
    for cond in conds:
        d = pd.read_csv(run_dir / f"llm_{cond}_predictions.csv")
        wanted[cond] = set(d.loc[d["fold"] > 0, "row_id"])
    phase2 = load_provenance(paths.provenance_phase2)
    emu = Emulator(provenance=strip_labels(phase2))
    units = build_process_units(phase2, emu, window=args.window, mask=False)
    units["unit_key"] = [f"p2:{pid}:{st:.3f}" for pid, st in zip(units["pid"], units["start_time"])]
    units["row_id"] = [f"{k}:{pp}:{e}:{n}" for k, pp, e, n in zip(units["unit_key"], units["ppid"], units["exe"], units["name"])]
    units = units.set_index("row_id")
    cfg = LLMConfig(mode="openai", model=rt.openai_model, temperature=rt.openai_temperature, max_output_tokens=max(rt.llm_max_output_tokens, 300), api_key=rt.openai_api_key)
    scorer = SemanticScorer(cfg, cache=None)  # no cache: every call is a fresh sample

    def job(cond, row_id, rep):
        row = units.loc[row_id]
        narrative = narrative_only_for_unit(row) if cond == "narrative" else (sweep_for_unit(row, emu, args.window) if cond == "sweep" else narrative_for_unit(row, emu, args.window))
        res = scorer.score(cond, row["unit_key"], narrative, emulator=emu, pid=int(row["pid"]), decision_time=float(row["decision_time"]))
        if not res.error:
            store.put(run_dir.name, cond, row_id, rep, res.to_json())
        return res.error

    stop = False
    for cond in conds:
        for rep in range(1, args.reps + 1):
            todo = [r for r in sorted(wanted[cond]) if not store.has(run_dir.name, cond, r, rep)]
            if not todo or stop:
                continue
            t0, errs = time.time(), 0
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(job, cond, r, rep) for r in todo]
                for i, f in enumerate(as_completed(futs), 1):
                    e = f.result()
                    errs += bool(e)
                    if e and "quota" in e:
                        stop = True
                    if i % 100 == 0:
                        print(f"  [{cond} rep {rep}] {i}/{len(todo)} ({time.time() - t0:.0f}s, {errs} errors)", flush=True)
            print(f"[{cond} rep {rep}] done: {len(todo)} requested, {errs} errors, {time.time() - t0:.0f}s", flush=True)
    report(run_dir, store)


if __name__ == "__main__":
    main()
