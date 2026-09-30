"""Rolling-origin evaluation of the process-unit detector, with optional LLM conditions.

Protocol
--------
* Phase 2 process units are cut into ``n_folds + 1`` chronological chunks. Fold k
  trains a Random Forest on Phase 1 + chunks 0..k-1 and tests on chunk k.
* Isotonic calibration is fit on the training set's out-of-bag scores (no
  separate validation slice, so it always has attack support).
* LLM conditions score a fixed evaluation sample: ALL Phase 2 attack units plus a
  random benign sample. Fusion weights, temperature and the stacking model are fit
  per fold on scored units in that fold's training chunks and applied to scored
  units in the test chunk. LLM-only metrics are fold-independent.
* Everything is pooled across test chunks; chunk 0 is never tested.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from calibration import fit_isotonic, temperature_scale
from config import default_paths, load_runtime_config
from data.provenance import load_provenance
from emulator import Emulator, strip_labels
from features import build_feature_pipeline, process_unit_features
from fusion import fuse_logit, grid_search_alpha_beta
from llm import LLMCache, LLMConfig, LLMResult, SemanticScorer
from metrics import bootstrap_metrics, compute_metrics, recall_at_fpr, reliability_bins
from models.baselines import RandomForestConfig, oob_proba, predict_proba, train_random_forest
from report import plot_calibration, plot_pr, save_json
from split import rolling_folds, rolling_folds_time
from investigator import CaseNotebook, HostBaseline, PathIndex, behaviour_for_unit, memory_section, unit_paths
from units import build_process_units, narrative_for_unit, narrative_only_for_unit, sweep_for_unit

# List prices (USD per 1M tokens) at time of writing; used only for a cost estimate.
PRICE_PER_M = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1": (2.00, 8.00)}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_rev(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return "unknown"


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def evaluate(name: str, y: np.ndarray, p: np.ndarray, n_boot: int, seed: int) -> Dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    m = compute_metrics(y, p)
    tn, fp, fn, tp = m.confusion
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return {"name": name, **m.__dict__, **recall_at_fpr(y, p), "fpr": float(fpr), "fp_per_1000_benign": float(1000.0 * fpr),
            "ci95": bootstrap_metrics(y, p, n_boot=n_boot, seed=seed) if n_boot > 0 and y.sum() > 0 else None,
            "reliability": reliability_bins(y, p)}


def tactic_table(tactics: np.ndarray, y: np.ndarray, pred: np.ndarray) -> Dict:
    out = {}
    for t in sorted(set(tactics)):
        if t in ("", "0", "nan"):
            continue
        m = (tactics == t) & (y == 1)
        if m.sum():
            out[t] = {"support": int(m.sum()), "recall": float(np.mean(pred[m] == 1))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Rolling-origin evaluation with optional LLM conditions.")
    ap.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--fold-mode", type=str, choices=["count", "time"], default="count",
                    help="count: chunks of equal unit count (cut on the full set); time: chunks of equal duration")
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--mask-signature", action="store_true", help="delete base64/eval/decode tokens from all command lines")
    ap.add_argument("--mask-level", type=str, choices=["tokens", "decode"], default="tokens",
                    help="tokens: delete base64/eval/decode words and blobs; decode: replace the whole Caldera launcher wrapper "
                         "by its decoded payload first, so no trace of the emulation idiom remains")
    ap.add_argument("--mask-test-only", action="store_true",
                    help="train the forest on unmasked command lines and test it on masked ones (no retraining on masked data); "
                         "the zero-shot LLM conditions see masked evidence as under --mask-signature")
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--no-cmdline", action="store_true")
    ap.add_argument("--no-ancestry", action="store_true")
    ap.add_argument("--llm", type=str, choices=["none", "openai"], default="none")
    ap.add_argument("--conditions", type=str, default="narrative,context,agent")
    ap.add_argument("--n-benign", type=int, default=1000, help="benign Phase 2 units to score with the LLM; -1 = every benign unit")
    ap.add_argument("--n-benign-phase1", type=int, default=300, help="benign Phase 1 units scored (fusion fitting only)")
    ap.add_argument("--smoke", type=int, default=0, help="score only N attack + N benign units per condition (API check)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tool-budget", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--cache-only", action="store_true", help="never call the API; uncached units are errors")
    ap.add_argument("--llm-backend", type=str, choices=["responses", "chat"], default="responses",
                    help="responses: OpenAI Responses API (gpt-4o-mini runs). chat: any OpenAI-compatible chat endpoint (open models)")
    ap.add_argument("--llm-model", type=str, default="", help="model id for the chat backend, e.g. 'Qwen/Qwen3-8B:nscale' (':provider' pins the HF provider)")
    ap.add_argument("--llm-base-url", type=str, default="https://router.huggingface.co/v1")
    ap.add_argument("--llm-key-env", type=str, default="HF_TOKEN", help="environment variable (or experiments/.env entry) holding the key for the chat backend")
    ap.add_argument("--llm-no-think", action="store_true", help="append Qwen3's /no_think switch (hybrid reasoning models)")
    ap.add_argument("--llm-seed", type=int, default=42)
    ap.add_argument("--investigator-max-tokens", type=int, default=900, help="output limit per step of the investigator condition (raise it for reasoning models)")
    ap.add_argument("--investigator-no-memory", action="store_true", help="ablation: the investigator runs without its case notebook")
    ap.add_argument("--relabel", type=str, choices=["none", "exclude_staging", "positive_staging", "exclude_staging_helpers", "positive_staging_helpers"], default="none",
                    help="label audit. staging: benign-labelled processes that touch the attacker's staging directories "
                         "(children of the file stager whose parent subshell is not in the provenance). helpers: benign-labelled "
                         "`base64 --decode` processes with an unrecorded parent, pid- and time-adjacent to a labelled launcher "
                         "(the launcher's command-substitution subshell). exclude_* drops them from the negatives; positive_* counts them as attacks")
    args = ap.parse_args()
    if args.mask_signature and args.mask_test_only:
        ap.error("--mask-signature and --mask-test-only are exclusive")
    # The LLM is zero-shot, so "masked at test time" and "masked everywhere" are the same evidence for it.
    llm_mask = args.mask_signature or args.mask_test_only
    if llm_mask and args.mask_level == "decode":
        import units as _units
        _units.MASK_LEVEL = "decode"

    t0 = time.time()
    np.random.seed(args.seed)
    root = Path(args.root)
    paths = default_paths(root)
    rt = load_runtime_config(root)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()] if args.llm != "none" else []

    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv, "args": vars(args),
                "git_rev": _git_rev(root), "python": platform.python_version(),
                "versions": {m: __import__(m).__version__ for m in ("numpy", "pandas", "sklearn")},
                "dataset_md5": {"Phase1_Provenance.csv": _md5(paths.provenance_phase1), "Phase2_Provenance.csv": _md5(paths.provenance_phase2)},
                "prompt_version": __import__("llm").PROMPT_VERSION,
                "llm_model": ((args.llm_model if args.llm_backend == "chat" else rt.openai_model) if args.llm != "none" else None),
                "llm_backend": args.llm_backend if args.llm != "none" else None, "llm_base_url": args.llm_base_url if args.llm_backend == "chat" else None}

    # ------------------------------------------------------------------ data
    phase1 = load_provenance(paths.provenance_phase1)
    phase2 = load_provenance(paths.provenance_phase2)
    phase1["label"] = 0
    phase1["subLabel"] = ""
    emu1 = Emulator(provenance=strip_labels(phase1))
    emu2 = Emulator(provenance=strip_labels(phase2))
    units1 = build_process_units(phase1, emu1, window=args.window, mask=args.mask_signature)
    units2 = build_process_units(phase2, emu2, window=args.window, mask=args.mask_signature)
    units1["phase"] = 1
    units2["phase"] = 2
    # Test-only masking: a second Phase 2 frame with the signature deleted, used only as the forest's test input.
    units2_masked = build_process_units(phase2, emu2, window=args.window, mask=True) if args.mask_test_only else None
    if units2_masked is not None:
        units2_masked["phase"] = 2
    for u in (units1, units2) + ((units2_masked,) if units2_masked is not None else ()):
        u["unit_key"] = [f"p{ph}:{pid}:{st:.3f}" for ph, pid, st in zip(u["phase"], u["pid"], u["start_time"])]
        # unit_key is not unique (a pid that execve's yields several units with one start time); uid is.
        u["row_id"] = [f"{k}:{pp}:{e}:{n}" for k, pp, e, n in zip(u["unit_key"], u["ppid"], u["exe"], u["name"])]
        assert u["row_id"].is_unique
    if units2_masked is not None:
        units2_masked = units2_masked.set_index("row_id")
        assert set(units2_masked.index) == set(units2["row_id"])
    units2 = units2.sort_values("start_time").reset_index(drop=True)
    STAGING_RE = r"/tmp/\.s(/|\b)|/home/erfan/staged|(^|\s)staged/"
    units2["staging_suspect"] = ((units2["is_attack"] == 0) & units2["command line"].fillna("").astype(str).str.contains(STAGING_RE, regex=True)).astype(int)
    units1["staging_suspect"] = 0
    manifest["staging_suspects"] = int(units2["staging_suspect"].sum())
    # Launcher helpers: `sh -c eval "$(echo <b64> | base64 --decode)"` forks an unrecorded subshell that runs
    # `base64 --decode`; the decoder is labelled benign. Rule: benign, name base64, parent not a recorded unit,
    # a labelled attack unit with pid in [ppid-4, ppid] that started within 2 s.
    _att = units2[units2["is_attack"] == 1][["pid", "start_time"]].to_numpy()
    _unit_pids = set(units2["pid"].tolist())
    def _is_helper(row) -> bool:
        if row["is_attack"] == 1 or row["name"] != "base64" or int(row["ppid"]) in _unit_pids:
            return False
        m = (_att[:, 0] <= row["ppid"]) & (_att[:, 0] >= row["ppid"] - 4) & (np.abs(_att[:, 1] - row["start_time"]) < 2.0)
        return bool(m.any())
    units2["helper_suspect"] = units2.apply(_is_helper, axis=1).astype(int) if len(_att) else 0
    units1["helper_suspect"] = 0
    manifest["helper_suspects"] = int(units2["helper_suspect"].sum())
    if not args.mask_signature:
        # Release artefact: the suspect lists behind label views A, B and C (rule-derived, detector-independent).
        sus_dir = paths.outputs / "suspects"
        sus_dir.mkdir(parents=True, exist_ok=True)
        _cols = ["row_id", "pid", "ppid", "name", "start_time", "command line"]
        units2.loc[units2["staging_suspect"] == 1, _cols].to_csv(sus_dir / "staging_suspects.csv", index=False)
        units2.loc[units2["helper_suspect"] == 1, _cols].to_csv(sus_dir / "helper_suspects.csv", index=False)
    # Chunks are cut on the FULL unit set so that every relabel mode shares the same fold boundaries.
    folds_full = (rolling_folds_time if args.fold_mode == "time" else rolling_folds)(units2, "start_time", args.n_folds)
    chunk_of = np.full(len(units2), 0)
    for k, (_, te) in enumerate(folds_full, start=1):
        chunk_of[te] = k
    units2["chunk"] = chunk_of
    # Evidence side of the investigator (baseline counts, command-line mentions): every recorded process, label-free
    # columns only, taken BEFORE any relabel so that a label-derived exclusion cannot shape what a lookup returns.
    units2_evidence = units2[["pid", "name", "parent_name", "start_time", "command line"]].copy()
    # The LLM benign sample is drawn on the full set too, so cached scores are reused across relabel modes.
    rng = np.random.default_rng(args.seed)
    attack_idx_full = np.flatnonzero(units2["is_attack"].to_numpy() == 1)
    benign_idx_full = np.flatnonzero(units2["is_attack"].to_numpy() == 0)
    if args.smoke:
        attack_idx_full = rng.choice(attack_idx_full, size=min(args.smoke, len(attack_idx_full)), replace=False)
        benign_idx_full = rng.choice(benign_idx_full, size=min(args.smoke, len(benign_idx_full)), replace=False)
        p1_idx = np.array([], dtype=int)
    else:
        if args.n_benign >= 0:
            benign_idx_full = rng.choice(benign_idx_full, size=min(args.n_benign, len(benign_idx_full)), replace=False)
        # n_benign < 0: score every benign Phase 2 unit, so LLM metrics live on the same pooled set as the forest.
        p1_idx = rng.choice(len(units1), size=min(args.n_benign_phase1, len(units1)), replace=False)
    sampled_row_ids = set(units2["row_id"].to_numpy()[np.concatenate([attack_idx_full, benign_idx_full])])
    suspect = units2["staging_suspect"] == 1
    if args.relabel.endswith("_helpers"):
        suspect = suspect | (units2["helper_suspect"] == 1)
    if args.relabel.startswith("positive"):
        units2.loc[suspect & (units2["staging_suspect"] == 1), "attack_tactic"] = "collection(staging-relabelled)"
        units2.loc[suspect & (units2["helper_suspect"] == 1), "attack_tactic"] = "execution(helper-relabelled)"
        units2.loc[suspect, "is_attack"] = 1
    elif args.relabel.startswith("exclude"):
        units2 = units2[~suspect].reset_index(drop=True)
    folds = [(np.flatnonzero(units2["chunk"].to_numpy() < k), np.flatnonzero(units2["chunk"].to_numpy() == k)) for k in range(1, args.n_folds + 1)]
    chunk_of = units2["chunk"].to_numpy()
    manifest["chunks"] = {int(k): {"n": int((units2.chunk == k).sum()), "n_attack": int(units2.loc[units2.chunk == k, "is_attack"].sum()),
                                   "t_min": float(units2.loc[units2.chunk == k, "start_time"].min()),
                                   "tactics": {kk: int(v) for kk, v in units2.loc[(units2.chunk == k) & (units2.is_attack == 1), "attack_tactic"].value_counts().items()}}
                          for k in sorted(set(chunk_of))}

    # --------------------------------------------------------- LLM sampling (drawn above on the full set)
    sample2 = units2[units2["row_id"].isin(sampled_row_ids)].copy()
    benign_idx = benign_idx_full
    sample1 = units1.iloc[np.sort(p1_idx)].copy() if len(p1_idx) else units1.iloc[0:0].copy()
    sample1["chunk"] = 0
    llm_sample = pd.concat([sample1, sample2], ignore_index=True)
    manifest["llm_sample"] = {"n": int(len(llm_sample)), "n_attack": int(llm_sample.is_attack.sum()),
                              "n_benign_phase2": int(len(benign_idx)), "n_benign_phase1": int(len(p1_idx))}

    # ------------------------------------------------------------ LLM scoring
    llm_scores: Dict[str, pd.DataFrame] = {}
    if conditions:
        cache = LLMCache(str(paths.outputs / "llm_cache_v3.sqlite"))
        if args.llm_backend == "chat":
            if not args.llm_model:
                raise RuntimeError("--llm-backend chat needs --llm-model")
            cfg = LLMConfig(mode="openai", backend="chat", base_url=args.llm_base_url, model=args.llm_model, temperature=rt.openai_temperature,
                            max_output_tokens=512, api_key=os.environ.get(args.llm_key_env, ""), tool_budget=args.tool_budget,
                            mask_signature=llm_mask, cache_only=args.cache_only, seed=args.llm_seed, no_think=args.llm_no_think,
                            investigator_max_output_tokens=args.investigator_max_tokens)
            if not cfg.api_key and not args.cache_only:
                raise RuntimeError(f"{args.llm_key_env} missing (experiments/.env)")
        else:
            cfg = LLMConfig(mode="openai", model=rt.openai_model, temperature=rt.openai_temperature, max_output_tokens=max(rt.llm_max_output_tokens, 300),
                            api_key=rt.openai_api_key, tool_budget=args.tool_budget, mask_signature=llm_mask, cache_only=args.cache_only,
                            investigator_max_output_tokens=args.investigator_max_tokens)
            if not cfg.api_key:
                raise RuntimeError("OPENAI_API_KEY missing (experiments/.env)")
        scorer = SemanticScorer(cfg, cache=cache)
        emus = {1: emu1, 2: emu2}

        # Investigator only: host baseline (counts, no labels), path history, and one case notebook per phase.
        if "investigator" in conditions:
            _bl_cols = ["name", "parent_name", "command line", "start_time"]
            baselines = {1: HostBaseline(units1[_bl_cols], units1[_bl_cols]), 2: HostBaseline(units1[_bl_cols], units2_evidence[_bl_cols])}
            _pi_cols = ["pid", "name", "start_time", "command line"]
            path_index = {1: PathIndex(emu1, units1[_pi_cols]), 2: PathIndex(emu2, units2_evidence[_pi_cols])}
            notebooks = {1: CaseNotebook(), 2: CaseNotebook()}
            trajectories: List[Dict] = []

        def investigate(row: pd.Series):
            """Memory is retrieved into the record before the call and written after it, so records must be judged
            one at a time in decision-time order (see the loop below)."""
            ph = int(row["phase"])
            emu, nb, t, pid = emus[ph], notebooks[ph], float(row["decision_time"]), int(row["pid"])
            record = behaviour_for_unit(row, emu, args.window, baseline=baselines[ph], mask=llm_mask)
            tree = emu.get_process_tree(pid, t, depth=6)
            ancestors = [int(p["pid"]) for p in tree[1:] if p.get("pid") is not None]
            paths = unit_paths(row, emu, args.window)
            notes = [] if args.investigator_no_memory else nb.retrieve(pid=pid, ancestors=ancestors, ppid=int(row["ppid"]), paths=paths, t=t)
            record += "\n" + memory_section(notes)
            extra_tools = {"get_path_history": lambda a: path_index[ph].history(str(a.get("path", "")), t),
                           "get_host_baseline": lambda a: baselines[ph].lookup(str(a.get("name", "")), a.get("parent_name") or None, float(row["start_time"]))}
            if not args.investigator_no_memory:
                extra_tools["recall_case_notes"] = lambda a: nb.about(int(a.get("pid", -1)), t)
            res = scorer.score("investigator", row["unit_key"], record, emulator=emu, pid=pid, decision_time=t, extra_tools=extra_tools)
            if not res.error and not res.parse_failed:
                cmd = row.get("command line")
                nb.add(pid=pid, ppid=int(row["ppid"]), name=str(row.get("name", "")), command_line=cmd if isinstance(cmd, str) else "", decision_time=t,
                       verdict=res.verdict, score=res.score, stage=res.extra.get("stage", ""), note=res.extra.get("note", "") or res.evidence, paths=paths)
            trajectories.append({"row_id": row["row_id"], "record": record, "n_memory_notes": len(notes), "trajectory": res.trajectory,
                                 "lookups": [{k: c.get(k) for k in ("tool", "args", "step", "n_records")} for c in res.tool_calls],
                                 "verdict": res.verdict, "score": res.score, "evidence": res.evidence, **res.extra})
            return res

        def job(cond: str, row: pd.Series) -> Dict:
            emu = emus[int(row["phase"])]
            if cond == "investigator":
                res = investigate(row)
                return {"row_id": row["row_id"], "unit_key": row["unit_key"], "condition": cond, "score": res.score, "verdict": res.verdict, "evidence": res.evidence,
                        "input_tokens": res.input_tokens, "output_tokens": res.output_tokens, "latency_s": res.latency_s,
                        "n_api_calls": res.n_api_calls, "n_tool_calls": len(res.tool_calls), "tools": ",".join(c["tool"] for c in res.tool_calls),
                        "cached": res.cached, "error": res.error, "model_snapshot": res.model_snapshot, "response_id": res.response_id,
                        "parse_failed": bool(getattr(res, "parse_failed", False))}
            if cond == "narrative":
                narrative = narrative_only_for_unit(row)
            elif cond == "sweep":
                narrative = sweep_for_unit(row, emu, args.window, mask=llm_mask)
            else:
                narrative = narrative_for_unit(row, emu, args.window, mask=llm_mask)
            res: LLMResult = scorer.score(cond, row["unit_key"], narrative, emulator=emu, pid=int(row["pid"]), decision_time=float(row["decision_time"]))
            return {"row_id": row["row_id"], "unit_key": row["unit_key"], "condition": cond, "score": res.score, "verdict": res.verdict, "evidence": res.evidence,
                    "input_tokens": res.input_tokens, "output_tokens": res.output_tokens, "latency_s": res.latency_s,
                    "n_api_calls": res.n_api_calls, "n_tool_calls": len(res.tool_calls), "tools": ",".join(c["tool"] for c in res.tool_calls),
                    "cached": res.cached, "error": res.error, "model_snapshot": res.model_snapshot, "response_id": res.response_id,
                    "parse_failed": bool(getattr(res, "parse_failed", False))}

        for cond in conditions:
            rows = []
            t1 = time.time()
            # The investigator has memory: one worker, records in (phase, decision time) order, so that a note exists
            # before any later record can retrieve it and a rerun from the cache rebuilds the same notebook.
            ordered = llm_sample.sort_values(["phase", "decision_time"], kind="stable") if cond == "investigator" else llm_sample
            with ThreadPoolExecutor(max_workers=1 if cond == "investigator" else args.workers) as ex:
                futs = [ex.submit(job, cond, r) for _, r in ordered.iterrows()]
                for i, f in enumerate(as_completed(futs), 1):
                    rows.append(f.result())
                    if i % 200 == 0:
                        print(f"  [{cond}] {i}/{len(futs)} scored ({time.time() - t1:.0f}s)", flush=True)
            df = pd.DataFrame(rows)
            n_err = int((df["error"] != "").sum())
            print(f"[{cond}] done: {len(df)} units, {n_err} errors, {int(df.cached.sum())} cached, {time.time() - t1:.0f}s", flush=True)
            # An errored unit (quota, parse failure) has no verdict. It must not enter any metric as a 0.5
            # score (which the >= 0.5 rule would count as MALICIOUS); it is simply unscored.
            manifest.setdefault("llm_errors", {})[cond] = n_err
            manifest.setdefault("llm_parse_failures", {})[cond] = int(df["parse_failed"].sum())
            df = df[df["error"] == ""]
            llm_scores[cond] = df.set_index("row_id")
            manifest.setdefault("model_snapshots", {})[cond] = {str(k): int(v) for k, v in df["model_snapshot"].fillna("").value_counts().items()}

    # ----------------------------------------------------------- RF per fold
    feature_spec = process_unit_features(pd.concat([units1, units2]), use_context=not args.no_context, use_cmdline=not args.no_cmdline, use_ancestry=not args.no_ancestry)
    manifest["feature_spec"] = feature_spec.__dict__
    rf_cfg = RandomForestConfig()
    manifest["rf_config"] = rf_cfg.__dict__

    pooled = units2[units2.chunk > 0].copy()
    pooled["p_rf_raw"] = np.nan
    pooled["p_rf_cal"] = np.nan
    pooled["p_rf_rethr"] = np.nan
    pooled["p_rf_thrOOB"] = np.nan
    fold_metrics = []
    fused_rows: Dict[str, List[pd.DataFrame]] = {c: [] for c in conditions}
    fold_params = []

    for k, (tr_idx, te_idx) in enumerate(folds, start=1):
        train = pd.concat([units1, units2.iloc[tr_idx]], ignore_index=True)
        test = units2.iloc[te_idx]
        if units2_masked is not None:
            # Same records, same index (so pooled.loc[test.index] still lines up), masked command lines and features.
            test = units2_masked.loc[test["row_id"].to_numpy()].reset_index().set_index(test.index)
        pipe = build_feature_pipeline(feature_spec)
        X_tr = pipe.fit_transform(train)
        y_tr = train["is_attack"].to_numpy().astype(int)
        X_te = pipe.transform(test)
        y_te = test["is_attack"].to_numpy().astype(int)
        if y_tr.sum() == 0:
            print(f"fold {k}: no attacks in training data; skipped", flush=True)
            fold_metrics.append({"fold": k, "skipped": "no training attacks", "n_test": int(len(test)), "n_test_attack": int(y_te.sum())})
            continue
        rf = train_random_forest(X_tr, y_tr, rf_cfg)
        p_oob = oob_proba(rf, X_tr)
        iso = fit_isotonic(y_tr, p_oob)
        p_te = predict_proba(rf, X_te)
        p_te_cal = iso.transform(p_te)
        pooled.loc[test.index, "p_rf_raw"] = p_te
        pooled.loc[test.index, "p_rf_cal"] = p_te_cal
        # RF-only references with the decision rule fit on training OOB scores (same information as fusion/stacking get).
        if y_tr.sum() >= 2:
            lr_rf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(_logit(p_oob).reshape(-1, 1), y_tr)
            pooled.loc[test.index, "p_rf_rethr"] = lr_rf.predict_proba(_logit(p_te).reshape(-1, 1))[:, 1]
            cands = np.unique(np.round(p_oob, 4))
            f1s = [compute_metrics(y_tr, p_oob, threshold=c).f1 for c in cands]
            thr = float(cands[int(np.argmax(f1s))])
            pooled.loc[test.index, "p_rf_thrOOB"] = (p_te >= thr).astype(float)
            fold_thr = thr
        else:
            pooled.loc[test.index, "p_rf_rethr"] = p_te
            pooled.loc[test.index, "p_rf_thrOOB"] = (p_te >= 0.5).astype(float)
            fold_thr = 0.5
        fm = {"fold": k, "oob_f1_threshold": fold_thr, "n_train": int(len(train)), "n_train_attack": int(y_tr.sum()), "n_test": int(len(test)), "n_test_attack": int(y_te.sum()),
              "rf_raw": compute_metrics(y_te, p_te).__dict__ if y_te.sum() else None,
              "rf_cal": compute_metrics(y_te, p_te_cal).__dict__ if y_te.sum() else None}
        fold_metrics.append(fm)
        print(f"fold {k}: train={len(train)} ({int(y_tr.sum())} attacks) test={len(test)} ({int(y_te.sum())} attacks) "
              f"RF raw R={fm['rf_raw']['recall'] if fm['rf_raw'] else float('nan'):.3f} P={fm['rf_raw']['precision'] if fm['rf_raw'] else float('nan'):.3f}", flush=True)

        # Per-fold fusion / stacking for each LLM condition.
        if conditions:
            tr_keys = train["row_id"].to_numpy()
            p_tr_cal = iso.transform(p_oob)
            for cond in conditions:
                sc = llm_scores[cond]
                tr_mask = np.isin(tr_keys, sc.index)
                te_keys = test["row_id"].to_numpy()
                te_mask = np.isin(te_keys, sc.index)
                if te_mask.sum() == 0:
                    continue
                s_tr = sc.loc[tr_keys[tr_mask], "score"].to_numpy(dtype=float)
                y_trs = y_tr[tr_mask]
                p_trs = p_tr_cal[tr_mask]
                s_te = sc.loc[te_keys[te_mask], "score"].to_numpy(dtype=float)
                p_tes = p_te_cal[te_mask]
                if y_trs.sum() >= 2 and (y_trs == 0).sum() >= 2:
                    T, s_tr_cal = temperature_scale(_logit(s_tr), y_trs)
                    fus = grid_search_alpha_beta(y_trs, p_trs, s_tr_cal)
                    stack = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(np.c_[_logit(p_trs), _logit(s_tr_cal)], y_trs)
                    params = {"fold": k, "condition": cond, "temperature": float(T), "alpha": fus.alpha, "beta": fus.beta,
                              "stack_coef": stack.coef_.ravel().tolist(), "n_train_scored": int(tr_mask.sum()), "n_train_scored_attack": int(y_trs.sum())}
                else:
                    T, fus_alpha, fus_beta, stack = 1.0, 1.0, 1.0, None
                    fus = type("F", (), {"alpha": 1.0, "beta": 1.0})()
                    params = {"fold": k, "condition": cond, "temperature": 1.0, "alpha": 1.0, "beta": 1.0, "stack_coef": None,
                              "n_train_scored": int(tr_mask.sum()), "n_train_scored_attack": int(y_trs.sum()), "note": "insufficient training positives; identity fusion"}
                fold_params.append(params)
                s_te_cal = _sigmoid(_logit(s_te) / T)
                fused = fuse_logit(p_tes, s_te_cal, fus.alpha, fus.beta)
                stacked = stack.predict_proba(np.c_[_logit(p_tes), _logit(s_te_cal)])[:, 1] if stack is not None else fused
                fused_rows[cond].append(pd.DataFrame({"row_id": te_keys[te_mask], "fold": k, "y": y_te[te_mask],
                                                      "p_rf_cal": p_tes, "p_rf_raw": p_te[te_mask], "s_llm_raw": s_te, "s_llm_cal": s_te_cal,
                                                      "p_fused": fused, "p_stacked": stacked}))

    # ---------------------------------------------------------- pooled eval
    y = pooled["is_attack"].to_numpy().astype(int)
    tactics = pooled["attack_tactic"].fillna("").astype(str).to_numpy()
    own = pooled["command line"].fillna("").astype(str).str.contains("base64|eval", regex=True).to_numpy()
    if units2_masked is not None:
        # The rule is scored on what the detectors saw at test time; the exported command lines stay unmasked.
        pooled["ancestor_cmdline_masked"] = units2_masked.loc[pooled["row_id"].to_numpy(), "ancestor_cmdline"].to_numpy()
        pooled["command_line_masked"] = units2_masked.loc[pooled["row_id"].to_numpy(), "command line"].to_numpy()
        anc = pooled["ancestor_cmdline_masked"].fillna("").astype(str).str.contains("base64|eval", regex=True).to_numpy()
    else:
        anc = pooled["ancestor_cmdline"].fillna("").astype(str).str.contains("base64|eval", regex=True).to_numpy()
    rf_hit = pooled["p_rf_raw"].to_numpy() >= 0.5
    results = {"manifest": manifest, "folds": fold_metrics, "fold_params": fold_params,
               "pooled_n": int(len(pooled)), "pooled_n_attack": int(y.sum()),
               "rf_raw": evaluate("rf_raw", y, pooled["p_rf_raw"], args.n_boot, args.seed),
               "rf_calibrated": evaluate("rf_calibrated", y, pooled["p_rf_cal"], args.n_boot, args.seed),
               "rf_rethresholded_logistic": evaluate("rf_rethresholded_logistic", y, pooled["p_rf_rethr"], args.n_boot, args.seed),
               "rf_threshold_oob_f1": evaluate("rf_threshold_oob_f1", y, pooled["p_rf_thrOOB"], 0, args.seed),
               "rule_ancestor_b64_eval": evaluate("rule_ancestor_b64_eval", y, anc.astype(float), 0, args.seed),
               "rf_raw_or_ancestor_rule": evaluate("rf_raw_or_ancestor_rule", y, (rf_hit | anc).astype(float), 0, args.seed),
               "tactic_breakdown": {"rf_raw": tactic_table(tactics, y, rf_hit.astype(int)),
                                    "rf_calibrated": tactic_table(tactics, y, (pooled["p_rf_cal"].to_numpy() >= 0.5).astype(int)),
                                    "rf_rethresholded_logistic": tactic_table(tactics, y, (pooled["p_rf_rethr"].to_numpy() >= 0.5).astype(int)),
                                    "rule_ancestor": tactic_table(tactics, y, anc.astype(int)),
                                    "rf_raw_or_ancestor_rule": tactic_table(tactics, y, (rf_hit | anc).astype(int))}}

    out_name = f"rolling{args.n_folds}" + ("_time" if args.fold_mode == "time" else "") + ("_allbenign" if args.n_benign < 0 and conditions else "") + ("_masked" if args.mask_signature else "") + ("_decode" if llm_mask and args.mask_level == "decode" else "") + ("_masktest" if args.mask_test_only else "") + (f"_{args.relabel}" if args.relabel != "none" else "") + ("_noctx" if args.no_context else "") + \
        ("_nocmd" if args.no_cmdline else "") + ("_noanc" if args.no_ancestry else "") + \
        (("_m-" + re.sub(r"[^A-Za-z0-9.]+", "-", args.llm_model.split("/")[-1].split(":")[0]).strip("-").lower()) if (args.llm_backend == "chat" and conditions) else "") + \
        (f"_{args.tag}" if args.tag else "")
    out_dir = paths.outputs / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Every LLM score, including chunk 0 and the Phase 1 sample (training side of each fold). Used by
    # benign_filter.py to derive a benign-behaviour filter from pre-test records only.
    _meta = llm_sample.set_index("row_id")[["phase", "chunk", "is_attack", "pid", "ppid", "exe", "name", "parent_name", "start_time"]]
    for cond, sc in llm_scores.items():
        _meta.join(sc[["score", "verdict", "parse_failed"]], how="inner").to_csv(out_dir / f"llm_scores_{cond}.csv")
    if "investigator" in conditions:
        with open(out_dir / "investigator_trajectories.jsonl", "w") as fh:
            for tr in trajectories:
                fh.write(json.dumps(tr, default=str) + "\n")
    keep = ["p_rf_rethr", "p_rf_thrOOB", "row_id", "unit_key", "pid", "ppid", "exe", "name", "command line", "ancestor_cmdline", "start_time", "decision_time", "chunk", "is_attack", "attack_tactic", "p_rf_raw", "p_rf_cal"]
    keep += [c for c in ("command_line_masked", "ancestor_cmdline_masked") if c in pooled.columns]
    pooled[keep].to_csv(out_dir / "test_predictions.csv", index=False)

    # LLM conditions on the scored test subset (pooled across folds).
    for cond in conditions:
        if not fused_rows[cond]:
            continue
        df = pd.concat(fused_rows[cond], ignore_index=True)
        sc = llm_scores[cond]
        df = df.join(sc[["verdict", "evidence", "input_tokens", "output_tokens", "latency_s", "n_api_calls", "n_tool_calls", "tools", "error", "model_snapshot", "response_id", "parse_failed"]], on="row_id")
        df = df.join(pooled.set_index("row_id")[["unit_key", "attack_tactic", "name", "command line", "p_rf_rethr"]], on="row_id")
        df.to_csv(out_dir / f"llm_{cond}_predictions.csv", index=False)
        yy = df["y"].to_numpy().astype(int)
        tt = df["attack_tactic"].fillna("").astype(str).to_numpy()
        block = {"n": int(len(df)), "n_attack": int(yy.sum()),
                 "rf_calibrated_same_subset": evaluate("rf_calibrated_same_subset", yy, df["p_rf_cal"], args.n_boot, args.seed),
                 "rf_raw_same_subset": evaluate("rf_raw_same_subset", yy, df["p_rf_raw"], args.n_boot, args.seed),
                 "rf_rethresholded_same_subset": evaluate("rf_rethresholded_same_subset", yy, df["p_rf_rethr"], args.n_boot, args.seed),
                 "union_rf_or_llm": evaluate("union_rf_or_llm", yy, ((df["p_rf_raw"] >= 0.5) | (df["s_llm_raw"] >= 0.5)).astype(float), 0, args.seed),
                 "llm_raw": evaluate("llm_raw", yy, df["s_llm_raw"], args.n_boot, args.seed),
                 "llm_calibrated": evaluate("llm_calibrated", yy, df["s_llm_cal"], args.n_boot, args.seed),
                 "fused": evaluate("fused", yy, df["p_fused"], args.n_boot, args.seed),
                 "stacked": evaluate("stacked", yy, df["p_stacked"], args.n_boot, args.seed),
                 "tactic_breakdown": {"llm_raw": tactic_table(tt, yy, (df["s_llm_raw"] >= 0.5).astype(int).to_numpy()),
                                      "fused": tactic_table(tt, yy, (df["p_fused"] >= 0.5).astype(int).to_numpy()),
                                      "rf_raw": tactic_table(tt, yy, (df["p_rf_raw"] >= 0.5).astype(int).to_numpy())},
                 "cost": {"mean_input_tokens": float(df["input_tokens"].mean()), "mean_output_tokens": float(df["output_tokens"].mean()),
                          "mean_latency_s": float(df.loc[~df["error"].astype(bool), "latency_s"].mean()) if (~df["error"].astype(bool)).any() else None,
                          "mean_api_calls": float(df["n_api_calls"].mean()), "mean_tool_calls": float(df["n_tool_calls"].mean()),
                          "tool_call_histogram": {str(k): int(v) for k, v in df["n_tool_calls"].value_counts().sort_index().items()},
                          "tools_used": {str(k): int(v) for k, v in pd.Series(",".join(df["tools"].fillna("")).split(",")).value_counts().items() if k},
                          "n_errors": int(df["error"].astype(bool).sum()),
                          "est_usd_total": float(df["input_tokens"].sum() / 1e6 * PRICE_PER_M.get(rt.openai_model, (0, 0))[0] + df["output_tokens"].sum() / 1e6 * PRICE_PER_M.get(rt.openai_model, (0, 0))[1])}}
        results[f"llm_{cond}"] = block
        plot_calibration(yy, df["s_llm_cal"].to_numpy(), out_dir / f"calibration_llm_{cond}.png")
        plot_pr(yy, df["p_fused"].to_numpy(), out_dir / f"pr_fused_{cond}.png")
        print(f"[{cond}] n={len(df)} attacks={int(yy.sum())} | RF cal F1={block['rf_calibrated_same_subset']['f1']:.3f} | LLM raw F1={block['llm_raw']['f1']:.3f} "
              f"R={block['llm_raw']['recall']:.3f} P={block['llm_raw']['precision']:.3f} | fused F1={block['fused']['f1']:.3f} | stacked F1={block['stacked']['f1']:.3f} "
              f"| tool calls/unit={block['cost']['mean_tool_calls']:.2f} | est ${block['cost']['est_usd_total']:.4f}", flush=True)

    manifest["elapsed_seconds"] = round(time.time() - t0, 1)
    save_json(out_dir / "metrics.json", json.loads(json.dumps(results, default=lambda o: o.item() if hasattr(o, "item") else str(o))))
    save_json(paths.outputs / "runs" / f"{out_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", json.loads(json.dumps(results, default=str)))
    plot_calibration(y, pooled["p_rf_cal"].to_numpy(), out_dir / "calibration_rf.png")
    plot_pr(y, pooled["p_rf_raw"].to_numpy(), out_dir / "pr_rf.png")
    r, c = results["rf_raw"], results["rf_calibrated"]

    def ci(b):
        return f"CI=[{b['ci95']['f1']['lo']:.3f},{b['ci95']['f1']['hi']:.3f}]" if b.get("ci95") else "CI=n/a"

    print(f"[{out_name}] pooled n={len(pooled)} attacks={int(y.sum())}")
    print(f"  RF raw : P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} {ci(r)} AUPRC={r['auprc']:.3f} R@FPR1e-3={r['recall_at_fpr_0.001']:.3f}")
    print(f"  RF cal : P={c['precision']:.3f} R={c['recall']:.3f} F1={c['f1']:.3f} {ci(c)} ECE={c['ece']:.4f}")
    a, o = results["rule_ancestor_b64_eval"], results["rf_raw_or_ancestor_rule"]
    print(f"  ancestry rule: R={a['recall']:.3f} P={a['precision']:.3f} | RF OR rule: R={o['recall']:.3f} P={o['precision']:.3f}")
    print(f"  outputs: {out_dir}")


if __name__ == "__main__":
    main()
