"""Label-free reference detector: an Isolation Forest fitted on the attack-free reference period only.

The zero-shot language models see no labelled attack. Their honest statistical peer is a detector that sees none
either: it learns what the host looks like in Phase 1 (the attack-free baseline the dataset ships) and scores how
far each Phase 2 process record departs from that. No attack label enters training, calibration or thresholding.

* Features: the dense summary features of the process-record pipeline (command-line and ancestry flags, event counts,
  user ids), fitted on Phase 1 records. The hashed command-line text is left out by default: an Isolation Forest
  draws split features at random, and 8,192 mostly-empty hash columns drown the few informative ones (AUROC 0.47
  with them, 0.96 without; ``--features full`` reproduces the former).
* Model: Isolation Forest on those features. Two novelty rules with no model at all are reported next to it:
  a (name <- parent name) pair never seen in Phase 1, and a command line never seen in Phase 1.
* Operating points chosen without labels: alert when the anomaly score exceeds the q-quantile of the Phase 1 scores,
  i.e. the alert rate an operator accepts on a clean host (1 in 100, 1 in 1,000, 1 in 10,000 records).
* Test set: the pooled test chunks of an existing rolling run (its test_predictions.csv fixes row ids, chunks and
  the label view), so every detector is scored on identical records.
* A contaminated variant refits per fold on Phase 1 plus the earlier Phase 2 chunks with no labels, which is the
  realistic case of a baseline period that already contains the intruder.

Usage: .venv/bin/python label_free.py outputs/rolling5_exclude_staging [--contaminated]
Writes label_free_predictions.csv and label_free.json in the run directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import default_paths
from data.provenance import load_provenance
from emulator import Emulator, strip_labels
from features import FeatureSpec, build_feature_pipeline, process_unit_features
from metrics import bootstrap_metrics, compute_metrics, recall_at_fpr
from units import build_process_units

ALERT_RATES = (0.01, 0.001, 0.0001)


def _eval(y: np.ndarray, alert: np.ndarray, score: np.ndarray, n_boot: int, seed: int) -> Dict:
    m = compute_metrics(y, alert.astype(float))
    tn, fp, fn, tp = m.confusion
    out = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "precision": m.precision, "recall": m.recall, "f1": m.f1,
           "fp_per_1000_benign": float(1000.0 * fp / (fp + tn)) if (fp + tn) else float("nan")}
    if n_boot:
        out["ci95"] = bootstrap_metrics(y, alert.astype(float), n_boot=n_boot, seed=seed)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Isolation Forest fitted on the attack-free reference period; no attack label used.")
    ap.add_argument("run_dir", type=str, help="rolling run directory whose test_predictions.csv fixes the test records and label view")
    ap.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--contaminated", action="store_true", help="refit per fold on Phase 1 + earlier Phase 2 chunks, unlabelled")
    ap.add_argument("--features", type=str, choices=["numeric", "numeric+categorical", "full"], default="numeric")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()

    root = Path(args.root)
    paths = default_paths(root)
    run_dir = Path(args.run_dir)
    test = pd.read_csv(run_dir / "test_predictions.csv")

    phase1 = load_provenance(paths.provenance_phase1)
    phase2 = load_provenance(paths.provenance_phase2)
    phase1["label"] = 0
    phase1["subLabel"] = ""
    emu1 = Emulator(provenance=strip_labels(phase1))
    emu2 = Emulator(provenance=strip_labels(phase2))
    units1 = build_process_units(phase1, emu1, window=args.window)
    units2 = build_process_units(phase2, emu2, window=args.window)
    for u, ph in ((units1, 1), (units2, 2)):
        u["row_id"] = [f"p{ph}:{pid}:{st:.3f}:{pp}:{e}:{n}" for pid, st, pp, e, n in zip(u["pid"], u["start_time"], u["ppid"], u["exe"], u["name"])]
    units2 = units2.set_index("row_id")
    missing = set(test["row_id"]) - set(units2.index)
    assert not missing, f"{len(missing)} test records not found among rebuilt units"
    pooled = units2.loc[test["row_id"]].copy()
    pooled["chunk"] = test["chunk"].to_numpy()
    pooled["is_attack"] = test["is_attack"].to_numpy()  # labels are used for scoring only, below
    full = process_unit_features(pd.concat([units1, units2]))
    spec = {"numeric": FeatureSpec(numeric=full.numeric, categorical=[], text=[]),
            "numeric+categorical": FeatureSpec(numeric=full.numeric, categorical=full.categorical, text=[]), "full": full}[args.features]
    y = pooled["is_attack"].to_numpy().astype(int)
    results: Dict = {"run_dir": str(run_dir), "n_test": int(len(pooled)), "n_attack": int(y.sum()), "n_reference": int(len(units1)),
                     "features": spec.__dict__, "n_estimators": args.n_estimators, "contaminated": args.contaminated}

    def fit_and_score(train: pd.DataFrame, target: pd.DataFrame):
        pipe = build_feature_pipeline(spec)
        X_tr = pipe.fit_transform(train)
        model = IsolationForest(n_estimators=args.n_estimators, random_state=args.seed, n_jobs=-1).fit(X_tr)
        # higher = more anomalous
        return -model.score_samples(X_tr), -model.score_samples(pipe.transform(target))

    score = np.full(len(pooled), np.nan)
    thresholds = {r: np.full(len(pooled), np.nan) for r in ALERT_RATES}
    if not args.contaminated:
        ref_score, score = fit_and_score(units1, pooled)
        for r in ALERT_RATES:
            thresholds[r][:] = np.quantile(ref_score, 1 - r)
    else:
        for k in sorted(set(pooled["chunk"])):
            # every Phase 2 record that started before this chunk (chunk 0 included), labels never read
            t_min = pooled.loc[pooled["chunk"] == k, "start_time"].min()
            earlier = units2[units2["start_time"] < t_min]
            train = pd.concat([units1, earlier.drop(columns=[c for c in ("is_attack", "attack_tactic") if c in earlier.columns])], ignore_index=True)
            m = (pooled["chunk"] == k).to_numpy()
            ref_score, sc = fit_and_score(train, pooled[m])
            score[m] = sc
            for r in ALERT_RATES:
                thresholds[r][m] = np.quantile(ref_score, 1 - r)
    pooled["anomaly_score"] = score
    results["auprc"] = compute_metrics(y, (score - score.min()) / (score.max() - score.min())).auprc
    from sklearn.metrics import roc_auc_score
    results["auroc"] = float(roc_auc_score(y, score))
    ref_all, _ = fit_and_score(units1, units1.iloc[:1]) if args.contaminated else (ref_score, None)
    results["mean_reference_percentile_of_attacks"] = float(np.mean([np.mean(ref_all < v) for v in score[y == 1]]))
    results.update(recall_at_fpr(y, score))
    results["operating_points"] = {}
    for r in ALERT_RATES:
        alert = score >= thresholds[r]
        pooled[f"alert_rate_{r:g}"] = alert.astype(int)
        results["operating_points"][f"alert_rate_{r:g}"] = _eval(y, alert, score, args.n_boot, args.seed)
    # Novelty rules: no model, no threshold; "never seen in the reference period".
    ref_sig = set(units1["name"].astype(str) + "<-" + units1["parent_name"].astype(str))
    ref_cmd = set(units1["command line"].dropna().astype(str))
    pooled["novel_signature"] = (~(pooled["name"].astype(str) + "<-" + pooled["parent_name"].astype(str)).isin(ref_sig)).astype(int)
    pooled["novel_command_line"] = (~pooled["command line"].astype(str).isin(ref_cmd)).astype(int)
    # Fingerprint rule: the Caldera launcher idiom (the word eval) in the record's own or an ancestor's command line.
    # No model, no threshold, no label; it measures how much of the campaign a one-line text match recovers.
    _own = pooled["command line"].fillna("").astype(str)
    _anc = pooled["ancestor_cmdline"].fillna("").astype(str)
    pooled["fingerprint_eval"] = (_own.str.contains(r"\beval\b", regex=True) | _anc.str.contains(r"\beval\b", regex=True)).astype(int)
    results["novelty_rules"] = {c: _eval(y, pooled[c].to_numpy() == 1, score, args.n_boot, args.seed) for c in ("novel_signature", "novel_command_line", "fingerprint_eval")}
    # Oracle threshold (best F1 on the test labels) is reported only as an upper bound, never as an operating point.
    cands = np.unique(np.round(score, 4))
    f1s = [compute_metrics(y, score, threshold=c).f1 for c in cands[:: max(1, len(cands) // 400)]]
    results["oracle_best_f1_upper_bound"] = float(max(f1s))
    tactics = pooled["attack_tactic"].fillna("").astype(str)
    results["fp_by_name_at_1e-3"] = {k: int(v) for k, v in pooled.loc[(y == 0) & (pooled["alert_rate_0.001"] == 1), "name"].value_counts().head(15).items()}
    results["recall_by_tactic_at_1e-3"] = {t: {"support": int(((tactics == t) & (y == 1)).sum()), "recall": float(pooled.loc[(tactics == t) & (y == 1), "alert_rate_0.001"].mean())}
                                           for t in sorted(set(tactics[y == 1])) if t}

    suffix = ("_contaminated" if args.contaminated else "") + ("" if args.features == "numeric" else f"_{args.features.replace('+', '_')}")
    keep = ["chunk", "pid", "ppid", "exe", "name", "command line", "ancestor_cmdline", "start_time", "is_attack", "attack_tactic", "anomaly_score", "novel_signature", "novel_command_line", "fingerprint_eval"] + [f"alert_rate_{r:g}" for r in ALERT_RATES]
    pooled[keep].to_csv(run_dir / f"label_free{suffix}_predictions.csv", index_label="row_id")
    with open(run_dir / f"label_free{suffix}.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"[label-free{suffix}] n={len(pooled)} attacks={int(y.sum())} AUPRC={results['auprc']:.3f} R@FPR1e-3={results['recall_at_fpr_0.001']:.3f}")
    for r in ALERT_RATES:
        o = results["operating_points"][f"alert_rate_{r:g}"]
        print(f"  alert rate {r:g}: TP {o['tp']} / FP {o['fp']}  P={o['precision']:.3f} R={o['recall']:.3f} F1={o['f1']:.3f}  FP/k={o['fp_per_1000_benign']:.1f}")
    for c, o in results["novelty_rules"].items():
        print(f"  {c:20s}: TP {o['tp']} / FP {o['fp']}  P={o['precision']:.3f} R={o['recall']:.3f} F1={o['f1']:.3f}  FP/k={o['fp_per_1000_benign']:.1f}")
    print(f"  oracle best F1 (label-chosen threshold, upper bound only): {results['oracle_best_f1_upper_bound']:.3f}")


if __name__ == "__main__":
    main()
