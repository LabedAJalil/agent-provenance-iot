"""Pairwise-aligned paired comparisons with multiplicity control and a lineage-clustered bootstrap.

``paired_tests.py`` aligns every detector on the units scored by ALL language-model conditions. When one
condition covers fewer units than another (the tool-using arm of the all-benign runs), that discards
units on which the other detectors are fully scored. This script aligns each PAIR on the units both
members scored, so that ``forest vs fixed context`` uses the complete pooled set.

For each requested pair it reports

* exact McNemar test on discordant decisions, plus the Holm-adjusted p-value over the requested family;
* paired stratified bootstrap of the F1 / recall / FPR difference, resampling RECORDS;
* the same bootstrap resampling LINEAGE CLUSTERS (a record's cluster is the top-most recorded ancestor
  reachable through pid/ppid links inside the pooled test set; clusters are stratified by whether they
  contain a labelled attack). Records of one launcher lineage are not independent; this interval is the
  one to quote when the two differ;
* the disjoint-miss table by tactic.

Detectors: rf_0.5, rf_logistic (logistic re-threshold on training OOB scores), rf_oobf1 (threshold that
maximises OOB F1), llm_<cond>, stack_<cond>, union_<cond> (rf_logistic OR llm_<cond>).

Usage:
  .venv/bin/python paired_pairwise.py outputs/rolling5_allbenign_exclude_staging_helpers \
      --pairs rf_logistic:llm_context,rf_logistic:llm_agent,llm_context:llm_agent
Writes ``paired_pairwise.json`` in the run directory. No label is used except to score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import binomtest

CONDS = ("narrative", "context", "agent")


def load(run_dir: Path) -> pd.DataFrame:
    base = pd.read_csv(run_dir / "test_predictions.csv").set_index("row_id")
    out = pd.DataFrame(index=base.index)
    out["y"] = base["is_attack"].astype(int)
    out["tactic"] = base["attack_tactic"].fillna("").astype(str)
    out["pid"], out["ppid"], out["start_time"] = base["pid"].astype(int), base["ppid"].astype(int), base["start_time"].astype(float)
    out["rf_0.5"] = (base["p_rf_raw"] >= 0.5).astype(float)
    out["rf_logistic"] = (base["p_rf_rethr"] >= 0.5).astype(float)
    out["rf_oobf1"] = (base["p_rf_thrOOB"] >= 0.5).astype(float)
    for cond in CONDS:
        f = run_dir / f"llm_{cond}_predictions.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f).set_index("row_id")
        d = d[d.index.isin(out.index)]
        out[f"llm_{cond}"] = (d["s_llm_raw"] >= 0.5).astype(float).reindex(out.index)
        out[f"stack_{cond}"] = (d["p_stacked"] >= 0.5).astype(float).reindex(out.index)
        u = out[f"llm_{cond}"].copy()
        m = u.notna()
        u[m] = ((u[m] == 1) | (out.loc[m, "rf_logistic"] == 1)).astype(float)
        out[f"union_{cond}"] = u
    return out


def lineage_clusters(df: pd.DataFrame) -> np.ndarray:
    """Cluster id = row position of the top-most recorded ancestor inside the pooled set.

    The parent of a record is the record with pid == ppid and the latest start_time not after the
    child's (pids are reused over 72 h). Records whose parent is not in the set are their own root."""
    by_pid: Dict[int, List[int]] = {}
    st = df["start_time"].to_numpy()
    pid = df["pid"].to_numpy()
    ppid = df["ppid"].to_numpy()
    for i in np.argsort(st, kind="stable"):
        by_pid.setdefault(int(pid[i]), []).append(int(i))
    parent = np.full(len(df), -1)
    for i in range(len(df)):
        cands = [j for j in by_pid.get(int(ppid[i]), []) if st[j] <= st[i] and j != i]
        if cands:
            parent[i] = cands[-1]
    root = np.arange(len(df))
    for i in range(len(df)):
        j, hops = i, 0
        while parent[j] >= 0 and hops < 64:
            j, hops = parent[j], hops + 1
        root[i] = j
    return root


def prf(y: np.ndarray, d: np.ndarray):
    tp = int(((d == 1) & (y == 1)).sum()); fp = int(((d == 1) & (y == 0)).sum())
    fn = int(((d == 0) & (y == 1)).sum()); tn = int(((d == 0) & (y == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return tp, fp, fn, tn, p, r, f1, fpr


def _summ(v: List[float]) -> Dict:
    return {"lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5)), "mean": float(np.mean(v))}


def boot_records(a, b, y, n_boot, rng) -> Dict:
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    out = {"f1": [], "recall": [], "fpr": []}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        ma, mb = prf(y[idx], a[idx]), prf(y[idx], b[idx])
        out["f1"].append(ma[6] - mb[6]); out["recall"].append(ma[5] - mb[5]); out["fpr"].append(ma[7] - mb[7])
    return {k: _summ(v) for k, v in out.items()}


def boot_clusters(a, b, y, cl, n_boot, rng) -> Dict:
    ids, inv = np.unique(cl, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(ids))]
    has_attack = np.array([y[m].any() for m in members])
    pos_c, neg_c = np.flatnonzero(has_attack), np.flatnonzero(~has_attack)
    out = {"f1": [], "recall": [], "fpr": []}
    for _ in range(n_boot):
        pick = np.concatenate([rng.choice(pos_c, len(pos_c)), rng.choice(neg_c, len(neg_c))]) if len(neg_c) else rng.choice(pos_c, len(pos_c))
        idx = np.concatenate([members[k] for k in pick])
        ma, mb = prf(y[idx], a[idx]), prf(y[idx], b[idx])
        out["f1"].append(ma[6] - mb[6]); out["recall"].append(ma[5] - mb[5]); out["fpr"].append(ma[7] - mb[7])
    res = {k: _summ(v) for k, v in out.items()}
    res["n_clusters"] = int(len(ids)); res["n_attack_clusters"] = int(len(pos_c)); res["n_benign_clusters"] = int(len(neg_c))
    return res


def holm(ps: List[float]) -> List[float]:
    order = np.argsort(ps)
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = float(min(1.0, running))
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str)
    ap.add_argument("--pairs", type=str, required=True, help="comma-separated a:b pairs; this list is the Holm family")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="paired_pairwise.json")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    df = load(run_dir)
    cl_all = lineage_clusters(df)
    pairs = [tuple(p.split(":")) for p in args.pairs.split(",") if p]
    out = {"run": run_dir.name, "n_pooled": int(len(df)), "n_boot": args.n_boot, "alignment": "pairwise (units scored by both members)",
           "cluster_definition": "top-most recorded ancestor within the pooled test set via pid/ppid links", "detectors": {}, "pairs": []}
    for name in sorted({d for p in pairs for d in p}):
        m = df[name].notna().to_numpy()
        tp, fp, fn, tn, p, r, f1, fpr = prf(df["y"].to_numpy()[m], df[name].to_numpy()[m])
        fp_rows = df[m & (df[name] == 1).to_numpy() & (df["y"] == 0).to_numpy()]
        out["detectors"][name] = {"n": int(m.sum()), "n_benign": int((df["y"].to_numpy()[m] == 0).sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                                  "precision": p, "recall": r, "f1": f1, "fp_per_1000_benign": 1000.0 * fpr,
                                  "fp_lineage_clusters": int(len(set(cl_all[np.flatnonzero(m & (df[name] == 1).to_numpy() & (df['y'] == 0).to_numpy())]))) if len(fp_rows) else 0}
    for a, b in pairs:
        m = (df[a].notna() & df[b].notna()).to_numpy()
        y, da, db, cl = df["y"].to_numpy()[m], df[a].to_numpy()[m], df[b].to_numpy()[m], cl_all[m]
        tac = df["tactic"].to_numpy()[m]
        ca, cb = da == y, db == y
        n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        p = float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0
        pos = y == 1
        tab = lambda mm: {str(t): int(v) for t, v in pd.Series(tac[mm]).value_counts().items()}
        only_a, only_b, both = pos & (da == 1) & (db == 0), pos & (db == 1) & (da == 0), pos & (da == 0) & (db == 0)
        ma, mb = prf(y, da), prf(y, db)
        out["pairs"].append({"a": a, "b": b, "n": int(m.sum()), "n_attack": int(pos.sum()), "n_benign": int((~pos).sum()),
                             "a_tp_fp_f1": [ma[0], ma[1], ma[6]], "b_tp_fp_f1": [mb[0], mb[1], mb[6]],
                             "mcnemar": {"a_right_b_wrong": n01, "b_right_a_wrong": n10, "p_value": p},
                             "diff_a_minus_b_point": {"f1": ma[6] - mb[6], "recall": ma[5] - mb[5], "fpr": ma[7] - mb[7]},
                             "diff_a_minus_b_record_bootstrap": boot_records(da, db, y, args.n_boot, np.random.default_rng(args.seed)),
                             "diff_a_minus_b_cluster_bootstrap": boot_clusters(da, db, y, cl, args.n_boot, np.random.default_rng(args.seed)),
                             "misses": {"only_a": int(only_a.sum()), "only_b": int(only_b.sum()), "both_miss": int(both.sum()),
                                        "only_a_by_tactic": tab(only_a), "only_b_by_tactic": tab(only_b), "both_miss_by_tactic": tab(both)}})
    for rec, hp in zip(out["pairs"], holm([r["mcnemar"]["p_value"] for r in out["pairs"]])):
        rec["mcnemar"]["p_holm"] = hp
    (run_dir / args.out).write_text(json.dumps(out, indent=1))

    print(f"[{run_dir.name}] pooled n={len(df)}; Holm family of {len(pairs)}")
    for k, d in out["detectors"].items():
        print(f"  {k:16s} n={d['n']:5d} TP={d['tp']:3d} FP={d['fp']:4d} P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f} FP/1000={d['fp_per_1000_benign']:.1f} FP-lineages={d['fp_lineage_clusters']}")
    for r in out["pairs"]:
        m, pt, rb, cb = r["mcnemar"], r["diff_a_minus_b_point"], r["diff_a_minus_b_record_bootstrap"], r["diff_a_minus_b_cluster_bootstrap"]
        print(f"  {r['a']} vs {r['b']} (n={r['n']}): {m['a_right_b_wrong']}/{m['b_right_a_wrong']} p={m['p_value']:.2e} holm={m['p_holm']:.2e} | "
              f"dF1={pt['f1']:+.3f} rec[{rb['f1']['lo']:+.3f},{rb['f1']['hi']:+.3f}] clu[{cb['f1']['lo']:+.3f},{cb['f1']['hi']:+.3f}] | "
              f"dR={pt['recall']:+.3f} clu[{cb['recall']['lo']:+.3f},{cb['recall']['hi']:+.3f}] | dFPR={pt['fpr']:+.4f} clu[{cb['fpr']['lo']:+.4f},{cb['fpr']['hi']:+.4f}] | "
              f"clusters {cb['n_attack_clusters']}+{cb['n_benign_clusters']} | misses {r['misses']['only_a']}/{r['misses']['only_b']}/{r['misses']['both_miss']}")


if __name__ == "__main__":
    main()
