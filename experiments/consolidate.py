"""One aligned table of every detector on the full pooled test set, and the analyses the manuscript quotes.

Reads run directories written by ``run_rolling.py --n-benign -1`` (they must contain ``llm_scores_<cond>.csv``,
which carries the LLM score of every record, training chunks included). No API call, no label use except to score
and, for the benign-behaviour filter, the training labels of each fold (the same labels the forest trains on).

Analyses
* detectors: TP / FP / precision / recall / F1 / FP per 1,000 benign, with a lineage-cluster bootstrap interval
  and an hour-block bootstrap interval (records of one launcher lineage, and cron jobs of one hour, are dependent);
* pairs: clustered and hour-block bootstrap of the F1, recall and FPR difference (primary inference), and an exact
  McNemar test, which is a record-level test of error rates, not of F1, reported as secondary;
* benign-behaviour filter: for fold k and detector D, a process signature (name, parent name) is suppressed when D
  flagged at least ``--min-flag`` benign records with that signature in chunks < k (and the Phase 1 sample) and no
  attack record in chunks < k has it. The list uses pre-test records only and is applied to every detector;
* prevalence: precision and F1 implied by the measured TPR and FPR at other attack prevalences;
* false positives by process name; per-chunk counts (cold start); parse-failure recall bounds.

Usage:
  .venv/bin/python consolidate.py --view A
Writes ``outputs/consolidated/view<A|B>.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from paired_pairwise import lineage_clusters, prf

OUT = Path(__file__).resolve().parent / "outputs"
SLM = {"qwen3-4b": "qwen3-4b-instruct-2507", "qwen3-8b": "qwen3-8b", "llama-3.1-8b": "llama-3.1-8b-instruct", "gemma-3-4b": "gemma-3-4b-it"}
CONDS = ("narrative", "context", "sweep", "agent")


def _dirs(view: str, folds: str = "count") -> Dict[str, List[Path]]:
    suf = "exclude_staging" + ("_helpers" if view == "B" else "")
    pre = "rolling5_time_allbenign_" if folds == "time" else "rolling5_allbenign_"
    d = {"gpt-4o-mini": [OUT / f"{pre}{suf}", OUT / f"{pre}{suf}_sweep"],
         # Same model, launcher wrapper decoded (--mask-signature --mask-level decode): the fingerprint-free view of the campaign.
         "gpt-4o-mini-decoded": [OUT / f"{pre}masked_decode_{suf}"]}
    for short, slug in SLM.items():
        d[short] = [OUT / f"{pre}{suf}_m-{slug}", OUT / f"{pre}{suf}_m-{slug}_sweep"]
    return d


def build(view: str, folds: str = "count"):
    dirs = _dirs(view, folds)
    base = pd.read_csv(dirs["gpt-4o-mini"][0] / "test_predictions.csv").set_index("row_id")
    test = pd.DataFrame(index=base.index)
    test["y"] = base["is_attack"].astype(int)
    for c in ("pid", "ppid", "start_time", "chunk", "name"):
        test[c] = base[c]
    test["tactic"] = base["attack_tactic"].fillna("").astype(str)
    dec = pd.DataFrame(index=base.index)
    dec["forest@0.5"] = (base["p_rf_raw"] >= 0.5).astype(float)
    dec["forest@oobf1"] = (base["p_rf_thrOOB"] >= 0.5).astype(float)
    dec["forest"] = (base["p_rf_rethr"] >= 0.5).astype(float)
    text = (base["command line"].fillna("").astype(str) + " || " + base["ancestor_cmdline"].fillna("").astype(str))
    dec["signature-rule"] = text.str.contains("base64|eval", regex=True).astype(float)
    # Label-free references (label_free.py): Isolation Forest fitted on Phase 1 at fixed clean-host alert rates, and two novelty rules.
    lf = OUT / ("rolling5_exclude_staging" + ("_helpers" if view == "B" else "")) / "label_free_predictions.csv"
    if lf.exists():
        l = pd.read_csv(lf).set_index("row_id")
        dec["iforest@1e-2"] = l["alert_rate_0.01"].astype(float).reindex(dec.index)
        dec["iforest@1e-3"] = l["alert_rate_0.001"].astype(float).reindex(dec.index)
        dec["novel-signature"] = l["novel_signature"].astype(float).reindex(dec.index)
        dec["novel-cmdline"] = l["novel_command_line"].astype(float).reindex(dec.index)
        if "fingerprint_eval" in l.columns:
            # One-line text match on the launcher idiom (eval in own or ancestor command line); no model, no label.
            dec["fingerprint-eval"] = l["fingerprint_eval"].astype(float).reindex(dec.index)
    allscores: Dict[str, pd.DataFrame] = {}
    parse_fail: Dict[str, pd.Series] = {}
    for model, ds in dirs.items():
        for d in ds:
            for cond in CONDS:
                f = d / f"llm_scores_{cond}.csv"
                if not f.exists():
                    continue
                s = pd.read_csv(f).set_index("row_id")
                name = f"{model}/{cond}"
                allscores[name] = s
                dec[name] = (s["score"] >= 0.5).astype(float).reindex(dec.index)
                parse_fail[name] = s["parse_failed"].astype(bool).reindex(dec.index).fillna(False).astype(bool)
                p = d / f"llm_{cond}_predictions.csv"
                if model == "gpt-4o-mini" and p.exists():
                    pp = pd.read_csv(p).set_index("row_id")
                    dec[f"{model}/stack-{cond}"] = (pp["p_stacked"] >= 0.5).astype(float).reindex(dec.index)
    return test, dec, allscores, parse_fail


def _boot(y, cols: List[np.ndarray], groups: np.ndarray, n_boot: int, seed: int, fn):
    rng = np.random.default_rng(seed)
    ids, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(ids))]
    has = np.array([y[m].any() for m in members])
    pos_c, neg_c = np.flatnonzero(has), np.flatnonzero(~has)
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([rng.choice(pos_c, len(pos_c)), rng.choice(neg_c, len(neg_c))]) if len(neg_c) else rng.choice(pos_c, len(pos_c))
        idx = np.concatenate([members[k] for k in pick])
        vals.append(fn(y[idx], [c[idx] for c in cols]))
    v = np.array(vals)
    return [[float(np.percentile(v[:, j], 2.5)), float(np.percentile(v[:, j], 97.5))] for j in range(v.shape[1])]


def detector_row(y, d, cl, hr, n_boot, seed) -> Dict:
    tp, fp, fn, tn, p, r, f1, fpr = prf(y, d)
    one = lambda yy, cc: [prf(yy, cc[0])[6], prf(yy, cc[0])[5], 1000 * prf(yy, cc[0])[7]]
    ci_c = _boot(y, [d], cl, n_boot, seed, one)
    ci_h = _boot(y, [d], hr, n_boot, seed, one)
    return {"n": int(len(y)), "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1, "fp_per_1000": 1000 * fpr,
            "f1_ci_cluster": ci_c[0], "recall_ci_cluster": ci_c[1], "fp_per_1000_ci_cluster": ci_c[2], "f1_ci_hourblock": ci_h[0]}


def pair_row(y, a, b, cl, hr, n_boot, seed) -> Dict:
    diff = lambda yy, cc: [prf(yy, cc[0])[6] - prf(yy, cc[1])[6], prf(yy, cc[0])[5] - prf(yy, cc[1])[5], 1000 * (prf(yy, cc[0])[7] - prf(yy, cc[1])[7])]
    pt = diff(y, [a, b])
    ci_c, ci_h = _boot(y, [a, b], cl, n_boot, seed, diff), _boot(y, [a, b], hr, n_boot, seed, diff)
    ca, cb = a == y, b == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return {"n": int(len(y)), "d_f1": pt[0], "d_f1_ci_cluster": ci_c[0], "d_f1_ci_hourblock": ci_h[0], "d_recall": pt[1], "d_recall_ci_cluster": ci_c[1],
            "d_fp_per_1000": pt[2], "d_fp_per_1000_ci_cluster": ci_c[2],
            "mcnemar_record_level": {"a_right_b_wrong": n01, "b_right_a_wrong": n10, "p": float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0}}


def benign_filter(test: pd.DataFrame, dec: pd.DataFrame, allscores: Dict[str, pd.DataFrame], min_flag: int, source: str = "pretest") -> tuple[pd.DataFrame, Dict]:
    """Per fold, per detector: suppress signatures the detector itself flagged on benign pre-test records.
    source="phase1": the list is built from the attack-free reference period only, so no label is read at all
    (Phase 1 is benign by construction); the same list then applies to every fold."""
    meta = next(iter(allscores.values()))[["phase", "chunk", "is_attack", "name", "parent_name"]].copy()
    meta["sig"] = meta["name"].fillna("").astype(str) + "<-" + meta["parent_name"].fillna("").astype(str)
    sig_test = meta["sig"].reindex(test.index)
    out, info = dec.copy(), {}
    for name in dec.columns:
        # The decoded-mask variant is the same model on the same clean period (Phase 1 holds no launcher, so the mask
        # changes nothing there in principle); it reuses the list its unmasked run learned rather than re-learning one
        # from a re-scored 300-record sample, where a single flipped verdict can drop a signature below min_flag.
        src = name.replace("gpt-4o-mini-decoded/", "gpt-4o-mini/") if name.startswith("gpt-4o-mini-decoded/") else name
        if src in allscores:  # LLM: a score exists for every pre-test record
            hist = meta.join((allscores[src]["score"] >= 0.5).rename("flag"), how="inner")
        else:  # forest / rule: decisions exist for earlier test chunks only
            hist = meta.join(dec[name].rename("flag"), how="inner")
        hist = hist[hist["flag"].notna()]
        lists = {}
        for k in sorted(test["chunk"].unique()):
            if source == "phase1":
                pre = hist[hist["phase"] == 1]
                attack_sigs = set()
                cnt = pre[pre["flag"] == 1]["sig"].value_counts()
            else:
                pre = hist[(hist["chunk"] < k) | (hist["phase"] == 1)]
                # attack signatures come from every pre-test record, also for detectors whose history starts at chunk 1
                attack_sigs = set(meta.loc[((meta["chunk"] < k) | (meta["phase"] == 1)) & (meta["is_attack"] == 1), "sig"])
                cnt = pre[(pre["is_attack"] == 0) & (pre["flag"] == 1)]["sig"].value_counts()
            sup = sorted(s for s, c in cnt.items() if c >= min_flag and s not in attack_sigs)
            lists[int(k)] = sup
            m = (test["chunk"] == k) & sig_test.isin(sup)
            out.loc[m[m].index, name] = out.loc[m[m].index, name] * 0.0
        info[name] = lists
    return out, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", choices=["A", "B"], default="A")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-flag", type=int, default=3)
    ap.add_argument("--folds", choices=["count", "time"], default="count")
    ap.add_argument("--suffix", type=str, default="", help="appended to the output file name (sensitivity runs)")
    ap.add_argument("--filter-source", choices=["pretest", "phase1"], default="pretest", help="phase1: label-free filter from the reference period only")
    args = ap.parse_args()
    test, dec, allscores, parse_fail = build(args.view, args.folds)
    y = test["y"].to_numpy()
    cl = lineage_clusters(test)
    hr = (test["start_time"].to_numpy() // 3600).astype(int)
    filt, filt_lists = benign_filter(test, dec, allscores, args.min_flag, args.filter_source)
    res = {"view": args.view, "folds": args.folds, "min_flag": args.min_flag, "filter_source": args.filter_source, "n": int(len(test)), "n_attack": int(y.sum()), "n_lineage_clusters_attack": int(len(set(cl[y == 1]))),
           "n_hour_blocks": int(len(set(hr))), "detectors": {}, "detectors_filtered": {}, "filter_lists": {}, "pairs": {}, "pairs_filtered": {},
           "fp_by_name": {}, "per_chunk": {}, "parse_failures": {}, "prevalence": {}}
    for name in dec.columns:
        m = dec[name].notna().to_numpy()
        if m.sum() == 0:
            continue
        res["detectors"][name] = detector_row(y[m], dec[name].to_numpy()[m], cl[m], hr[m], args.n_boot, args.seed)
        res["detectors_filtered"][name] = detector_row(y[m], filt[name].to_numpy()[m], cl[m], hr[m], args.n_boot, args.seed)
        res["filter_lists"][name] = {str(k): v for k, v in filt_lists[name].items() if v}
        fpm = m & (dec[name].to_numpy() == 1) & (y == 0)
        vc = test.loc[fpm, "name"].value_counts()
        res["fp_by_name"][name] = {"n_types": int(len(vc)), "top": {str(k): int(v) for k, v in vc.head(6).items()}}
        res["per_chunk"][name] = {int(k): {"n_attack": int(((test["chunk"] == k) & (y == 1) & m).sum()),
                                           "tp": int(((test["chunk"] == k) & (y == 1) & (dec[name] == 1)).sum()),
                                           "fp": int(((test["chunk"] == k) & (y == 0) & (dec[name] == 1)).sum()),
                                           "fp_filtered": int(((test["chunk"] == k) & (y == 0) & (filt[name] == 1)).sum()),
                                           "tp_filtered": int(((test["chunk"] == k) & (y == 1) & (filt[name] == 1)).sum())}
                                  for k in sorted(test["chunk"].unique())}
        if name in parse_fail:
            pf = parse_fail[name].to_numpy() & m
            d_hi = dec[name].to_numpy().copy(); d_hi[pf & (y == 1)] = 1.0  # upper bound: failed attacks counted as detected
            ok = m & ~pf
            res["parse_failures"][name] = {"n": int(pf.sum()), "n_attack": int((pf & (y == 1)).sum()),
                                           "recall_failures_as_no_alert": prf(y[m], dec[name].to_numpy()[m])[5],
                                           "recall_failures_excluded": prf(y[ok], dec[name].to_numpy()[ok])[5],
                                           "recall_failures_as_alert": prf(y[m], d_hi[m])[5]}
        tpr, fpr = res["detectors"][name]["recall"], res["detectors"][name]["fp_per_1000"] / 1000
        res["prevalence"][name] = {str(pi): (lambda P: {"precision": P, "f1": (2 * P * tpr / (P + tpr) if P + tpr else 0.0)})(
            (tpr * pi / (tpr * pi + fpr * (1 - pi))) if (tpr * pi + fpr * (1 - pi)) else 0.0) for pi in (0.276, 0.1, 0.027, 0.01, 0.001)}
    g = "gpt-4o-mini/"
    pairs = [("forest", g + "narrative"), ("forest", g + "context"), ("forest", g + "sweep"), ("forest", g + "agent"), (g + "context", g + "sweep"),
             (g + "sweep", g + "agent"), (g + "context", g + "agent"), ("forest", g + "stack-agent"), ("forest", "signature-rule"),
             ("forest@0.5", g + "agent"), ("forest@oobf1", g + "agent")]
    for short in SLM:
        pairs += [("forest", f"{short}/narrative"), ("forest", f"{short}/context"), ("forest", f"{short}/sweep"), (g + "context", f"{short}/context")]
    for ref in ("iforest@1e-2", "novel-signature", "novel-cmdline", "fingerprint-eval"):
        pairs += [(m + "/" + c, ref) for m in ["gpt-4o-mini", "gpt-4o-mini-decoded"] + list(SLM) for c in ("narrative", "context", "sweep", "agent")] + [("signature-rule", ref), ("forest", ref)]
    pairs += [(g + c, "gpt-4o-mini-decoded/" + c) for c in ("narrative", "context", "agent")]
    for a, b in pairs:
        if a not in dec.columns or b not in dec.columns:
            continue
        m = (dec[a].notna() & dec[b].notna()).to_numpy()
        res["pairs"][f"{a} vs {b}"] = pair_row(y[m], dec[a].to_numpy()[m], dec[b].to_numpy()[m], cl[m], hr[m], args.n_boot, args.seed)
        res["pairs_filtered"][f"{a} vs {b}"] = pair_row(y[m], filt[a].to_numpy()[m], filt[b].to_numpy()[m], cl[m], hr[m], args.n_boot, args.seed)
    (OUT / "consolidated").mkdir(exist_ok=True)
    (OUT / "consolidated" / f"view{args.view}{"_time" if args.folds == "time" else ""}{args.suffix}.json").write_text(json.dumps(res, indent=1))
    print(f"view {args.view}: n={res['n']} attacks={res['n_attack']} attack lineages={res['n_lineage_clusters_attack']} hour blocks={res['n_hour_blocks']}")
    for k, d in res["detectors"].items():
        f = res["detectors_filtered"][k]
        print(f"  {k:28s} {d['tp']:3d}/{d['fp']:4d} F1={d['f1']:.3f} [{d['f1_ci_cluster'][0]:.2f},{d['f1_ci_cluster'][1]:.2f}] FP/k={d['fp_per_1000']:6.1f} | filtered {f['tp']:3d}/{f['fp']:4d} F1={f['f1']:.3f} "
              f"[{f['f1_ci_cluster'][0]:.2f},{f['f1_ci_cluster'][1]:.2f}] | FP types {res['fp_by_name'][k]['n_types']}")
    for k, d in res["pairs"].items():
        f = res["pairs_filtered"][k]
        print(f"  {k:52s} dF1={d['d_f1']:+.3f} clu[{d['d_f1_ci_cluster'][0]:+.3f},{d['d_f1_ci_cluster'][1]:+.3f}] hr[{d['d_f1_ci_hourblock'][0]:+.3f},{d['d_f1_ci_hourblock'][1]:+.3f}]"
              f" | filtered dF1={f['d_f1']:+.3f} clu[{f['d_f1_ci_cluster'][0]:+.3f},{f['d_f1_ci_cluster'][1]:+.3f}]")


if __name__ == "__main__":
    main()
