"""Post hoc tables for the multi-model and sensitivity results. No API call. Writes outputs/posthoc/*.json.

* tool_policy_by_model.json : lookups per record on the enriched sample, per model (distinct lookups, repeats, budget hits)
* sample_by_model.json      : TP / FP / F1, tokens, calls and latency on the enriched sample, per model and condition
* served_models.json        : model identifier returned by the provider, per run and condition, and parse failures
* window_sweep.json         : forest under decision windows of 1, 2, 5, 10 and 30 s (view B)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"
MODELS = {"gpt-4o-mini": "", "qwen3-4b-instruct-2507": "_m-qwen3-4b-instruct-2507", "qwen3-8b": "_m-qwen3-8b",
          "llama-3.1-8b-instruct": "_m-llama-3.1-8b-instruct", "gemma-3-4b-it": "_m-gemma-3-4b-it"}


def prf(y, d):
    tp, fp, fn = int(((d == 1) & (y == 1)).sum()), int(((d == 1) & (y == 0)).sum()), int(((d == 0) & (y == 1)).sum())
    p, r = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
    return {"tp": tp, "fp": fp, "precision": p, "recall": r, "f1": (2 * p * r / (p + r) if p + r else 0.0), "n": int(len(y)), "n_benign": int((y == 0).sum())}


def main() -> None:
    policy, sample, served = {}, {}, {}
    for model, suf in MODELS.items():
        run = OUT / f"rolling5_exclude_staging{suf}"
        sample[model] = {}
        for cond in ("narrative", "context", "agent"):
            f = run / f"llm_{cond}_predictions.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            row = prf(d["y"].to_numpy(), (d["s_llm_raw"] >= 0.5).astype(int).to_numpy())
            row.update({"tok_in": float(d["input_tokens"].mean()), "tok_out": float(d["output_tokens"].mean()), "calls": float(d["n_api_calls"].mean()),
                        "lookups": float(d["n_tool_calls"].mean()), "latency_s": float(d["latency_s"].mean()),
                        "parse_failed": int(d["parse_failed"].sum()) if "parse_failed" in d else 0})
            sample[model][cond] = row
            if cond == "agent":
                tools = d["tools"].fillna("").astype(str).map(lambda s: [t for t in s.split(",") if t])
                policy[model] = {"n": int(len(d)), "calls_histogram": {str(k): int(v) for k, v in sorted(Counter(tools.map(len)).items())},
                                 "distinct_histogram": {str(k): int(v) for k, v in sorted(Counter(tools.map(lambda t: len(set(t)))).items())},
                                 "records_repeating_a_lookup": int(tools.map(lambda t: len(t) > len(set(t))).sum()),
                                 "records_at_budget": int((tools.map(len) >= 6).sum()),
                                 "most_common_sequences": {k: int(v) for k, v in Counter(tools.map(",".join)).most_common(3)}}
        for tag in ("rolling5_exclude_staging", "rolling5_allbenign_exclude_staging"):
            for extra in ("", "_sweep"):
                m = OUT / f"{tag}{suf}{extra}" / "metrics.json"
                if m.exists():
                    man = json.loads(m.read_text())["manifest"]
                    served[f"{tag}{suf}{extra}"] = {"requested": man.get("llm_model"), "backend": man.get("llm_backend"), "argv_no_think": "--llm-no-think" in man.get("argv", []),
                                                    "returned": man.get("model_snapshots"), "parse_failures": man.get("llm_parse_failures"), "errors": man.get("llm_errors")}
    win = {}
    for w, suf in ((1, "_w1"), (2, "_w2"), (5, ""), (10, "_w10"), (30, "_w30")):
        m = json.loads((OUT / f"rolling5_exclude_staging_helpers{suf}" / "metrics.json").read_text())
        win[str(w)] = {k: {"tp": m[k]["confusion"][3], "fp": m[k]["confusion"][1], "f1": m[k]["f1"]} for k in ("rf_raw", "rf_threshold_oob_f1", "rf_rethresholded_logistic", "rule_ancestor_b64_eval") if k in m}
    (OUT / "posthoc").mkdir(exist_ok=True)
    for name, obj in (("tool_policy_by_model", policy), ("sample_by_model", sample), ("served_models", served), ("window_sweep", win)):
        (OUT / "posthoc" / f"{name}.json").write_text(json.dumps(obj, indent=1))
    for mdl, conds in sample.items():
        print(mdl, {c: (r["tp"], r["fp"], round(r["f1"], 3), round(r["tok_in"]), round(r["lookups"], 1), r["parse_failed"]) for c, r in conds.items()})
    for mdl, p in policy.items():
        print(mdl, p["distinct_histogram"], "repeat", p["records_repeating_a_lookup"], "budget", p["records_at_budget"])
    print(json.dumps(win))
    print(json.dumps(served, indent=0)[:3000])


if __name__ == "__main__":
    main()
