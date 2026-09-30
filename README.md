# agent-provenance-iot

Code for zero-shot large and small language model detectors, label-free reference detectors and a supervised
random forest on the host provenance logs of CIC-APT-IIoT-2024, evaluated with time-consistent evidence replay
and label audits.

Naming: the condition called `agent` in code and file names is the tool-using LLM level (the model chooses
lookups in a bounded loop; no memory, planning or reflection). `context` is the fixed-context level, `sweep`
the full-context level, `narrative` the record alone.

```
Dataset/                 place CIC-APT-IIoT-2024 here (not redistributed, see Dataset/README.md)
experiments/
  emulator.py            evidence-replay layer: every lookup returns only what existed at decision time
  units.py, features.py  process records, narratives, forest features
  run_rolling.py         rolling-fold evaluation: forest, rules, LLM conditions (narrative, context, sweep, agent)
  llm.py, investigator.py  LLM scoring, tool loop, response cache
  label_free.py          Isolation Forest fitted on the attack-free Phase 1, novelty rules
  consolidate.py         aligned table of all detectors and models, paired differences, benign-behaviour filter
  posthoc_models.py, posthoc_tables.py, truncation_check.py, flip_rate.py, pipeline_audit.py   post hoc checks
  tests/test_emulator.py causality and isolation tests for the replay layer
  outputs/               the result files cited in the article (metrics and summary JSON only)
```

## Setup

```bash
python3 -m venv experiments/.venv            # Python 3.11-3.13
experiments/.venv/bin/pip install -r requirements.txt
experiments/.venv/bin/python experiments/tests/test_emulator.py
```

Keep scikit-learn at 1.8.x: 1.9.x returns out-of-bag probability 0 for the minority class with
`class_weight=balanced`, which silently breaks the calibration and re-threshold steps.

Download the dataset as described in [Dataset/README.md](Dataset/README.md).

For LLM runs copy `experiments/.env.example` to `experiments/.env` and set `OPENAI_API_KEY` (gpt-4o-mini) and
`HF_TOKEN` (open models through the Hugging Face router, `--llm-backend chat`). Responses are cached in
`experiments/outputs/llm_cache_v3.sqlite`, created on the first run; reruns with `--cache-only` make no API calls.

## Reproducing the results

Run from `experiments/`. Label views: `--relabel exclude_staging` is view A, `exclude_staging_helpers` is view B.

```bash
cd experiments
PY=.venv/bin/python

# 1. forest, rules and gpt-4o-mini on the enriched sample (view A) and forest only (view B)
$PY run_rolling.py --n-folds 5 --llm openai --relabel exclude_staging
$PY run_rolling.py --n-folds 5 --llm none   --relabel exclude_staging_helpers

# 2. every record scored (needed by consolidate.py), gpt-4o-mini
for V in exclude_staging exclude_staging_helpers; do
  $PY run_rolling.py --n-folds 5 --llm openai --conditions narrative,context,agent --relabel $V --n-benign -1
  $PY run_rolling.py --n-folds 5 --llm openai --conditions sweep --relabel $V --n-benign -1 --tag sweep
done

# 3. every record scored, open models (repeat with --relabel exclude_staging_helpers for view B)
for M in Qwen/Qwen3-8B:nscale Qwen/Qwen3-4B-Instruct-2507:nscale \
         meta-llama/Llama-3.1-8B-Instruct:deepinfra google/gemma-3-4b-it:deepinfra; do
  $PY run_rolling.py --n-folds 5 --llm openai --llm-backend chat --llm-model $M \
      --conditions narrative,context --relabel exclude_staging --n-benign -1
  $PY run_rolling.py --n-folds 5 --llm openai --llm-backend chat --llm-model $M \
      --conditions sweep --relabel exclude_staging --n-benign -1 --tag sweep
done
# add --llm-no-think for Qwen3-8B

# 4. signature masking (launcher wrapper removed or decoded)
$PY run_rolling.py --n-folds 5 --llm openai --relabel exclude_staging --mask-signature
$PY run_rolling.py --n-folds 5 --llm openai --relabel exclude_staging --mask-signature --mask-level decode
$PY run_rolling.py --n-folds 5 --llm openai --llm-backend chat --llm-model Qwen/Qwen3-8B:nscale --llm-no-think \
    --relabel exclude_staging --mask-signature --mask-level decode
$PY run_rolling.py --n-folds 5 --llm openai --conditions narrative,context,agent --relabel exclude_staging \
    --n-benign -1 --mask-signature --mask-level decode

# 5. label-free references
$PY label_free.py outputs/rolling5_exclude_staging
$PY label_free.py outputs/rolling5_exclude_staging --contaminated
$PY label_free.py outputs/rolling5_exclude_staging --features full
$PY label_free.py outputs/rolling5_exclude_staging_helpers

# 6. consolidated tables
$PY consolidate.py --view A --filter-source phase1 --suffix _labelfree
$PY consolidate.py --view B --filter-source phase1 --suffix _labelfree
$PY consolidate.py --view A

# 7. post hoc checks
$PY posthoc_models.py
$PY posthoc_tables.py
$PY truncation_check.py
$PY flip_rate.py outputs/rolling5_exclude_staging --conditions context --reps 3   # repeatability, bypasses the cache
$PY pipeline_audit.py > outputs/posthoc/pipeline_audit.txt
```

Every run writes `metrics.json` (with a manifest: git revision, argv, dataset md5, fold boundaries, feature
specification), prediction tables and plots into `outputs/<run>/`, and archives a copy in `outputs/runs/`.
Forest, rule and label-free numbers reproduce exactly; LLM numbers depend on the served model version.

## Released results

`experiments/outputs/` holds the files the article cites:

| File | Content |
|---|---|
| `consolidated/viewA_labelfree.json`, `viewB_labelfree.json` | main and filter tables, paired differences, false positives by process type |
| `consolidated/viewA.json` | longer-history benign-behaviour filter |
| `rolling5_exclude_staging/label_free*.json`, `rolling5_exclude_staging_helpers/label_free.json` | Isolation Forest and novelty rules |
| `rolling5_*/metrics.json` | per-run metrics for the enriched-sample and masking tables |
| `rolling5_exclude_staging/flip_rate.json` | repeatability |
| `posthoc/*.json`, `posthoc/pipeline_audit.txt` | served models, lookup policy, truncation, label checks, pipeline audit |

Prediction tables and the response cache are not included because they contain command lines copied from
the dataset; the commands above regenerate them.

## License

Code: MIT (see `LICENSE`). The dataset is subject to the Canadian Institute for Cybersecurity's terms.
