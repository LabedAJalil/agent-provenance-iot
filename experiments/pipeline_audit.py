import sys, json, collections, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np, pandas as pd
from config import default_paths
from data.provenance import load_provenance
from emulator import Emulator, strip_labels
from units import build_process_units
from features import process_unit_features, build_feature_pipeline, LABEL_LIKE, TIME_LIKE, ID_LIKE
from models.baselines import RandomForestConfig, train_random_forest, predict_proba, oob_proba
from sklearn.linear_model import LogisticRegression
root = Path("..").resolve(); paths = default_paths(root)
ph1, ph2 = load_provenance(paths.provenance_phase1), load_provenance(paths.provenance_phase2)
ph1["label"] = 0; ph1["subLabel"] = ""
e1, e2 = Emulator(provenance=strip_labels(ph1)), Emulator(provenance=strip_labels(ph2))
u1, u2 = build_process_units(ph1, e1, window=5.0), build_process_units(ph2, e2, window=5.0)
print("units: phase1", len(u1), "phase2", len(u2), "attacks", int(u2.is_attack.sum()))

print("\n[CHECK 1] Is the unit's own command line observable at its decision time?")
proc = ph2[(ph2.type == "Process") & ph2.pid.notna() & ph2.event_time.notna()].copy()
proc["pid"] = proc.pid.astype(int); proc["ppid_key"] = proc.ppid.fillna(-1).astype(int)
for c in ("exe", "name"): proc[c] = proc[c].fillna("").astype(str).replace({"nan": ""})
withcmd = proc[proc["command line"].notna()].groupby(["pid", "ppid_key", "exe", "name"])["event_time"].min().rename("first_cmd_time")
m = u2.merge(withcmd, left_on=["pid", "ppid", "exe", "name"], right_index=True, how="left")
has = m["command line"].notna()
late = has & (m.first_cmd_time > m.decision_time)
print(f"   units with a command line: {int(has.sum())}; command line first seen AFTER decision time: {int(late.sum())} ({int((late & (m.is_attack==1)).sum())} attacks)")
if late.any(): print("   max lateness (s):", float((m.first_cmd_time - m.decision_time)[late].max()), "| examples:", m.loc[late, ["pid","name"]].head(3).to_dict("records"))
span = proc.groupby(["pid", "ppid_key", "exe", "name"])["event_time"].agg(["min", "max"])
print(f"   units whose node versions extend beyond start+5s: {int(((span['max']-span['min'])>5).sum())} of {len(span)} (label uses max over versions; features use emulator at t)")

print("\n[CHECK 2] Feature spec: anything label-, time- or id-like?")
spec = process_unit_features(pd.concat([u1, u2]))
used = set(spec.numeric) | set(spec.categorical) | set(spec.text)
print("   forbidden columns used:", sorted(used & (LABEL_LIKE | TIME_LIKE | ID_LIKE)) or "none", "| n numeric", len(spec.numeric), "categorical", spec.categorical, "text", spec.text)

print("\n[CHECK 3] Fold time ordering and train/test attack command-line duplication (view B)")
STAGING_RE = r"/tmp/\.s(/|\b)|/home/erfan/staged|(^|\s)staged/"
u2 = u2.sort_values("start_time").reset_index(drop=True)
order = np.argsort(u2.start_time.to_numpy(), kind="stable"); chunk = np.zeros(len(u2), int)
for k, idx in enumerate(np.array_split(order, 6)): chunk[idx] = k
u2["chunk"] = chunk
stag = (u2.is_attack == 0) & u2["command line"].fillna("").str.contains(STAGING_RE, regex=True)
att = u2[u2.is_attack == 1][["pid", "start_time"]].to_numpy(); pids = set(u2.pid)
def helper(r):
    if r.is_attack == 1 or r["name"] != "base64" or int(r.ppid) in pids: return False
    return bool(((att[:,0] <= r.ppid) & (att[:,0] >= r.ppid - 4) & (np.abs(att[:,1] - r.start_time) < 2)).any())
hel = u2.apply(helper, axis=1)
print("   staging suspects", int(stag.sum()), "helper suspects", int(hel.sum()))
vB = u2[~(stag | hel)].reset_index(drop=True)
import re
norm = lambda c: re.sub(r"\s+", " ", str(c)).strip()
for k in range(1, 6):
    tr, te = vB[vB.chunk < k], vB[vB.chunk == k]
    ok = tr.start_time.max() <= te.start_time.min() if len(te) else True
    tr_cmd = set(tr[tr.is_attack == 1]["command line"].map(norm)); tr_anc = set(tr[tr.is_attack == 1]["ancestor_cmdline"].map(norm))
    ta = te[te.is_attack == 1]
    dup = int(ta["command line"].map(norm).isin(tr_cmd).sum()); dupa = int((ta["ancestor_cmdline"].map(norm).isin(tr_anc) & (ta["ancestor_cmdline"].fillna("") != "")).sum())
    print(f"   fold {k}: train max time <= test min time: {ok} | test attacks {len(ta)}; identical OWN cmdline in training attacks: {dup}; identical ANCESTOR cmdline: {dupa}")

print("\n[CHECK 4] Label-shuffle sanity test (fold 5, view B): a leak-free pipeline must collapse")
def fit_eval(y_tr_override=None, seed=0):
    tr = pd.concat([u1.assign(chunk=-1), vB[vB.chunk < 5]], ignore_index=True); te = vB[vB.chunk == 5]
    pipe = build_feature_pipeline(spec); Xtr = pipe.fit_transform(tr); Xte = pipe.transform(te)
    ytr = tr.is_attack.to_numpy().astype(int) if y_tr_override is None else y_tr_override(tr)
    rf = train_random_forest(Xtr, ytr, RandomForestConfig()); p = predict_proba(rf, Xte); y = te.is_attack.to_numpy().astype(int)
    oob = oob_proba(rf, Xtr); lg = lambda q: np.log(np.clip(q,1e-6,1-1e-6)/(1-np.clip(q,1e-6,1-1e-6)))
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(lg(oob).reshape(-1,1), ytr); pr = lr.predict_proba(lg(p).reshape(-1,1))[:,1]
    out = {}
    for nm, d in (("0.5", p >= .5), ("logistic", pr >= .5)):
        tp, fp, fn = int((d & (y==1)).sum()), int((d & (y==0)).sum()), int((~d & (y==1)).sum())
        out[nm] = (tp, fp, round(2*tp/(2*tp+fp+fn), 3) if tp else 0.0)
    return out, int(ytr.sum()), int(y.sum())
print("   true labels    :", fit_eval())
rng = np.random.default_rng(1)
print("   shuffled labels:", fit_eval(lambda tr: rng.permutation(tr.is_attack.to_numpy().astype(int))))

print("\n[CHECK 5] Recompute pooled forest counts vs archived metrics.json (view B)")
arch = json.load(open("outputs/rolling5_exclude_staging_helpers/metrics.json"))
pred = pd.read_csv("outputs/rolling5_exclude_staging_helpers/test_predictions.csv")
for nm, col in (("rf_raw", "p_rf_raw"), ("rf_rethresholded_logistic", "p_rf_rethr"), ("rf_threshold_oob_f1", "p_rf_thrOOB")):
    d = pred[col] >= .5; y = pred.is_attack == 1
    print(f"   {nm}: recomputed TP/FP {int((d&y).sum())}/{int((d&~y).sum())} | archived {arch[nm]['confusion'][3]}/{arch[nm]['confusion'][1]}")
print("   pooled records match my rebuild of view B chunks 1-5:", len(pred) == int((vB.chunk > 0).sum()), "| attacks", int(pred.is_attack.sum()), "vs", int(vB[vB.chunk>0].is_attack.sum()))
