"""How much launcher evidence the fixed-context summary cuts away. Writes outputs/posthoc/truncation.json.
Compares, for every pooled test record (view A), the fixed-context text with the full-context text."""
import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
from config import default_paths
from data.provenance import load_provenance
from emulator import Emulator, strip_labels
from units import build_process_units, narrative_for_unit, sweep_for_unit

root = Path(__file__).resolve().parents[1]; paths = default_paths(root)
ph2 = load_provenance(paths.provenance_phase2)
emu = Emulator(provenance=strip_labels(ph2))
u = build_process_units(ph2, emu, window=5.0)
u["row_id"] = [f"p2:{pid}:{st:.3f}:{pp}:{e}:{n}" for pid, st, pp, e, n in zip(u["pid"], u["start_time"], u["ppid"], u["exe"], u["name"])]
test = pd.read_csv(paths.outputs / "rolling5_allbenign_exclude_staging" / "test_predictions.csv")
u = u[u["row_id"].isin(set(test["row_id"]))]
rows = []
for _, r in u.iterrows():
    fixed, full = narrative_for_unit(r, emu, 5.0), sweep_for_unit(r, emu, 5.0)
    tree = emu.get_process_tree(int(r["pid"]), float(r["decision_time"]), depth=4)[1:]
    cut = any(len(p.get("command_line") or "") > 120 for p in tree if p.get("status") == "OK")
    rows.append({"attack": int(r["is_attack"]), "cut": cut, "b64_fixed": "base64" in fixed, "b64_full": "base64" in full})
d = pd.DataFrame(rows); a = d[d.attack == 1]
out = {"pooled_records": int(len(d)), "records_with_a_cut_ancestor_line": int(d.cut.sum()), "attacks_with_a_cut_ancestor_line": int(a.cut.sum()),
       "test_attacks": int(len(a)), "attacks_base64_visible_in_full_context": int(a.b64_full.sum()), "attacks_base64_visible_in_fixed_context": int(a.b64_fixed.sum()),
       "attacks_base64_lost_by_the_cut": int((a.b64_full & ~a.b64_fixed).sum())}
(paths.outputs / "posthoc" / "truncation.json").write_text(json.dumps(out, indent=1)); print(out)
