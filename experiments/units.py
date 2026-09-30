"""Prediction units for the provenance data.

Row unit (legacy): every CSV row (node or edge) is one example. This is
internally inconsistent: an attack process's ``Process`` row is labelled 1
while its ``Used``/``WasGeneratedBy``/``WasTriggeredBy`` edges are labelled 0,
and ``Artifact`` rows have no timestamp and are dropped before the split.

Process unit: one example per observed process (merged node versions).
Decision time is ``start + window``; every feature is computed from evidence
at or before the decision time through the label-free emulator.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

import json

import numpy as np
import pandas as pd

from emulator import Emulator

_B64_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9+/=])([A-Za-z0-9+/]{24,}={0,2})(?![A-Za-z0-9+/=])")


def looks_like_base64_blob(text: str) -> bool:
    """A long base64-looking token: mixed case + digits, not a filesystem path.
    (The previous regex accepted '/', so every long path matched.)"""
    for tok in _B64_TOKEN_RE.findall(text or ""):
        if tok.startswith("/") or tok.count("/") > 2 or "//" in tok:
            continue
        if any(ch.isupper() for ch in tok) and any(ch.islower() for ch in tok) and any(ch.isdigit() for ch in tok):
            return True
    return False


_SIGNATURE_WORDS_RE = re.compile(r"\b(base64|eval|decode)\b|--decode", re.IGNORECASE)

# Caldera launcher wrapper: sh -c eval "$(echo <base64> | base64 --decode)". Group 1 is the payload.
_LAUNCHER_RE = re.compile(r'eval\s+"\$\(echo\s+([A-Za-z0-9+/=]+)\s*\|\s*base64\s+(?:--decode|-d)\)"')

# "tokens": delete base64/eval/decode words and base64 blobs (leaves the wrapper shape sh -c "$(echo | )").
# "decode": first replace the whole wrapper by its decoded payload, so the launcher reads as the command
# the attacker actually ran (sh -c cp ... /home/erfan/staged), then apply the token deletion. Set by
# run_rolling.py --mask-level.
MASK_LEVEL = "tokens"


def decode_launcher(text: str) -> str:
    """Replace every Caldera launcher wrapper in ``text`` by its decoded payload."""
    def _sub(m: "re.Match[str]") -> str:
        import base64
        try:
            return base64.b64decode(m.group(1)).decode("utf-8", errors="replace").strip()
        except Exception:
            return ""
    return _LAUNCHER_RE.sub(_sub, text)


def mask_signature(text: str) -> str:
    """Remove the Caldera execution signature from a command line: long base64-like
    blobs and the words base64/eval/decode. Tokens are deleted, not replaced, so
    no placeholder can act as a new signature. Used by --mask-signature.
    With MASK_LEVEL == "decode" the launcher wrapper is decoded first (see above)."""
    if not text:
        return text
    if MASK_LEVEL == "decode":
        text = decode_launcher(text)
    out = []
    for tok in text.split():
        core = tok.strip("\"'()$")
        if looks_like_base64_blob(core):
            continue
        out.append(tok)
    text = " ".join(out)
    return _SIGNATURE_WORDS_RE.sub("", text)


def _cmd_features(cmd: str) -> Dict[str, float]:
    c = cmd or ""
    return {
        "cmd_len": float(len(c)),
        "cmd_n_tokens": float(len(c.split())),
        "cmd_has_base64": float(looks_like_base64_blob(c) or "base64" in c),
        "cmd_has_pipe": float("|" in c),
        "cmd_has_eval": float("eval" in c),
        "cmd_has_http": float("http://" in c or "https://" in c),
        "cmd_has_redirect": float(">" in c or "<" in c),
    }


def _first_valid(s: pd.Series):
    s = s.dropna()
    return s.iloc[0] if len(s) else np.nan


def build_process_units(
    provenance: pd.DataFrame,
    emulator: Emulator,
    window: float,
    label_col: Optional[str] = "label",
    tactic_col: Optional[str] = "subLabel",
    ancestry_depth: int = 5,
    mask: bool = False,
) -> pd.DataFrame:
    """One row per process. ``provenance`` may carry labels; ``emulator`` must not."""
    proc = provenance[provenance["type"] == "Process"].copy()
    proc = proc.dropna(subset=["pid", "event_time"])
    proc["pid"] = proc["pid"].astype(int)
    for c in ("exe", "name"):
        proc[c] = proc[c].fillna("").astype(str).replace({"nan": ""})
    proc["ppid_key"] = proc["ppid"].fillna(-1).astype(int)

    agg = {
        "event_time": "min",
        "command line": _first_valid,
        "uid": _first_valid,
        "euid": _first_valid,
        "gid": _first_valid,
        "egid": _first_valid,
    }
    if label_col and label_col in proc.columns:
        proc[label_col] = pd.to_numeric(proc[label_col], errors="coerce").fillna(0).astype(int)
        agg[label_col] = "max"
    if tactic_col and tactic_col in proc.columns:
        agg[tactic_col] = lambda s: _first_valid(s.where(s.astype(str).str.len() > 0).where(s.astype(str) != "0"))

    units = proc.groupby(["pid", "ppid_key", "exe", "name"], sort=False).agg(agg).reset_index()
    units = units.rename(columns={"ppid_key": "ppid", "event_time": "start_time"})
    if mask:
        units["command line"] = units["command line"].map(lambda c: mask_signature(c) if isinstance(c, str) else c)
    units["decision_time"] = units["start_time"] + float(window)
    units["is_attack"] = units[label_col].astype(int) if label_col and label_col in units.columns else 0
    if tactic_col and tactic_col in units.columns:
        units["attack_tactic"] = units[tactic_col].where(units["is_attack"] == 1, "").fillna("").astype(str)
    else:
        units["attack_tactic"] = ""

    cmd_feats = pd.DataFrame([_cmd_features(c if isinstance(c, str) else "") for c in units["command line"]])
    units = pd.concat([units.reset_index(drop=True), cmd_feats], axis=1)

    parent_exe, parent_name, grand_exe, anc_text, anc_rows, ctx_rows = [], [], [], [], [], []
    for pid, t in zip(units["pid"], units["decision_time"]):
        # Ancestry is retrieved causally (nodes observed at or before the decision time).
        tree = emulator.get_process_tree(int(pid), float(t), depth=ancestry_depth + 1)
        ancestors = tree[1:]  # tree[0] is the process itself
        ok = [a for a in ancestors if a.get("status") == "OK"]
        parent = ancestors[0] if ancestors else {"status": "UNKNOWN"}
        parent_exe.append(parent.get("exe", "") if parent.get("status") == "OK" else parent.get("status", "UNKNOWN"))
        parent_name.append(parent.get("name", "") if parent.get("status") == "OK" else parent.get("status", "UNKNOWN"))
        grand = ancestors[1] if len(ancestors) > 1 else {"status": "UNKNOWN"}
        grand_exe.append(grand.get("exe", "") if grand.get("status") == "OK" else grand.get("status", "UNKNOWN"))
        cmds = [mask_signature(a.get("command_line") or "") if mask else (a.get("command_line") or "") for a in ok]
        joined = " || ".join(c for c in cmds if c)
        anc_text.append(joined)
        anc_rows.append(
            {
                "anc_depth_ok": float(len(ok)),
                "anc_root_unknown": float(bool(ancestors) and ancestors[-1].get("status") != "OK"),
                "anc_n_shell": float(sum(1 for a in ok if (a.get("name") or "") in ("sh", "bash", "dash"))),
                "anc_n_with_cmd": float(sum(1 for c in cmds if c)),
                **{f"anc_{k[4:]}": v for k, v in _cmd_features(joined).items() if k != "cmd_len" and k != "cmd_n_tokens"},
                "anc_cmd_len_max": float(max((len(c) for c in cmds), default=0)),
            }
        )
        ctx_rows.append(emulator.context_features(int(pid), float(t), window=float(window)))
    emulator.query_log.clear()  # feature construction is not an LLM tool call
    units["parent_exe"] = parent_exe
    units["parent_name"] = parent_name
    units["grandparent_exe"] = grand_exe
    units["ancestor_cmdline"] = anc_text
    units = pd.concat([units, pd.DataFrame(anc_rows), pd.DataFrame(ctx_rows)], axis=1)
    return units.sort_values("start_time").reset_index(drop=True)


def narrative_only_for_unit(row: pd.Series) -> str:
    """Process attributes + own command line. No retrieval of any kind."""
    cmd = row.get("command line", "")
    cmd = cmd if isinstance(cmd, str) else ""
    return "\n".join([
        f"process: pid={int(row['pid'])} ppid={int(row.get('ppid', -1))} exe={row.get('exe','')} name={row.get('name','')} uid={row.get('uid','')} euid={row.get('euid','')}",
        f"command_line: {cmd}",
        f"observed_at: start+{float(row['decision_time']) - float(row['start_time']):.0f}s",
    ])


def narrative_for_unit(row: pd.Series, emulator: Emulator, window: float, max_items: int = 8, mask: bool = False) -> str:
    """Fixed-context narrative (no model-directed tool use). Causal w.r.t. decision_time."""
    t = float(row["decision_time"])
    pid = int(row["pid"])
    tree = emulator.get_process_tree(pid, t, depth=4)
    if mask:
        for p in tree:
            if p.get("command_line"):
                p["command_line"] = mask_signature(p["command_line"])
    files = emulator.get_file_events(pid, t, window=window)
    socks = emulator.get_socket_events(pid, t, window=window)
    children = emulator.get_children(pid, t, window=window)
    if mask:
        for c in children:
            if c.get("command_line"):
                c["command_line"] = mask_signature(c["command_line"])
    parts = [
        f"process: pid={pid} exe={row.get('exe','')} name={row.get('name','')} uid={row.get('uid','')}",
        f"command_line: {row.get('command line','') if isinstance(row.get('command line',''), str) else ''}",
        "ancestry: " + " <- ".join(
            f"{p.get('name','?')}[{p.get('exe','?')}] cmd='{(p.get('command_line') or '')[:120]}'" if p.get("status") == "OK" else f"{p.get('status')}"
            for p in tree
        ),
        f"file_events({len(files)}): " + "; ".join(f"{f['operation']} {f['path']}" for f in files[:max_items]),
        f"socket_events({len(socks)}): " + "; ".join(f"{s['operation']} {s['remote_address']}:{s['remote_port']}/{s['protocol']}" for s in socks[:max_items]),
        f"children({len(children)}): " + "; ".join(f"{c['name']} '{(c['command_line'] or '')[:80]}'" for c in children[:max_items]),
    ]
    return "\n".join(parts)


def sweep_for_unit(row: pd.Series, emulator: Emulator, window: float, depth: int = 6, max_items: int = 25, max_chars: int = 6000, mask: bool = False) -> str:
    """Full-context record (deterministic sweep). The narrative-only record plus the verbatim result of the four
    lookups for the process itself, with the limits the tool-using condition has (25 records, 6000 characters per
    lookup) and the ancestry depth the forest features use (the process + 5 ancestors). No command line is cut to
    120 characters. Causal w.r.t. decision_time."""
    t = float(row["decision_time"])
    pid = int(row["pid"])
    tree = emulator.get_process_tree(pid, t, depth=depth)
    files = emulator.get_file_events(pid, t, window=window)[:max_items]
    socks = emulator.get_socket_events(pid, t, window=window)[:max_items]
    children = emulator.get_children(pid, t, window=window)[:max_items]
    if mask:
        for p in list(tree) + list(children):
            if isinstance(p, dict) and p.get("command_line"):
                p["command_line"] = mask_signature(p["command_line"])
    dump = lambda o: json.dumps(o, default=str)[:max_chars]
    return "\n".join([
        narrative_only_for_unit(row),
        f"get_process_tree(pid={pid}, depth={depth}): {dump(tree)}",
        f"get_file_events(pid={pid}, window={window:g}): {dump(files)}",
        f"get_socket_events(pid={pid}, window={window:g}): {dump(socks)}",
        f"get_children(pid={pid}, window={window:g}): {dump(children)}",
    ])
