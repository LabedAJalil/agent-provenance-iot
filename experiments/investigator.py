"""Investigator condition: the parts that sit around the language model.

The three earlier conditions judge one record in one pass. The investigator adds what they lack:

behaviour  : the record is written as actions (what the process loaded, renamed, deleted, connected to,
             started; who started it) instead of a field dump, with the host baseline next to it.
baseline   : how often this kind of process ran in the attack-free reference period (Phase 1) and earlier in
             the current period. Counts only; no label is read.
memory     : a case notebook holding the investigator's OWN earlier conclusions. Notes are retrieved into
             the next record by lineage (an ancestor or sibling was judged before) and by artefact (a path
             this process names or touches was touched by a process judged suspicious before).
loop       : plan (competing hypotheses) -> lookups -> reflection -> verdict; implemented in llm.py.

Everything is causal: a note or a count is visible to a record only if it was produced at or before the
record's decision time, and records are judged in decision-time order.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from emulator import Emulator

INVESTIGATOR_PROMPT_VERSION = "i2"  # i1 showed the verdict format in the system prompt; Qwen3-8B then answered in the PLAN step

PATH_RE = re.compile(r"(/[^\s'\";|&<>()$]+)")
# Paths that almost every process touches; they carry no information about which processes are related.
GENERIC_PREFIXES = ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc/ld.so", "/proc", "/sys", "/dev", "/run", "/var/lib", "/var/cache", "/snap")

INVESTIGATOR_RULES = (
    "You are a host-intrusion investigator on a Linux industrial-IoT workstation. You receive a CASE FILE about ONE "
    "process: what it did, who started it, how usual it is on this host, and your own earlier notes about related "
    "processes. Decide whether the process is part of an attack campaign (living-off-the-land activity: staging, "
    "credential access, discovery, exfiltration, persistence, defence evasion, lateral movement, C2) or ordinary "
    "system/administrative activity. Attackers use ordinary utilities, so a harmless-looking command can still be one "
    "step of an attack; a scary-looking word can still be routine. Judge what the actions DO and what they could lead "
    "to, not how the command looks. You work in three steps and the user message tells you which step you are in. "
    "PLAN: restate the actions in plain words, then give competing explanations (at least one benign, at least one "
    "malicious naming the attack stage and what it would enable next) and, for each, the observation that would "
    "confirm or refute it. INVESTIGATE: call lookups that separate the explanations; do not call a lookup whose answer "
    "cannot change your verdict. All lookups are evaluated at the decision time; nothing after it is visible; `window` "
    "may be widened (for example 3600) to look further back. REFLECT: for each explanation say supported, refuted or "
    "untested and cite the observation; name what is still unexplained; question your own leaning (what would a "
    "system administrator's reason be, what would an attacker gain). Rare is not malicious and common is not benign "
    "on their own. If evidence is missing say so. You give a verdict only in the REFLECT step, in the format that step "
    "states; a verdict given earlier is discarded."
)

VERDICT_FORMAT = (
    'STRICT JSON, no markdown: {"verdict":"MALICIOUS"|"BENIGN","risk_score":<0..1>,"evidence":"<one or two sentences citing the records used>",'
    '"stage":"<attack stage or none>","leads_to":"<what this action enables next, or none>",'
    '"note":"<one line for your case notebook: what this process did and why it matters to later processes>"}. '
    "risk_score is your probability that the process is malicious."
)

PLAN_PROMPT = ("STEP 1 - PLAN. Plain text, no JSON, no verdict and no risk score yet: you have not investigated. In at most 150 words: (a) the actions in plain words; "
               "(b) explanations H1, H2, ... (benign and malicious) each with the observation that would confirm or refute it; "
               "(c) the lookups you intend to make, most decisive first.")
INVESTIGATE_PROMPT = ("STEP 2 - INVESTIGATE. Make the lookups that separate your explanations (at most {budget} in total). "
                      "When no further lookup could change your verdict, reply with the single word DONE.")
REFLECT_PROMPT = ("STEP 3 - REFLECT. In at most 120 words: for each explanation say supported / refuted / untested with the observation; "
                  "what is still unexplained; what this action could lead to next. {more}"
                  "Then output the verdict on its own line as " + VERDICT_FORMAT)

EXTRA_TOOL_SCHEMAS: List[Dict] = [
    {
        "type": "function",
        "name": "get_path_history",
        "description": "Which processes touched a file or directory (renamed, deleted, loaded, opened it) or named it in their command line, at or before the decision time. "
                       "`path` is an absolute path, a directory prefix, or a bare file name.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
    {
        "type": "function",
        "name": "get_host_baseline",
        "description": "How usual a kind of process is on this host: how often a process with this name (optionally started by this parent name) ran "
                       "in the attack-free reference period and earlier in the current period.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "parent_name": {"type": "string"}}, "required": ["name"]},
    },
    {
        "type": "function",
        "name": "recall_case_notes",
        "description": "Your own earlier conclusions about a pid and the processes it started, from the case notebook.",
        "parameters": {"type": "object", "properties": {"pid": {"type": "integer"}}, "required": ["pid"]},
    },
]


def _is_generic(path: str) -> bool:
    return (not path) or path.count("/") < 2 or path.startswith(GENERIC_PREFIXES)


def paths_in(text: str) -> Set[str]:
    return {p.rstrip("/.,") for p in PATH_RE.findall(text or "") if not _is_generic(p)}


def _shorten(cmd: str, head: int = 220, tail: int = 120) -> str:
    """Keep both ends of a long command line; a launcher's decoder sits after its payload."""
    cmd = cmd or ""
    return cmd if len(cmd) <= head + tail + 5 else f"{cmd[:head]} ... {cmd[-tail:]}"


# --------------------------------------------------------------------------- path history (emulator side)
class PathIndex:
    """'Who touched or named this file before t'. Built from the label-free emulator. The auditing records no plain
    write, so the process that CREATED a file is usually visible only because its command line names the path;
    ``units`` (pid, name, start_time, command line; no label) supplies those mentions."""

    def __init__(self, emulator: Emulator, units: Optional[pd.DataFrame] = None) -> None:
        self.emulator = emulator
        u = units.dropna(subset=["command line"]).sort_values("start_time") if units is not None else None
        self._u_cmd = u["command line"].astype(str).to_numpy() if u is not None else np.array([], dtype=object)
        self._u_pid = u["pid"].to_numpy(dtype=int) if u is not None else np.array([], dtype=int)
        self._u_start = u["start_time"].to_numpy(dtype=float) if u is not None else np.array([], dtype=float)
        by_path: Dict[str, List] = {}
        for pid, ev in emulator._events_by_pid.items():
            for r in ev.records:
                if r["path"]:
                    by_path.setdefault(r["path"], []).append((r["time"], int(pid), r["operation"]))
        self._by_path = {p: sorted(v) for p, v in by_path.items()}
        self._sorted = sorted(self._by_path)
        self._by_base: Dict[str, List[str]] = {}
        for p in self._sorted:
            self._by_base.setdefault(p.rsplit("/", 1)[-1], []).append(p)

    def _match(self, path: str, max_paths: int = 200) -> List[str]:
        path = (path or "").strip()
        if not path:
            return []
        if "/" not in path:
            return self._by_base.get(path, [])[:max_paths]
        path = path.rstrip("/")
        out = [path] if path in self._by_path else []
        lo = bisect.bisect_left(self._sorted, path + "/")
        hi = bisect.bisect_left(self._sorted, path + "/￿")
        return out + self._sorted[lo:hi][: max_paths - len(out)]

    def _mentions(self, path: str, t: float) -> List:
        """Processes started at or before t whose command line, as observed at t, names the path."""
        path = (path or "").strip().rstrip("/")
        if len(path) < 4:
            return []
        out = []
        for i in range(int(np.searchsorted(self._u_start, t, side="right"))):
            if path in self._u_cmd[i]:
                proc = self.emulator.get_process(int(self._u_pid[i]), t)
                self.emulator.query_log.pop()
                if path in (proc.get("command_line") or ""):
                    out.append((float(self._u_start[i]), int(self._u_pid[i]), "named in command line", path))
        return out

    def history(self, path: str, t: float, limit: int = 25) -> List[Dict]:
        recs = [(tm, pid, op, p) for p in self._match(path) for tm, pid, op in self._by_path[p] if tm <= t]
        recs += self._mentions(path, t)
        recs.sort()
        out = []
        for tm, pid, op, p in recs[-limit:]:
            assert tm <= t, "causality guard violated"
            proc = self.emulator.get_process(pid, tm)
            self.emulator.query_log.pop()
            out.append({"time": tm, "seconds_before_decision": round(t - tm, 1), "operation": op, "path": p, "pid": pid,
                        "name": proc.get("name", ""), "command_line": _shorten(proc.get("command_line", "") or "")})
        return out


# --------------------------------------------------------------------------- host baseline
class HostBaseline:
    """Frequency of process kinds. ``reference`` is the attack-free week (Phase 1); ``current`` is the period under
    watch, counted only up to the asking record's decision time. No label column is read."""

    def __init__(self, reference: pd.DataFrame, current: pd.DataFrame) -> None:
        ref_days = max((reference["start_time"].max() - reference["start_time"].min()) / 86400.0, 1e-9) if len(reference) else 1.0
        self._ref_days = float(ref_days)
        self._ref_name = reference["name"].value_counts().to_dict()
        self._ref_sig = (reference["name"].astype(str) + "<-" + reference["parent_name"].astype(str)).value_counts().to_dict()
        self._ref_cmd = reference["command line"].dropna().astype(str).value_counts().to_dict()
        cur = current.sort_values("start_time")
        self._cur_times: Dict[str, np.ndarray] = {}
        for key, g in cur.groupby(cur["name"].astype(str) + "<-" + cur["parent_name"].astype(str), sort=False):
            self._cur_times[key] = g["start_time"].to_numpy(dtype=float)
        for key, g in cur.groupby(cur["name"].astype(str), sort=False):
            self._cur_times["name:" + key] = g["start_time"].to_numpy(dtype=float)

    def _before(self, key: str, t: float) -> int:
        times = self._cur_times.get(key)
        return 0 if times is None else int(np.searchsorted(times, t, side="left"))

    def lookup(self, name: str, parent_name: Optional[str], t_start: float, command_line: Optional[str] = None) -> Dict:
        out = {"name": name, "reference_period_runs": int(self._ref_name.get(name, 0)),
               "earlier_runs_in_current_period": self._before("name:" + str(name), t_start)}
        if parent_name:
            sig = f"{name}<-{parent_name}"
            out.update({"started_by": parent_name, "reference_period_runs_with_this_parent": int(self._ref_sig.get(sig, 0)),
                        "earlier_runs_with_this_parent_in_current_period": self._before(sig, t_start)})
        if command_line:
            out["reference_period_runs_of_this_exact_command"] = int(self._ref_cmd.get(command_line, 0))
        out["reference_period_days"] = round(self._ref_days, 1)
        return out

    def sentence(self, row: pd.Series) -> str:
        cmd = row.get("command line")
        b = self.lookup(str(row.get("name", "")), str(row.get("parent_name", "")), float(row["start_time"]), cmd if isinstance(cmd, str) else None)
        s = (f"'{b['name']}' ran {b['reference_period_runs']} times in the attack-free reference period ({b['reference_period_days']} days), "
             f"{b.get('reference_period_runs_with_this_parent', 0)} of them started by '{b.get('started_by', '?')}'; "
             f"earlier in the current period: {b['earlier_runs_in_current_period']} runs, {b.get('earlier_runs_with_this_parent_in_current_period', 0)} with this parent.")
        if "reference_period_runs_of_this_exact_command" in b:
            s += f" This exact command line ran {b['reference_period_runs_of_this_exact_command']} times in the reference period."
        return s


# --------------------------------------------------------------------------- behaviour record
def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def behaviour_for_unit(row: pd.Series, emulator: Emulator, window: float, baseline: Optional[HostBaseline] = None,
                       mask: bool = False, max_items: int = 10) -> str:
    """The record as actions. Causal w.r.t. decision_time; same evidence source as every other condition."""
    from units import mask_signature

    t = float(row["decision_time"])
    pid = int(row["pid"])
    m = (lambda c: mask_signature(c) if mask else c)
    tree = emulator.get_process_tree(pid, t, depth=6)
    files = emulator.get_file_events(pid, t, window=window)
    socks = emulator.get_socket_events(pid, t, window=window)
    children = emulator.get_children(pid, t, window=window)
    cmd = row.get("command line", "")
    cmd = cmd if isinstance(cmd, str) else ""

    by_op: Dict[str, List[str]] = {}
    n_system = 0
    for f in files:
        if f["operation"] == "load" and _is_generic(f["path"] or ""):
            n_system += 1  # shared libraries and the binary itself
            continue
        by_op.setdefault(f["operation"], []).append(f["path"] or "?")
    verbs = {"load": "loaded (executed or mapped)", "open": "opened", "rename (read)": "renamed/moved away", "rename (write)": "renamed/moved into place",
             "rename": "renamed", "unlink": "deleted", "chmod": "changed permissions of", "update": "updated"}
    actions = []
    for op, paths in by_op.items():
        uniq = list(dict.fromkeys(paths))
        actions.append(f"- {verbs.get(op, op)} {_plural(len(uniq), 'path')}: " + ", ".join(uniq[:max_items]) + (" ..." if len(uniq) > max_items else ""))
    if n_system:
        actions.append(f"- loaded {_plural(n_system, 'system library/binary file')} (routine, not listed)")
    if socks:
        remotes = list(dict.fromkeys(f"{s['remote_address']}:{s['remote_port']}/{s['protocol']}" for s in socks))
        actions.append(f"- network: {_plural(len(socks), 'connection')} to " + ", ".join(remotes[:max_items]))
    else:
        actions.append("- network: no connection recorded")
    if children:
        actions.append(f"- started {_plural(len(children), 'child process')}: " + "; ".join(f"{c['name']} (pid {c['pid']}) '{_shorten(m(c['command_line'] or ''), 120, 60)}'" for c in children[:max_items]))
    else:
        actions.append("- started no child process")

    chain = []
    for i, p in enumerate(tree[1:], start=1):
        if p.get("status") == "OK":
            chain.append(f"  {i}. {p.get('name', '?')} (pid {p['pid']}, {p.get('exe', '?')}) command: '{_shorten(m(p.get('command_line') or ''))}'")
        else:
            chain.append(f"  {i}. {p.get('status')} (pid {p.get('pid')}): this process was never recorded by the auditing, so its identity is unknown")
    parts = [
        f"PROCESS: {row.get('name', '')} (pid {pid}, executable {row.get('exe', '')}, uid {row.get('uid', '')}, euid {row.get('euid', '')})",
        f"COMMAND LINE: {cmd if cmd else '(not recorded)'}",
        f"WHAT IT DID in its first {window:g} s (the auditing records loads, opens, renames, deletions, permission changes and connections; plain reads and writes are NOT recorded, so a copy shows only in the command line):",
        *actions,
        "WHO STARTED IT (nearest first):",
        *(chain or ["  (no parent recorded)"]),
    ]
    if baseline is not None:
        parts += ["HOW USUAL IT IS ON THIS HOST: " + baseline.sentence(row)]
    return "\n".join(parts)


# --------------------------------------------------------------------------- memory
@dataclass
class CaseNotebook:
    """External memory: one note per judged process, written by the investigator, retrieved into later records."""

    notes: List[Dict] = field(default_factory=list)
    _by_pid: Dict[int, List[int]] = field(default_factory=dict)
    _by_ppid: Dict[int, List[int]] = field(default_factory=dict)

    def add(self, *, pid: int, ppid: int, name: str, command_line: str, decision_time: float, verdict: str, score: float,
            stage: str, note: str, paths: Sequence[str]) -> None:
        i = len(self.notes)
        self.notes.append({"pid": int(pid), "ppid": int(ppid), "name": name, "command_line": _shorten(command_line or "", 120, 60), "decision_time": float(decision_time),
                           "verdict": verdict, "score": float(score), "stage": stage, "note": (note or "")[:300], "paths": sorted(set(paths))})
        self._by_pid.setdefault(int(pid), []).append(i)
        self._by_ppid.setdefault(int(ppid), []).append(i)

    def _fmt(self, n: Dict, relation: str, t: float) -> Dict:
        return {"relation": relation, "seconds_before_decision": round(t - n["decision_time"], 1), "pid": n["pid"], "name": n["name"], "command_line": n["command_line"],
                "your_verdict": n["verdict"], "your_risk_score": n["score"], "stage": n["stage"], "note": n["note"]}

    def retrieve(self, *, pid: int, ancestors: Sequence[int], ppid: int, paths: Set[str], t: float, k: int = 8) -> List[Dict]:
        """Notes made at or before ``t``. Ancestors are returned whatever the verdict; siblings and artefact
        matches only when the earlier verdict was suspicious (a benign sibling says little)."""
        out: List[Dict] = []
        seen: Set[int] = set()

        def take(i: int, relation: str) -> None:
            n = self.notes[i]
            if i in seen or n["decision_time"] > t or n["pid"] == pid:
                return
            seen.add(i)
            out.append(self._fmt(n, relation, t))

        for depth, a in enumerate(ancestors, start=1):
            for i in self._by_pid.get(int(a), []):
                take(i, "parent" if depth == 1 else f"ancestor (depth {depth})")
        for i in self._by_ppid.get(int(ppid), []):
            if self.notes[i]["score"] >= 0.5:
                take(i, "sibling (same parent)")
        if paths:
            for i, n in enumerate(self.notes):
                if n["score"] < 0.5 or n["decision_time"] > t:
                    continue
                hit = next((p for p in paths for q in n["paths"] if p == q or p.startswith(q + "/") or q.startswith(p + "/")), None)
                if hit:
                    take(i, f"touched the same path ({hit})")
        out.sort(key=lambda r: (r["your_risk_score"] < 0.5, r["seconds_before_decision"]))
        return out[:k]

    def about(self, pid: int, t: float, k: int = 10) -> List[Dict]:
        idx = self._by_pid.get(int(pid), []) + self._by_ppid.get(int(pid), [])
        return [self._fmt(self.notes[i], "the pid itself" if self.notes[i]["pid"] == int(pid) else "child of the pid", t) for i in idx if self.notes[i]["decision_time"] <= t][:k]


def memory_section(notes: List[Dict]) -> str:
    if not notes:
        return "YOUR CASE NOTEBOOK: no earlier note about a related process."
    lines = ["YOUR CASE NOTEBOOK (your own earlier conclusions; they can be wrong, so weigh them, do not copy them):"]
    for n in notes:
        lines.append(f"- {n['relation']}, {n['seconds_before_decision']:.0f} s earlier: {n['name']} (pid {n['pid']}) '{n['command_line']}' -> "
                     f"{n['your_verdict']} {n['your_risk_score']:.2f}" + (f", stage {n['stage']}" if n["stage"] and n["stage"].lower() != "none" else "") + f". {n['note']}")
    return "\n".join(lines)


def unit_paths(row: pd.Series, emulator: Emulator, window: float) -> Set[str]:
    """Non-generic paths the process names or touches by its decision time (for artefact memory)."""
    t = float(row["decision_time"])
    cmd = row.get("command line", "")
    out = paths_in(cmd if isinstance(cmd, str) else "")
    for f in emulator.get_file_events(int(row["pid"]), t, window=window):
        if f["path"] and not _is_generic(f["path"]):
            out.add(f["path"])
    emulator.query_log.pop()
    return out
