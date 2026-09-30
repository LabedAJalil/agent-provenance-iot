"""Causal log-retrieval emulator over the CIC-APT-IIoT-2024 provenance graph.

The provenance CSV is a graph, not a flat event table:

* ``Process`` nodes carry pid/ppid/exe/name/command line and a timestamp
  (``seen time`` or ``start time``).
* ``Artifact`` nodes carry file attributes (path, permissions) or socket
  attributes (remote address/port, protocol). They have NO pid and NO time.
* Edges carry ``time`` and ``operation`` and link node ids:
  ``Used``            Process  -> Artifact   (load, open, rename(read))
  ``WasGeneratedBy``  Artifact -> Process    (unlink, connect, rename(write), chmod)
  ``WasTriggeredBy``  Process  -> Process    (execve, setuid, setgid)
  ``WasDerivedFrom``  Artifact -> Artifact   (rename; carries pid directly)

So a file or socket lookup for a pid requires joining edges to artifacts.
The previous emulator searched pid-grouped rows for artifact columns and
therefore always returned empty lists.

Guarantees enforced here:

1. Causality: every returned record has ``time <= t``. Lookbacks are
   ``[t - window, t]``, never ``t + window``.
2. Source isolation: the emulator is built from a frame that has had every
   label column removed, and asserts this at construction.
3. Honest gaps: missing data returns an explicit sentinel, never an imputed
   value.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

LABEL_COLUMNS = ("label", "subLabel", "is_attack", "attack_tactic", "attack_technique")

UNKNOWN = "UNKNOWN"
NOT_YET_SEEN = "NOT_YET_SEEN"


def strip_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with every known label column removed."""
    return df.drop(columns=[c for c in LABEL_COLUMNS if c in df.columns])


@dataclass
class _PidEvents:
    """Time-sorted edge events attributed to one pid."""

    times: np.ndarray
    records: List[Dict]


@dataclass
class Emulator:
    provenance: pd.DataFrame
    flows: Optional[pd.DataFrame] = None
    _proc_by_pid: Dict[int, pd.DataFrame] = field(default_factory=dict, init=False, repr=False)
    _children_by_ppid: Dict[int, pd.DataFrame] = field(default_factory=dict, init=False, repr=False)
    _events_by_pid: Dict[int, _PidEvents] = field(default_factory=dict, init=False, repr=False)
    query_log: List[Dict] = field(default_factory=list, init=False, repr=False)

    # ------------------------------------------------------------------ build
    def __post_init__(self) -> None:
        leaked = [c for c in LABEL_COLUMNS if c in self.provenance.columns]
        assert not leaked, f"Emulator must be built from a label-free frame; found {leaked}"

        df = self.provenance
        self._time_col = "event_time" if "event_time" in df.columns else "time"

        proc = df[df["type"] == "Process"].copy()
        proc = proc.dropna(subset=["pid", self._time_col])
        proc["pid"] = proc["pid"].astype(int)
        proc = proc.sort_values(self._time_col)
        self._proc_by_pid = {int(pid): g for pid, g in proc.groupby("pid", sort=False)}
        # Children index: one entry per distinct (pid, exe, name) so node versions are not double-counted.
        proc_with_ppid = proc.dropna(subset=["ppid"]).copy()
        proc_with_ppid["ppid"] = proc_with_ppid["ppid"].astype(int)
        proc_with_ppid["_cmd_rank"] = proc_with_ppid["command line"].notna().astype(int) if "command line" in proc_with_ppid.columns else 0
        proc_with_ppid = (
            proc_with_ppid.sort_values([self._time_col, "_cmd_rank"], ascending=[True, False])
            .groupby(["pid", "exe", "name"], sort=False, dropna=False)
            .agg({self._time_col: "first", "ppid": "first", **({"command line": "first"} if "command line" in proc_with_ppid.columns else {})})
            .reset_index()
            .sort_values(self._time_col)
        )
        self._children_by_ppid = {int(ppid): g for ppid, g in proc_with_ppid.groupby("ppid", sort=False)}

        # id -> pid for Process nodes (ids are unique per Process node in this dataset).
        id_to_pid = proc.drop_duplicates("id").set_index("id")["pid"].to_dict()

        art = df[df["type"] == "Artifact"].drop_duplicates("id").set_index("id")
        art_cols = {
            "subtype": art["subtype"] if "subtype" in art.columns else None,
            "path": art["path"] if "path" in art.columns else None,
            "permissions": art["permissions"] if "permissions" in art.columns else None,
            "remote address": art["remote address"] if "remote address" in art.columns else None,
            "remote port": art["remote port"] if "remote port" in art.columns else None,
            "protocol": art["protocol"] if "protocol" in art.columns else None,
        }

        edges = df[df["from"].notna() & df["to"].notna()].copy()
        edges = edges.dropna(subset=[self._time_col])
        # Attribute each edge to a pid and an artifact id.
        etype = edges["type"]
        pid_series = pd.Series(np.nan, index=edges.index, dtype="float64")
        art_series = pd.Series(pd.NA, index=edges.index, dtype="object")

        m = etype == "Used"
        pid_series[m] = edges.loc[m, "from"].map(id_to_pid)
        art_series[m] = edges.loc[m, "to"]

        m = etype == "WasGeneratedBy"
        pid_series[m] = edges.loc[m, "to"].map(id_to_pid)
        art_series[m] = edges.loc[m, "from"]

        m = etype == "WasDerivedFrom"
        if "pid" in edges.columns:
            pid_series[m] = pd.to_numeric(edges.loc[m, "pid"], errors="coerce")
        art_series[m] = edges.loc[m, "to"]

        m = etype == "WasTriggeredBy"
        pid_series[m] = edges.loc[m, "from"].map(id_to_pid)

        edges["_pid"] = pid_series
        edges["_art"] = art_series
        edges = edges.dropna(subset=["_pid"])
        edges["_pid"] = edges["_pid"].astype(int)
        for name, col in art_cols.items():
            if col is not None:
                edges["_" + name.replace(" ", "_")] = edges["_art"].map(col)
            else:
                edges["_" + name.replace(" ", "_")] = pd.NA
        edges = edges.sort_values(self._time_col)

        keep = [self._time_col, "type", "operation", "_subtype", "_path", "_permissions",
                "_remote_address", "_remote_port", "_protocol"]
        for pid, g in edges.groupby("_pid", sort=False):
            recs = []
            for row in g[keep].itertuples(index=False):
                recs.append(
                    {
                        "time": float(row[0]),
                        "edge": row[1],
                        "operation": _clean(row[2]),
                        "subtype": _clean(row[3]),
                        "path": _clean(row[4]),
                        "permissions": _clean(row[5]),
                        "remote_address": _clean(row[6]),
                        "remote_port": _clean(row[7]),
                        "protocol": _clean(row[8]),
                    }
                )
            self._events_by_pid[int(pid)] = _PidEvents(times=g[self._time_col].to_numpy(dtype=float), records=recs)

        if self.flows is not None and len(self.flows):
            self._flow_index = self.flows.set_index("flow_id", drop=False)
        else:
            self._flow_index = pd.DataFrame()

    # ---------------------------------------------------------------- helpers
    def _log(self, tool: str, t: float, **kw) -> None:
        self.query_log.append({"tool": tool, "t": float(t), **kw})

    def _events_in(self, pid: int, t: float, window: Optional[float]) -> List[Dict]:
        ev = self._events_by_pid.get(int(pid))
        if ev is None:
            return []
        hi = int(np.searchsorted(ev.times, t, side="right"))
        lo = 0 if window is None else int(np.searchsorted(ev.times, t - window, side="left"))
        out = ev.records[lo:hi]
        for r in out:
            assert r["time"] <= t, "causality guard violated"
        return out

    # ------------------------------------------------------------------ tools
    def get_process(self, pid: int, t: float) -> Dict:
        """Latest Process node version for ``pid`` observed at or before ``t``."""
        self._log("get_process", t, pid=int(pid))
        df = self._proc_by_pid.get(int(pid))
        if df is None or df.empty:
            return {"status": UNKNOWN}
        times = df[self._time_col].to_numpy(dtype=float)
        idx = int(np.searchsorted(times, t, side="right")) - 1
        if idx < 0:
            return {"status": NOT_YET_SEEN}
        # Prefer the latest version that carries a command line, if one exists at or before t.
        sub = df.iloc[: idx + 1]
        with_cmd = sub[sub["command line"].notna()] if "command line" in sub.columns else sub.iloc[0:0]
        row = with_cmd.iloc[-1] if len(with_cmd) else sub.iloc[-1]
        assert float(row[self._time_col]) <= t, "causality guard violated"
        return {
            "status": "OK",
            "pid": int(row["pid"]),
            "ppid": int(row["ppid"]) if not pd.isna(row.get("ppid")) else None,
            "exe": _clean(row.get("exe")),
            "name": _clean(row.get("name")),
            "command_line": _clean(row.get("command line")),
            "uid": _num(row.get("uid")),
            "euid": _num(row.get("euid")),
            "observed_at": float(row[self._time_col]),
            "age_seconds": float(t - float(row[self._time_col])),
        }

    def get_process_tree(self, pid: int, t: float, depth: int = 5) -> List[Dict]:
        """Ancestry chain pid -> ppid -> ... using only nodes observed at or before ``t``."""
        self._log("get_process_tree", t, pid=int(pid), depth=depth)
        tree: List[Dict] = []
        current = int(pid)
        seen = set()
        for _ in range(depth):
            if current in seen:
                break
            seen.add(current)
            proc = self.get_process(current, t)
            self.query_log.pop()  # do not double-count nested lookups
            if proc.get("status") != "OK":
                tree.append({"status": proc.get("status"), "pid": current})
                break
            tree.append(proc)
            if proc.get("ppid") is None or proc["ppid"] <= 0:
                break
            current = int(proc["ppid"])
        return tree

    def get_socket_events(self, pid: int, t: float, window: Optional[float] = None) -> List[Dict]:
        """Socket-related edges of ``pid`` in ``[t - window, t]``."""
        self._log("get_socket_events", t, pid=int(pid), window=window)
        return [
            {k: r[k] for k in ("time", "operation", "remote_address", "remote_port", "protocol")}
            for r in self._events_in(pid, t, window)
            if r["subtype"] == "network socket"
        ]

    def get_file_events(self, pid: int, t: float, window: Optional[float] = None) -> List[Dict]:
        """File-related edges of ``pid`` in ``[t - window, t]``."""
        self._log("get_file_events", t, pid=int(pid), window=window)
        return [
            {k: r[k] for k in ("time", "edge", "operation", "subtype", "path", "permissions")}
            for r in self._events_in(pid, t, window)
            if r["subtype"] in ("file", "directory", "link")
        ]

    def get_children(self, pid: int, t: float, window: Optional[float] = None) -> List[Dict]:
        """Child processes (by ppid) first observed in ``[t - window, t]``."""
        self._log("get_children", t, pid=int(pid), window=window)
        df = self._children_by_ppid.get(int(pid))
        if df is None or df.empty:
            return []
        times = df[self._time_col].to_numpy(dtype=float)
        hi = int(np.searchsorted(times, t, side="right"))
        lo = 0 if window is None else int(np.searchsorted(times, t - window, side="left"))
        sub = df.iloc[lo:hi]
        out = []
        for _, row in sub.iterrows():
            assert float(row[self._time_col]) <= t
            out.append(
                {
                    "time": float(row[self._time_col]),
                    "pid": int(row["pid"]),
                    "exe": _clean(row.get("exe")),
                    "name": _clean(row.get("name")),
                    "command_line": _clean(row.get("command line")),
                }
            )
        return out

    def get_flow_stats(self, flow_id: str) -> Dict:
        if isinstance(self._flow_index, pd.DataFrame) and not self._flow_index.empty:
            if flow_id in self._flow_index.index:
                return self._flow_index.loc[flow_id].to_dict()
        return {"status": UNKNOWN}

    # --------------------------------------------------------------- features
    def context_features(self, pid: int, t: float, window: Optional[float]) -> Dict[str, float]:
        """Numeric summary of causal context, usable as RF features (label-free)."""
        recs = self._events_in(pid, t, window)
        files = [r for r in recs if r["subtype"] in ("file", "directory", "link")]
        socks = [r for r in recs if r["subtype"] == "network socket"]
        paths = [r["path"] for r in files if r["path"]]
        ops = [r["operation"] for r in recs]
        children = self._children_by_ppid.get(int(pid))
        n_children = 0
        if children is not None and len(children):
            ct = children[self._time_col].to_numpy(dtype=float)
            hi = int(np.searchsorted(ct, t, side="right"))
            lo = 0 if window is None else int(np.searchsorted(ct, t - window, side="left"))
            n_children = hi - lo
        return {
            "ctx_n_events": float(len(recs)),
            "ctx_n_file_events": float(len(files)),
            "ctx_n_socket_events": float(len(socks)),
            "ctx_n_unique_paths": float(len(set(paths))),
            "ctx_n_unique_remote": float(len({(r["remote_address"], r["remote_port"]) for r in socks})),
            "ctx_n_unlink": float(sum(1 for o in ops if o == "unlink")),
            "ctx_n_connect": float(sum(1 for o in ops if o == "connect")),
            "ctx_n_rename": float(sum(1 for o in ops if o and o.startswith("rename"))),
            "ctx_n_load": float(sum(1 for o in ops if o == "load")),
            "ctx_n_chmod": float(sum(1 for o in ops if o == "chmod")),
            "ctx_n_children": float(n_children),
            "ctx_n_home_paths": float(sum(1 for p in paths if p.startswith("/home"))),
            "ctx_n_tmp_paths": float(sum(1 for p in paths if p.startswith("/tmp") or p.startswith("/dev/shm"))),
            "ctx_n_etc_paths": float(sum(1 for p in paths if p.startswith("/etc"))),
            "ctx_n_hidden_paths": float(sum(1 for p in paths if "/." in p)),
        }


def _clean(v):
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(v)
    return "" if s in ("nan", "<NA>", "None") else s


def _num(v):
    try:
        if v is None or pd.isna(v):
            return None
    except (TypeError, ValueError):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
