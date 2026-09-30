"""Unit checks for the emulator guarantees. Run: .venv/bin/python -m pytest tests -q
or plainly: .venv/bin/python tests/test_emulator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from emulator import NOT_YET_SEEN, UNKNOWN, Emulator, strip_labels
from metrics import expected_calibration_error, reliability_bins


def _toy() -> pd.DataFrame:
    cols = ["id", "type", "from", "to", "pid", "ppid", "exe", "name", "command line", "event_time",
            "operation", "subtype", "path", "permissions", "remote address", "remote port", "protocol", "label", "subLabel"]
    rows = [
        # parent process, child process (two versions), file artifact, socket artifact
        dict(id="P1", type="Process", pid=1, ppid=0, exe="/bin/bash", name="bash", event_time=100.0),
        dict(id="P2a", type="Process", pid=2, ppid=1, exe="/bin/sh", name="sh", event_time=110.0),
        dict(id="P2b", type="Process", pid=2, ppid=1, exe="/bin/sh", name="sh", event_time=110.0, **{"command line": "sh -c evil"}),
        dict(id="A1", type="Artifact", subtype="file", path="/etc/shadow", permissions=600),
        dict(id="A2", type="Artifact", subtype="network socket", **{"remote address": "10.0.0.9", "remote port": 4444, "protocol": "tcp"}),
        # edges: file read at 112 (past), socket connect at 118 (future relative to t=115), unlink at 120
        dict(id="E1", type="Used", **{"from": "P2b", "to": "A1"}, event_time=112.0, operation="open"),
        dict(id="E2", type="WasGeneratedBy", **{"from": "A2", "to": "P2b"}, event_time=118.0, operation="connect"),
        dict(id="E3", type="WasGeneratedBy", **{"from": "A1", "to": "P2b"}, event_time=120.0, operation="unlink"),
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["label"] = 1
    df["subLabel"] = "collection"
    return df


def test_rejects_labels():
    try:
        Emulator(provenance=_toy())
    except AssertionError:
        return
    raise AssertionError("emulator accepted a frame with label columns")


def test_causality():
    emu = Emulator(provenance=strip_labels(_toy()))
    t = 115.0
    files = emu.get_file_events(2, t, window=10.0)
    socks = emu.get_socket_events(2, t, window=10.0)
    assert [f["operation"] for f in files] == ["open"], files  # unlink at 120 excluded
    assert socks == [], socks  # connect at 118 excluded
    assert all(f["time"] <= t for f in files)
    later = emu.get_socket_events(2, 119.0, window=10.0)
    assert later and later[0]["remote_address"] == "10.0.0.9" and later[0]["remote_port"] == "4444.0"
    # lookback window respected
    assert emu.get_file_events(2, 130.0, window=5.0) == []


def test_process_versions_and_sentinels():
    emu = Emulator(provenance=strip_labels(_toy()))
    assert emu.get_process(2, 105.0)["status"] == NOT_YET_SEEN
    assert emu.get_process(99, 105.0)["status"] == UNKNOWN
    p = emu.get_process(2, 115.0)
    assert p["status"] == "OK" and p["command_line"] == "sh -c evil" and p["ppid"] == 1
    tree = emu.get_process_tree(2, 115.0, depth=5)
    assert [n.get("pid") for n in tree] == [2, 1]
    kids = emu.get_children(1, 115.0, window=None)
    assert len(kids) == 1 and kids[0]["pid"] == 2
    assert emu.get_children(1, 105.0, window=None) == []


def test_context_features():
    emu = Emulator(provenance=strip_labels(_toy()))
    c = emu.context_features(2, 121.0, window=20.0)
    assert c["ctx_n_file_events"] == 2 and c["ctx_n_socket_events"] == 1 and c["ctx_n_unlink"] == 1
    assert c["ctx_n_etc_paths"] == 2 and c["ctx_n_unique_remote"] == 1


def test_ece_includes_probability_one():
    y = np.array([1, 1, 0, 0])
    p = np.array([1.0, 1.0, 1.0, 1.0])
    # all four in the top bin: accuracy 0.5, confidence 1.0 -> ECE 0.5 (old code returned 0.0)
    assert abs(expected_calibration_error(y, p) - 0.5) < 1e-9
    rb = reliability_bins(y, p)
    assert rb["count"][-1] == 4 and rb["count"][0] == 0 and rb["accuracy"][0] is None




def test_base64_heuristic_ignores_paths():
    from units import looks_like_base64_blob

    assert looks_like_base64_blob('sh -c eval "$(echo bWtkaXIgLXAgc3RhZ2VkICYmIGVjaG8gJFBXRC9zdGFnZWQ= | base64 --decode)"')
    assert not looks_like_base64_blob("/usr/lib/x86_64-linux-gnu/e2fsprogs/e2scrub_all_cron")
    assert not looks_like_base64_blob("cp -t /tmp/.s /home/erfan/.mozilla/firefox/qphy68q1.default-release/pkcs11.txt")
    assert not looks_like_base64_blob("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")  # no case/digit mix




def test_rolling_folds_time_equal_duration():
    import pandas as pd
    from split import rolling_folds_time

    # 100 units in the first hour, 2 units in the second hour: count-based folds would put
    # the two late units into the same chunk as early ones; time-based folds keep them apart.
    df = pd.DataFrame({"t": list(np.linspace(0, 3500, 100)) + [7000.0, 7200.0]})  # span 0..7200, boundary at 3600
    folds = rolling_folds_time(df, "t", 1)
    tr, te = folds[0]
    assert set(te.tolist()) == {100, 101} and len(tr) == 100
    # every test unit is later than every training unit (rolling origin)
    assert df.t.iloc[te].min() > df.t.iloc[tr].max()


def test_memory_path_helpers():
    from memory_propagation import cmd_paths, interesting, matches

    assert cmd_paths("sh -c rm -f /tmp/.s/firefox_password.txt") == {"/tmp/.s/firefox_password.txt"}
    assert cmd_paths("rm /tmp") == set()  # top-level directories are never remembered
    baseline = {"/home/erfan/.bashrc"}
    assert not interesting("/usr/bin/dpkg", baseline) and not interesting("/home/erfan/.bashrc", baseline)
    assert interesting("/home/erfan/staged/x.txt", baseline)
    # directory containment in both directions, exact match otherwise
    assert matches({"/tmp/.s/a"}, {"/tmp/.s"}) == ["/tmp/.s/a"]
    assert matches({"/tmp/.s"}, {"/tmp/.s/a"}) == ["/tmp/.s"]
    assert matches({"/tmp/.sx"}, {"/tmp/.s"}) == []


def test_investigator_evidence_is_causal():
    """Path history, baseline counts and the case notebook never return anything later than the asking time."""
    from investigator import CaseNotebook, HostBaseline, PathIndex

    emu = Emulator(provenance=strip_labels(_toy()))
    units = pd.DataFrame({"pid": [2, 3], "name": ["sh", "rm"], "parent_name": ["bash", "sh"], "start_time": [110.0, 130.0],
                          "command line": ["sh -c evil", "rm /etc/shadow"]})
    idx = PathIndex(emu, units)
    assert [r["operation"] for r in idx.history("/etc/shadow", 115.0)] == ["open"]  # the unlink at 120 is in the future
    assert [r["operation"] for r in idx.history("shadow", 125.0)] == ["open", "unlink"]  # bare file name
    assert idx.history("/etc/shadow", 105.0) == []
    base = HostBaseline(units.iloc[:1], units)
    assert base.lookup("rm", "sh", 130.0)["earlier_runs_in_current_period"] == 0  # a process does not count itself
    assert base.lookup("sh", "bash", 130.0)["reference_period_runs_with_this_parent"] == 1

    nb = CaseNotebook()
    nb.add(pid=2, ppid=1, name="sh", command_line="sh -c evil", decision_time=115.0, verdict="MALICIOUS", score=0.9, stage="staging", note="n", paths=["/tmp/.s"])
    assert nb.retrieve(pid=3, ancestors=[2], ppid=2, paths=set(), t=114.0) == []  # note not written yet
    assert [n["relation"] for n in nb.retrieve(pid=3, ancestors=[2], ppid=2, paths=set(), t=135.0)] == ["parent"]
    assert nb.retrieve(pid=9, ancestors=[], ppid=7, paths={"/tmp/.s/a.txt"}, t=135.0)[0]["relation"].startswith("touched the same path")
    assert nb.retrieve(pid=9, ancestors=[], ppid=7, paths={"/tmp/.sx"}, t=135.0) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
