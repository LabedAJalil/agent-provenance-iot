from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

# Columns that must never be model inputs.
LABEL_LIKE = {"label", "subLabel", "is_attack", "attack_tactic", "attack_technique"}
# Absolute timestamps and kernel identifiers: with a chronological split they are
# time proxies (pid is monotone in time on a fresh VM) and do not transfer.
TIME_LIKE = {"time", "seen time", "start time", "epoch", "event_time", "ts", "decision_time"}
ID_LIKE = {"pid", "ppid", "tgid", "fd", "id", "from", "to", "event id", "flow_id", "pcap_file"}


@dataclass(frozen=True)
class FeatureSpec:
    numeric: List[str]
    categorical: List[str]
    text: List[str] = field(default_factory=list)


def _present(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns and df[c].notna().any()]


def legacy_provenance_features(df: pd.DataFrame) -> FeatureSpec:
    """The feature set used by the original runs. Kept only for comparison; it
    includes absolute timestamps and pids, which are time proxies."""
    numeric = ["uid", "euid", "gid", "egid", "pid", "ppid", "tgid", "fd", "time", "seen time",
               "start time", "epoch", "local port", "remote port"]
    categorical = ["type", "operation", "protocol", "subtype", "source", "exe", "name", "path",
                   "local address", "remote address"]
    return FeatureSpec(numeric=_present(df, numeric), categorical=_present(df, categorical))


def clean_provenance_features(df: pd.DataFrame) -> FeatureSpec:
    """Row-level features without time or identifier leakage."""
    numeric = ["uid", "euid", "gid", "egid", "local port", "remote port"]
    categorical = ["type", "operation", "protocol", "subtype", "source", "exe", "name",
                   "local address", "remote address"]
    return FeatureSpec(numeric=_present(df, numeric), categorical=_present(df, categorical))


def process_unit_features(
    df: pd.DataFrame, use_context: bool = True, use_cmdline: bool = True, use_ancestry: bool = True
) -> FeatureSpec:
    """Features for the process-level unit built by ``units.build_process_units``.

    use_cmdline : the process's own command line (text hash + summary flags)
    use_context : counts of file/socket/child events in the decision window
    use_ancestry: parent/grandparent identity, ancestor command lines (text hash + flags)
    """
    numeric = ["uid", "euid", "gid", "egid"]
    categorical = ["exe", "name"]
    text: List[str] = []
    if use_cmdline:
        numeric += ["cmd_len", "cmd_n_tokens", "cmd_has_base64", "cmd_has_pipe", "cmd_has_eval", "cmd_has_http", "cmd_has_redirect"]
        text.append("command line")
    if use_context:
        numeric += [c for c in df.columns if c.startswith("ctx_")]
    if use_ancestry:
        numeric += [c for c in df.columns if c.startswith("anc_")]
        categorical += ["parent_exe", "parent_name", "grandparent_exe"]
        text.append("ancestor_cmdline")
    return FeatureSpec(numeric=_present(df, numeric), categorical=_present(df, categorical), text=text)


def default_flow_features(df: pd.DataFrame) -> FeatureSpec:
    drop = LABEL_LIKE | TIME_LIKE | ID_LIKE
    numeric = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    return FeatureSpec(numeric=numeric, categorical=[])


def assert_no_leak(spec: FeatureSpec) -> None:
    used = set(spec.numeric) | set(spec.categorical) | set(spec.text)
    bad = used & LABEL_LIKE
    assert not bad, f"label columns in feature spec: {bad}"


def build_feature_pipeline(spec: FeatureSpec, n_text_features: int = 4096) -> ColumnTransformer:
    assert_no_leak(spec)
    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]
    )
    transformers = [("num", numeric_pipe, spec.numeric), ("cat", categorical_pipe, spec.categorical)]
    for i, col in enumerate(spec.text):
        text_pipe = Pipeline(
            steps=[
                ("to_str", FunctionTransformer(lambda s: s.fillna("").astype(str).to_numpy().ravel(), validate=False)),
                ("hash", HashingVectorizer(n_features=n_text_features, token_pattern=r"[A-Za-z0-9_./-]+", alternate_sign=False, norm=None)),
            ]
        )
        transformers.append((f"text{i}", text_pipe, [col]))
    return ColumnTransformer(transformers=transformers, remainder="drop")
