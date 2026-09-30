"""Semantic channel: three conditions for the same model.

narrative : process attributes + own command line only (no retrieval)
context   : narrative + fixed causal context (ancestry, files, sockets, children)
agent     : the CONTEXT record + read-only tools (run_rolling.py passes narrative_for_unit to this
            condition); the MODEL chooses which lookups to issue, within a tool-call budget, and must
            cite evidence. The three conditions are therefore nested.
investigator : a behaviour record with host baseline and case-notebook memory (investigator.py), judged in a
            plan -> lookups -> reflection loop with three more lookups. The zero-shot conditions above are unchanged.

Every call records tokens, latency, tool calls and the evidence string. Results
are cached in SQLite keyed by (condition, model, prompt-version, unit key).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

PROMPT_VERSION = "v3"

SYSTEM_RULES = (
    "You are a host-intrusion analyst reviewing ONE process observed on a Linux industrial-IoT workstation. "
    "Decide whether this process is part of an attack campaign (living-off-the-land activity: staging, "
    "credential access, discovery, exfiltration, persistence, defence evasion, lateral movement, C2) "
    "or ordinary system/administrative activity. Evidence in the record beats intuition; if evidence is "
    "missing say so. Output STRICT JSON only, no markdown: "
    '{"verdict":"MALICIOUS"|"BENIGN","risk_score":<0..1>,"evidence":"<one or two sentences citing the records used>"}. '
    "risk_score is your probability that the process is malicious."
)

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "name": "get_process_tree",
        "description": "Ancestry chain of a pid (parent, grandparent, ...) with executables and command lines, as observed at or before the decision time.",
        "parameters": {"type": "object", "properties": {"pid": {"type": "integer"}, "depth": {"type": "integer", "default": 4}}, "required": ["pid"]},
    },
    {
        "type": "function",
        "name": "get_file_events",
        "description": "File operations (open, load, rename, unlink, chmod) performed by a pid within the last `window` seconds before the decision time, with paths.",
        "parameters": {"type": "object", "properties": {"pid": {"type": "integer"}, "window": {"type": "number", "default": 5}}, "required": ["pid"]},
    },
    {
        "type": "function",
        "name": "get_socket_events",
        "description": "Network socket operations (connect) by a pid within the last `window` seconds before the decision time, with remote address and port.",
        "parameters": {"type": "object", "properties": {"pid": {"type": "integer"}, "window": {"type": "number", "default": 5}}, "required": ["pid"]},
    },
    {
        "type": "function",
        "name": "get_children",
        "description": "Child processes spawned by a pid within the last `window` seconds before the decision time, with command lines.",
        "parameters": {"type": "object", "properties": {"pid": {"type": "integer"}, "window": {"type": "number", "default": 5}}, "required": ["pid"]},
    },
]


def _extract_json_object(text: str) -> Dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return {}
    return {}


@dataclass
class LLMResult:
    score: float
    verdict: str
    evidence: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    n_api_calls: int = 1
    tool_calls: List[Dict] = field(default_factory=list)
    cached: bool = False
    error: str = ""
    model_snapshot: str = ""  # dated model id returned by the API (e.g. gpt-4o-mini-2024-07-18)
    response_id: str = ""  # id of the final API response, for provider-side audit
    parse_failed: bool = False  # chat backend: no valid JSON verdict after one retry; scored as "no alert" and counted
    extra: Dict[str, Any] = field(default_factory=dict)  # investigator: stage, leads_to, note (fields of the final JSON)
    trajectory: List[Dict] = field(default_factory=list)  # investigator: plan and reflection texts, in order

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @staticmethod
    def from_json(s: str) -> "LLMResult":
        d = json.loads(s)
        d["cached"] = True
        return LLMResult(**d)


class LLMCache:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        with sqlite3.connect(self.path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS llm_results (key TEXT PRIMARY KEY, result TEXT, created REAL)")

    def get(self, key: str) -> Optional[LLMResult]:
        with self._lock, sqlite3.connect(self.path) as conn:
            row = conn.execute("SELECT result FROM llm_results WHERE key = ?", (key,)).fetchone()
        return LLMResult.from_json(row[0]) if row else None

    def set(self, key: str, result: LLMResult) -> None:
        with self._lock, sqlite3.connect(self.path) as conn:
            conn.execute("INSERT OR REPLACE INTO llm_results (key, result, created) VALUES (?, ?, ?)", (key, result.to_json(), time.time()))


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_output_tokens: int = 300
    mode: str = "none"  # none | openai
    api_key: str = ""
    tool_budget: int = 6
    max_retries: int = 4
    mask_signature: bool = False  # strip base64/eval signature from tool outputs too
    cache_only: bool = False  # never call the API; return an error for uncached units
    # backend "responses": OpenAI Responses API (the gpt-4o-mini runs). backend "chat": any OpenAI-compatible
    # chat-completions endpoint, e.g. the Hugging Face router (https://router.huggingface.co/v1) for open models.
    backend: str = "responses"
    base_url: str = ""
    seed: Optional[int] = None
    no_think: bool = False  # append Qwen3's "/no_think" soft switch so hybrid models answer without a reasoning block
    investigator_max_output_tokens: int = 900  # per step; the plan and reflection steps need room to reason


class SemanticScorer:
    def __init__(self, cfg: LLMConfig, cache: Optional[LLMCache] = None) -> None:
        self.cfg = cfg
        self.cache = cache
        self._client = None

    # ------------------------------------------------------------------ client
    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            # chat backend: routed providers can stall; fail the request after 60 s and let the retry loop handle it
            self._client = OpenAI(api_key=self.cfg.api_key or None, base_url=self.cfg.base_url or None,
                                  **({"timeout": 60.0, "max_retries": 0} if self.cfg.backend == "chat" else {}))
        return self._client

    def _create(self, **kw):
        client = self._get_client()
        return self._with_retries(lambda: client.responses.create(model=self.cfg.model, temperature=self.cfg.temperature,
                                                                  max_output_tokens=self.cfg.max_output_tokens, **kw))

    def _create_chat(self, max_tokens: Optional[int] = None, **kw):
        client = self._get_client()
        extra = {"seed": self.cfg.seed} if self.cfg.seed is not None else {}
        return self._with_retries(lambda: client.chat.completions.create(model=self.cfg.model, temperature=self.cfg.temperature,
                                                                         max_tokens=max_tokens or self.cfg.max_output_tokens, **extra, **kw))

    def _with_retries(self, call):
        delay = 2.0
        patient_waits = 0
        attempt = 0
        while attempt < self.cfg.max_retries:
            try:
                return call()
            except Exception as exc:  # rate limit / transient
                if "per day" in str(exc) or "RPD" in str(exc):
                    # Daily request quota exhausted. Default: fail fast; the unit stays uncached and a later run
                    # with the same arguments fills it. With LLM_RPD_PATIENT=1 (use --workers 1) wait for the
                    # rolling 24 h window to release a request instead, so no partly scored unit is wasted.
                    if os.environ.get("LLM_RPD_PATIENT") == "1":
                        patient_waits += 1
                        if patient_waits <= 2000:
                            time.sleep(10.0)
                            continue
                    raise RuntimeError("daily request quota exhausted (RPD); rerun after the quota resets") from exc
                if attempt == self.cfg.max_retries - 1:
                    raise
                attempt += 1
                time.sleep(delay)
                delay *= 2

    # ------------------------------------------------------------------ public
    def score(self, condition: str, unit_key: str, narrative: str, emulator=None, pid: Optional[int] = None,
              decision_time: Optional[float] = None, extra_tools: Optional[Dict[str, Callable[[Dict], Any]]] = None) -> LLMResult:
        if self.cfg.mode == "none":
            return LLMResult(score=0.5, verdict="UNKNOWN", evidence="llm disabled")
        cond_key = condition
        if condition == "investigator":
            from investigator import INVESTIGATOR_PROMPT_VERSION

            cond_key = f"investigator-{INVESTIGATOR_PROMPT_VERSION}-{self.cfg.investigator_max_output_tokens}" + ("-nothink" if self.cfg.no_think else "")
        key = hashlib.sha256(f"{cond_key}|{self.cfg.model}|{PROMPT_VERSION}|{self.cfg.tool_budget}|{unit_key}|{narrative}".encode()).hexdigest()
        if self.cache:
            hit = self.cache.get(key)
            if hit:
                return hit
        if self.cfg.cache_only:
            return LLMResult(score=0.5, verdict="ERROR", evidence="", error="not cached (cache-only mode)")
        t0 = time.time()
        try:
            chat = self.cfg.backend == "chat"
            if condition in ("narrative", "context", "sweep"):
                res = self._single_shot_chat(narrative) if chat else self._single_shot(narrative)
            elif condition == "agent":
                assert emulator is not None and pid is not None and decision_time is not None
                loop = self._agent_loop_chat if chat else self._agent_loop
                res = loop(narrative, emulator, int(pid), float(decision_time))
            elif condition == "investigator":
                assert emulator is not None and pid is not None and decision_time is not None
                res = self._investigator_loop(narrative, emulator, int(pid), float(decision_time), extra_tools or {})
            else:
                raise ValueError(condition)
        except Exception as exc:
            res = LLMResult(score=0.5, verdict="ERROR", evidence="", error=str(exc)[:300])
        res.latency_s = time.time() - t0
        if self.cache and not res.error:
            self.cache.set(key, res)
        return res

    # --------------------------------------------------------------- variants
    def _parse(self, text: str) -> tuple[float, str, str]:
        d = _extract_json_object(text)
        try:
            score = float(np.clip(float(d.get("risk_score", 0.5)), 0.0, 1.0))
        except (TypeError, ValueError):
            score = 0.5
        return score, str(d.get("verdict", "UNKNOWN")), str(d.get("evidence", ""))[:600]

    def _single_shot(self, narrative: str) -> LLMResult:
        resp = self._create(input=[{"role": "system", "content": SYSTEM_RULES},
                                   {"role": "user", "content": "PROCESS RECORD:\n" + narrative + "\n\nReturn the JSON verdict."}])
        score, verdict, evidence = self._parse(getattr(resp, "output_text", "") or "")
        u = getattr(resp, "usage", None)
        return LLMResult(score=score, verdict=verdict, evidence=evidence,
                         input_tokens=getattr(u, "input_tokens", 0) or 0, output_tokens=getattr(u, "output_tokens", 0) or 0,
                         model_snapshot=str(getattr(resp, "model", "") or ""), response_id=str(getattr(resp, "id", "") or ""))

    def _agent_loop(self, narrative: str, emulator, pid: int, t: float) -> LLMResult:
        budget = self.cfg.tool_budget
        system = SYSTEM_RULES + (
            f" You may call the read-only lookup tools at most {budget} times in total to gather evidence about this "
            f"process (pid {pid}) or its relatives before answering. All lookups are evaluated at the decision time; "
            "nothing after it is visible. Call tools only when they could change your verdict, then return the JSON."
        )
        inputs: List[Dict[str, Any]] = [{"role": "system", "content": system},
                                        {"role": "user", "content": "PROCESS RECORD:\n" + narrative + "\n\nInvestigate if useful, then return the JSON verdict."}]
        calls: List[Dict] = []
        in_tok = out_tok = 0
        n_api = 0
        for _round in range(budget + 2):
            remaining = budget - len(calls)
            resp = self._create(input=inputs, tools=TOOL_SCHEMAS if remaining > 0 else [], tool_choice="auto" if remaining > 0 else "none")
            n_api += 1
            u = getattr(resp, "usage", None)
            in_tok += getattr(u, "input_tokens", 0) or 0
            out_tok += getattr(u, "output_tokens", 0) or 0
            fn_items = [o for o in resp.output if getattr(o, "type", "") == "function_call"]
            if not fn_items:
                score, verdict, evidence = self._parse(getattr(resp, "output_text", "") or "")
                return LLMResult(score=score, verdict=verdict, evidence=evidence, input_tokens=in_tok,
                                 output_tokens=out_tok, n_api_calls=n_api, tool_calls=calls,
                                 model_snapshot=str(getattr(resp, "model", "") or ""), response_id=str(getattr(resp, "id", "") or ""))
            for item in fn_items:
                inputs.append({"type": "function_call", "call_id": item.call_id, "name": item.name, "arguments": item.arguments})
                if len(calls) >= budget:
                    output = {"error": "tool budget exhausted"}
                else:
                    output = self._run_tool(emulator, item.name, item.arguments, t)
                    # The observation is stored with the call so that the trajectory (action, observation) can be audited from the cache.
                    calls.append({"tool": item.name, "args": item.arguments[:200], "n_records": len(output) if isinstance(output, list) else 1,
                                  "observation": json.dumps(output, default=str)[:6000]})
                inputs.append({"type": "function_call_output", "call_id": item.call_id, "output": json.dumps(output, default=str)[:6000]})
        return LLMResult(score=0.5, verdict="UNKNOWN", evidence="agent did not conclude", input_tokens=in_tok,
                         output_tokens=out_tok, n_api_calls=n_api, tool_calls=calls, error="no final answer")

    # ------------------------------------------------ chat-completions backend (open models, same prompts and tools)
    @staticmethod
    def _strip_think(text: str) -> str:
        """Drop a reasoning block emitted by hybrid models; only the answer is parsed."""
        import re as _re
        text = _re.sub(r"<think>.*?</think>", "", text or "", flags=_re.S)
        return text.split("</think>")[-1].strip()

    def _parse_strict(self, text: str):
        """(score, verdict, evidence, ok). ok is False when no JSON object with a valid verdict is present."""
        d = _extract_json_object(self._strip_think(text))
        verdict = str(d.get("verdict", "")).upper()
        if verdict not in ("MALICIOUS", "BENIGN"):
            return 0.0, "UNPARSABLE", "", False
        try:
            score = float(np.clip(float(d.get("risk_score")), 0.0, 1.0))
        except (TypeError, ValueError):
            score = 0.9 if verdict == "MALICIOUS" else 0.1
        return score, verdict, str(d.get("evidence", ""))[:600], True

    def _user(self, text: str) -> str:
        return text + (" /no_think" if self.cfg.no_think else "")

    def _finish_chat(self, messages, resp, in_tok, out_tok, n_api, calls) -> LLMResult:
        """Parse the final message; on an unparsable answer ask once more, then record a parse failure.
        A parse failure is scored as 'no alert' (score 0) and counted, never silently dropped."""
        score, verdict, evidence, ok = self._parse_strict(resp.choices[0].message.content or "")
        if not ok:
            messages = messages + [{"role": "assistant", "content": (resp.choices[0].message.content or "")[:2000]},
                                   {"role": "user", "content": self._user("Return ONLY the JSON object with verdict, risk_score and evidence.")}]
            resp = self._create_chat(messages=messages)
            n_api += 1
            u = getattr(resp, "usage", None)
            in_tok += getattr(u, "prompt_tokens", 0) or 0
            out_tok += getattr(u, "completion_tokens", 0) or 0
            score, verdict, evidence, ok = self._parse_strict(resp.choices[0].message.content or "")
        d = _extract_json_object(self._strip_think(resp.choices[0].message.content or "")) if ok else {}
        return LLMResult(score=score, verdict=verdict, evidence=evidence, input_tokens=in_tok, output_tokens=out_tok, n_api_calls=n_api,
                         tool_calls=calls, model_snapshot=str(getattr(resp, "model", "") or ""), response_id=str(getattr(resp, "id", "") or ""),
                         parse_failed=not ok, extra={k: str(d.get(k, ""))[:300] for k in ("stage", "leads_to", "note") if k in d})

    def _single_shot_chat(self, narrative: str) -> LLMResult:
        messages = [{"role": "system", "content": SYSTEM_RULES},
                    {"role": "user", "content": self._user("PROCESS RECORD:\n" + narrative + "\n\nReturn the JSON verdict.")}]
        resp = self._create_chat(messages=messages)
        u = getattr(resp, "usage", None)
        return self._finish_chat(messages, resp, getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0, 1, [])

    def _agent_loop_chat(self, narrative: str, emulator, pid: int, t: float) -> LLMResult:
        budget = self.cfg.tool_budget
        system = SYSTEM_RULES + (
            f" You may call the read-only lookup tools at most {budget} times in total to gather evidence about this "
            f"process (pid {pid}) or its relatives before answering. All lookups are evaluated at the decision time; "
            "nothing after it is visible. Call tools only when they could change your verdict, then return the JSON."
        )
        tools = [{"type": "function", "function": {k: v for k, v in sch.items() if k != "type"}} for sch in TOOL_SCHEMAS]
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system},
                                          {"role": "user", "content": self._user("PROCESS RECORD:\n" + narrative + "\n\nInvestigate if useful, then return the JSON verdict.")}]
        calls: List[Dict] = []
        in_tok = out_tok = n_api = 0
        for _round in range(budget + 2):
            remaining = budget - len(calls)
            kw = {"tools": tools, "tool_choice": "auto"} if remaining > 0 else {}
            if remaining <= 0:
                messages.append({"role": "user", "content": self._user("The lookup budget is exhausted. Return the JSON verdict now.")})
            resp = self._create_chat(messages=messages, **kw)
            n_api += 1
            u = getattr(resp, "usage", None)
            in_tok += getattr(u, "prompt_tokens", 0) or 0
            out_tok += getattr(u, "completion_tokens", 0) or 0
            msg = resp.choices[0].message
            tcs = list(getattr(msg, "tool_calls", None) or [])
            if not tcs or remaining <= 0:
                return self._finish_chat(messages, resp, in_tok, out_tok, n_api, calls)
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in tcs]})
            for tc in tcs:
                if len(calls) >= budget:
                    output = {"error": "tool budget exhausted"}
                else:
                    output = self._run_tool(emulator, tc.function.name, tc.function.arguments, t)
                    calls.append({"tool": tc.function.name, "args": (tc.function.arguments or "")[:200], "n_records": len(output) if isinstance(output, list) else 1,
                                  "observation": json.dumps(output, default=str)[:6000]})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(output, default=str)[:6000]})
        return LLMResult(score=0.0, verdict="UNPARSABLE", evidence="loop did not conclude", input_tokens=in_tok, output_tokens=out_tok,
                         n_api_calls=n_api, tool_calls=calls, parse_failed=True)

    # ------------------------------------------------ investigator: plan -> lookups -> reflection -> verdict
    def _investigator_loop(self, record: str, emulator, pid: int, t: float, extra_tools: Dict[str, Callable[[Dict], Any]]) -> LLMResult:
        """Chat-completions on either backend's client. The harness, not the model, moves between the steps, so every
        record has a plan before its first lookup and a reflection after its last; both are kept in the trajectory."""
        from investigator import EXTRA_TOOL_SCHEMAS, INVESTIGATE_PROMPT, INVESTIGATOR_RULES, PLAN_PROMPT, REFLECT_PROMPT

        budget = self.cfg.tool_budget
        big = self.cfg.investigator_max_output_tokens
        schemas = TOOL_SCHEMAS + [sch for sch in EXTRA_TOOL_SCHEMAS if sch["name"] in extra_tools]
        tools = [{"type": "function", "function": {k: v for k, v in sch.items() if k != "type"}} for sch in schemas]
        messages: List[Dict[str, Any]] = [{"role": "system", "content": INVESTIGATOR_RULES},
                                          {"role": "user", "content": self._user(f"CASE FILE (pid {pid}):\n{record}\n\n{PLAN_PROMPT}")}]
        calls: List[Dict] = []
        trajectory: List[Dict] = []
        usage = {"in": 0, "out": 0, "n": 0}

        def ask(**kw):
            resp = self._create_chat(messages=messages, max_tokens=big, **kw)
            u = getattr(resp, "usage", None)
            usage["in"] += getattr(u, "prompt_tokens", 0) or 0
            usage["out"] += getattr(u, "completion_tokens", 0) or 0
            usage["n"] += 1
            return resp

        def lookups(step: str) -> Any:
            """Let the model call tools until it answers in text or the budget is spent; returns the last response."""
            while True:
                remaining = budget - len(calls)
                resp = ask(**({"tools": tools, "tool_choice": "auto"} if remaining > 0 else {}))
                msg = resp.choices[0].message
                tcs = list(getattr(msg, "tool_calls", None) or []) if remaining > 0 else []
                if not tcs:
                    return resp
                messages.append({"role": "assistant", "content": msg.content or "",
                                 "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in tcs]})
                for tc in tcs:
                    if len(calls) >= budget:
                        output: Any = {"error": "lookup budget exhausted"}
                    else:
                        output = self._run_tool(emulator, tc.function.name, tc.function.arguments, t, extra_tools)
                        calls.append({"tool": tc.function.name, "args": (tc.function.arguments or "")[:200], "step": step,
                                      "n_records": len(output) if isinstance(output, list) else 1, "observation": json.dumps(output, default=str)[:6000]})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(output, default=str)[:6000]})

        plan = self._strip_think(ask().choices[0].message.content or "")
        trajectory.append({"step": "plan", "text": plan[:3000]})
        messages.append({"role": "assistant", "content": plan[:3000]})
        messages.append({"role": "user", "content": self._user(INVESTIGATE_PROMPT.format(budget=budget))})
        done = self._strip_think(lookups("investigate").choices[0].message.content or "")
        messages.append({"role": "assistant", "content": done[:2000] or "DONE"})

        resp = None
        for _reflection in range(2):  # a reflection may ask for more lookups once; the second one must conclude
            remaining = budget - len(calls)
            more = (f"If one more lookup would change your verdict you may make it now ({remaining} left) before the JSON. " if remaining > 0 and _reflection == 0 else "")
            messages.append({"role": "user", "content": self._user(REFLECT_PROMPT.replace("{more}", more))})
            n_before = len(calls)
            resp = lookups("reflect") if more else ask()
            text = self._strip_think(resp.choices[0].message.content or "")
            trajectory.append({"step": "reflect", "text": text[:3000], "lookups_added": len(calls) - n_before})
            if len(calls) == n_before:
                break
            messages.append({"role": "assistant", "content": text[:2000]})
        res = self._finish_chat(messages, resp, usage["in"], usage["out"], usage["n"], calls)
        res.trajectory = trajectory
        return res

    def _run_tool(self, emulator, name: str, arguments: str, t: float, extra_tools: Optional[Dict[str, Callable[[Dict], Any]]] = None):
        if extra_tools and name in extra_tools:
            try:
                out = extra_tools[name](json.loads(arguments or "{}"))
            except Exception as exc:
                out = {"error": f"bad arguments: {str(exc)[:100]}"}
        else:
            out = self._run_tool_raw(emulator, name, arguments, t)
        if self.cfg.mask_signature and isinstance(out, list):
            from units import mask_signature

            for rec in out:
                if isinstance(rec, dict) and rec.get("command_line"):
                    rec["command_line"] = mask_signature(rec["command_line"])
        return out

    @staticmethod
    def _run_tool_raw(emulator, name: str, arguments: str, t: float):
        try:
            args = json.loads(arguments or "{}")
        except Exception:
            return {"error": "bad arguments"}
        pid = int(args.get("pid", -1))
        window = float(args.get("window", 5.0))
        if name == "get_process_tree":
            return emulator.get_process_tree(pid, t, depth=int(args.get("depth", 4)))
        if name == "get_file_events":
            return emulator.get_file_events(pid, t, window=window)[:25]
        if name == "get_socket_events":
            return emulator.get_socket_events(pid, t, window=window)[:25]
        if name == "get_children":
            return emulator.get_children(pid, t, window=window)[:25]
        return {"error": f"unknown tool {name}"}
