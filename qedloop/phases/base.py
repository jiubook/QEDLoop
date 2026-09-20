# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Shared helpers for the five phase nodes.

A phase node is a plain function ``state -> delta``.  It may call LLMs, read the
state bus, and run commands; it must not mutate ``state``.  Every side effect it
reports goes back through the reducer channels declared in
:mod:`qedloop.state`.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..core import as_float, find_open, new_id
from ..prompts import PHASE_LABELS

LogFn = Callable[[str, Mapping[str, Any]], None]
#: Mutable box holding the active graph listener.  Phase nodes are built before
#: a trace exists, so they must read the listener at *call* time -- binding it at
#: construction time silently loses every agent event.
ListenerBox = Dict[str, Optional[Callable[[Dict[str, Any]], None]]]


def listener_box(listener: Optional[Callable[[Dict[str, Any]], None]] = None) -> ListenerBox:
    return {"listener": listener}


def make_logger(box: ListenerBox) -> LogFn:
    """Adapt the active graph listener into a ``log(event, payload)`` callable."""

    def log(event: str, payload: Mapping[str, Any]) -> None:
        target = box.get("listener")
        if target is None:
            return
        target({"type": "agent", "event": event, "ts": time.time(), **dict(payload)})

    return log


def node_header(phase: str) -> str:
    return PHASE_LABELS.get(phase, phase)


def agent_rows(results: Sequence[Any], *, phase: str, cycle: int, listener: LogFn | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "id": new_id("check"),
                "phase": phase,
                "role": getattr(result, "role", "?"),
                "lens": getattr(result, "lens", ""),
                "mode": getattr(result, "mode", "note"),
                "ok": bool(getattr(result, "ok", False)),
                "verdict": getattr(result, "verdict", "abstain"),
                "confidence": as_float((getattr(result, "data", {}) or {}).get("confidence", 0.5), 0.5),
                "note": (getattr(result, "note", lambda: "")() or getattr(result, "error", ""))[:400],
                "cycle": cycle,
                "tokens": int(getattr(result, "tokens", 0)),
                "latency_ms": float(getattr(result, "latency_ms", 0.0)),
                "fingerprint": getattr(result, "fingerprint", ""),
            }
        )
        if listener is not None:
            listener(
                "agent.done",
                {
                    "phase": phase,
                    "cycle": cycle,
                    "role": rows[-1]["role"],
                    "lens": rows[-1]["lens"],
                    "mode": rows[-1]["mode"],
                    "ok": rows[-1]["ok"],
                    "verdict": rows[-1]["verdict"],
                    "tokens": rows[-1]["tokens"],
                    "latency_ms": rows[-1]["latency_ms"],
                    "note": rows[-1]["note"][:200],
                },
            )
    return rows


def batch_issues(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """A cycle's work batch: exactly one target issue.

    The loop is scoped per issue on purpose.  Refining, reviewing, patching and
    verifying several issues at once looks parallel but is not: a review verdict
    or a test result cannot be attributed to one issue, and a failed patch stops
    the whole batch.  Other issues found in the same discovery pass stay in the
    ledger as ``open`` and become the target of a later cycle.
    """
    by_id = {str(i.get("id")): dict(i) for i in (state.get("issues") or [])}
    target = str(state.get("target_issue_id") or "")
    if target and target in by_id:
        return [by_id[target]]
    if target:
        return []
    return find_open(state.get("issues") or [])[:1]


def verified_issues(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    verified = set(state.get("batch_verified") or [])
    return [dict(i) for i in (state.get("issues") or []) if str(i.get("id")) in verified]


def fresh_batch(state: Mapping[str, Any], batch: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The parts of the batch still worth spending another cycle on."""
    verified = set(state.get("batch_verified") or [])
    out = []
    for issue in batch:
        if str(issue.get("id")) in verified:
            continue
        finding = latest_finding(state, str(issue.get("id")))
        if finding and str(finding.get("decision")) in ("reject", "defer"):
            continue
        out.append(dict(issue))
    return out


def latest_finding(state: Mapping[str, Any], issue_id: str) -> Dict[str, Any]:
    matches = [f for f in (state.get("findings") or []) if f.get("issue_id") == issue_id]
    return matches[-1] if matches else {}


def first_open(state: Mapping[str, Any]) -> Dict[str, Any]:
    issues = find_open(state.get("issues") or [])
    return issues[0] if issues else {"id": "NONE", "title": "(none)", "status": "closed"}


def tests_are_green(state: Mapping[str, Any]) -> bool:
    tests = ((state.get("evidence") or {}).get("tests") or {})
    after = tests.get("after") or {}
    return bool(tests.get("ran")) and int(after.get("failed") or 0) == 0 and int(after.get("errors") or 0) == 0
