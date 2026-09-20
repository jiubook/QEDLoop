# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Crew runtime: fan out one agent per lens, collect structured answers.

This is the "3 Agents" box in the architecture diagram.  Everything here is
deliberately provider-agnostic and side-effect free: the crew calls the LLM,
parses JSON, and returns normalised payloads.  It never mutates the state bus --
that is the job of the phase nodes.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from .core import Finding, Issue, Review, TestPlan, as_float, new_id, to_dicts, utc_now
from .llm import LLMError, LLMProvider, LLMReply, Message
from .prompts import AgentSpec, build_messages, find_spec, specs_for

# --------------------------------------------------------------------------- #
# tolerant JSON extraction
# --------------------------------------------------------------------------- #

FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(?P<body>.*?)```", re.S)


class AgentOutputError(RuntimeError):
    """The agent replied with something that is not the agreed JSON contract."""


def strip_fences(text: str) -> str:
    match = FENCE_RE.search(text or "")
    if match:
        return match.group("body").strip()
    return (text or "").strip()


def iter_balanced_objects(text: str) -> Iterable[str]:
    """Yield every top-level ``{...}`` region, ignoring braces inside strings."""
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start : i + 1]
                    start = -1


def extract_json(text: str) -> Dict[str, Any]:
    """Best-effort JSON object extraction from an LLM reply.

    Order: raw parse -> fenced block -> first balanced object -> outer braces.
    Raises :class:`AgentOutputError` with a clipped copy of the reply so the
    failure is debuggable from the trace alone.
    """
    cleaned = strip_fences(text)
    candidates: List[str] = []
    if cleaned:
        candidates.append(cleaned)
    for candidate in (text or "",):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    for candidate in list(candidates):
        candidates.extend(iter_balanced_objects(candidate))

    for candidate in candidates:
        candidate = candidate.strip().rstrip(",")
        if not candidate.startswith("{"):
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise AgentOutputError("no JSON object in reply: %s" % (text or "")[:400].replace("\n", " "))


# --------------------------------------------------------------------------- #
# agent result
# --------------------------------------------------------------------------- #


@dataclass
class AgentResult:
    role: str
    phase: str
    lens: str
    mode: str
    ok: bool = True
    data: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    raw: str = ""
    fingerprint: str = ""
    model: str = ""
    tokens: int = 0
    latency_ms: float = 0.0
    attempts: int = 1
    cached: bool = False

    @property
    def verdict(self) -> str:
        return str(self.data.get("verdict", "abstain")).lower()

    def note(self) -> str:
        return str(self.data.get("notes") or self.data.get("observations") or self.data.get("summary") or "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "phase": self.phase,
            "lens": self.lens,
            "mode": self.mode,
            "ok": self.ok,
            "error": self.error,
            "verdict": self.verdict,
            "fingerprint": self.fingerprint,
            "model": self.model,
            "tokens": self.tokens,
            "latency_ms": self.latency_ms,
            "attempts": self.attempts,
            "cached": self.cached,
            "data": self.data,
        }


@dataclass
class AgentTask:
    spec: AgentSpec
    state: Mapping[str, Any]
    ctx: Dict[str, Any] = field(default_factory=dict)
    optional: bool = False


# --------------------------------------------------------------------------- #
# crew
# --------------------------------------------------------------------------- #


class Crew:
    """Runs agent specs against one provider and normalises their answers."""

    def __init__(self, provider: LLMProvider, *, budget_tokens: int = 0, max_retries: int = 2) -> None:
        self.provider = provider
        self.budget_tokens = int(budget_tokens)
        self.max_retries = int(max_retries)
        self.replies: Dict[str, LLMReply] = {}
        self.tokens_used = 0
        # Kept apart because they are priced apart -- output tokens cost several
        # times what a cached prompt token does, so a report that lumps them
        # together cannot be used to estimate a run.
        self.prompt_tokens_used = 0
        self.completion_tokens_used = 0
        self.calls = 0
        self.results: List[AgentResult] = []

    # -- single agent ------------------------------------------------------ #
    def output_budget(self, spec: AgentSpec) -> int:
        """Tokens to allow for one answer.

        A channel that declares ``max_tokens`` is describing its model's output
        ceiling, and that ceiling wins: a thinking model spends part of it on a
        chain of thought that is billed as output and counted against the same
        limit, so the framework's small JSON-answer budget is not enough on its
        own -- the reply gets cut off at ``finish_reason: length`` and the JSON
        never arrives.  A channel that says nothing leaves the decision to the
        agent's own contract, which is the right default for local models whose
        context window is small.
        """
        declared = getattr(self.provider, "max_tokens", None)
        return int(declared) if declared else int(spec.max_tokens)

    def run_one(self, task: AgentTask) -> AgentResult:
        spec = task.spec
        messages = build_messages(spec, task.state, **task.ctx)
        result = AgentResult(role=spec.role, phase=spec.phase, lens=spec.lens, mode=spec.mode)
        last_error = ""
        for attempt in range(1, self.max_retries + 2):
            result.attempts = attempt
            try:
                reply = self.provider.complete(
                    messages,
                    temperature=spec.temperature,
                    max_tokens=self.output_budget(spec),
                )
            except LLMError as exc:
                last_error = "provider error: %s" % exc
                if attempt > self.max_retries:
                    return self._fail(result, last_error, task.optional)
                continue
            self.calls += 1
            self.tokens_used += reply.total_tokens
            self.prompt_tokens_used += int(reply.prompt_tokens)
            self.completion_tokens_used += int(reply.completion_tokens)
            result.raw = reply.text
            result.fingerprint = reply.fingerprint
            result.model = reply.model
            result.tokens += reply.total_tokens
            result.latency_ms = reply.latency_ms
            result.cached = reply.cached
            self.replies[spec.role] = reply
            try:
                result.data = extract_json(reply.text)
                result.ok = True
                self.results.append(result)
                return result
            except AgentOutputError as exc:
                last_error = str(exc)
                if str((reply.meta or {}).get("finish_reason") or "") == "length":
                    # "not a single JSON object" is the symptom; the cause is a
                    # budget the caller can fix, and it is invisible otherwise.
                    last_error += (
                        " (the reply hit max_tokens=%d and was cut off -- raise the channel's max_tokens; "
                        "a thinking model's reasoning tokens count against it)"
                        % self.output_budget(spec)
                    )
                # one repair round-trip: tell the agent what was wrong
                if attempt <= self.max_retries:
                    messages = list(messages) + [
                        Message("assistant", reply.text[:2000]),
                        Message(
                            "user",
                            "That reply was not a single JSON object (%s). "
                            "Return ONLY the JSON object, no prose, no markdown fences." % last_error[:200],
                        ),
                    ]
                    continue
        return self._fail(result, last_error, task.optional)

    def _fail(self, result: AgentResult, error: str, optional: bool) -> AgentResult:
        result.ok = False
        result.error = error
        result.data = {}
        if not optional:
            self.results.append(result)
        return result

    # -- one phase --------------------------------------------------------- #
    def run_phase(
        self,
        phase: str,
        state: Mapping[str, Any],
        *,
        roles: Optional[Sequence[str]] = None,
        **ctx: Any
    ) -> List[AgentResult]:
        """Run one phase's agents.  ``roles`` narrows the fan-out when a lens is
        not part of it -- the patch phase's reconcile lens is a tool its node
        calls with real candidates, not a peer of the generators."""
        specs = [s for s in specs_for(phase) if roles is None or s.role in roles]
        return self.run_tasks([AgentTask(spec, state, ctx) for spec in specs])

    def run_tasks(self, tasks: Sequence[AgentTask]) -> List[AgentResult]:
        if not tasks:
            return []
        if self.budget_tokens and self.tokens_used >= self.budget_tokens:
            return [
                AgentResult(
                    role=t.spec.role, phase=t.spec.phase, lens=t.spec.lens, mode=t.spec.mode,
                    ok=False, error="token budget exhausted (%d tokens)" % self.tokens_used,
                )
                for t in tasks
            ]
        workers = max(1, min(getattr(self.provider, "concurrency", 4), len(tasks)))
        if workers == 1:
            return [self.run_one(t) for t in tasks]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # submit in spec order; map() preserves it in the result
            return list(pool.map(self.run_one, tasks))

    # -- normalisers ------------------------------------------------------- #
    def issues(self, results: Sequence[AgentResult], *, cycle: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for result in results:
            for raw in result.data.get("issues") or []:
                if not isinstance(raw, Mapping):
                    continue
                payload = dict(raw)
                payload["lens"] = result.lens
                payload["cycle"] = cycle
                issue = Issue.from_dict(payload)
                if not issue.title:
                    continue
                if issue.confidence < 0.35:
                    continue  # below the reporting floor
                out.append(issue.to_dict())
        return out

    def findings(self, results: Sequence[AgentResult], *, issue_id: str, cycle: int) -> List[Dict[str, Any]]:
        synthesis = _first(results, "refine_synthesize")
        plan = _first(results, "plan_split")
        tests = _first(results, "test_design")
        if not synthesis and not plan and not tests:
            return []
        finding = Finding(
            id=new_id("finding"),
            issue_id=issue_id,
            decision=str((synthesis.data if synthesis else {}).get("decision", "act")).lower(),
            summary=str((synthesis.data if synthesis else {}).get("summary", "")),
            acceptance=[str(a) for a in ((synthesis.data if synthesis else {}).get("acceptance") or [])],
            non_goals=[str(a) for a in ((synthesis.data if synthesis else {}).get("non_goals") or [])],
            # Carried on the finding rather than only in the synthesis reply: the
            # review reads findings, and it rejected three rounds running for a
            # plan that could not say how the change would be shown to be the
            # thing making the test pass.
            falsification=str((synthesis.data if synthesis else {}).get("falsification", "")),
            blast_radius=str((synthesis.data if synthesis else {}).get("blast_radius", "module")),
            risk=str((synthesis.data if synthesis else {}).get("risk", "medium")),
            refined_at=utc_now(),
        )
        data = finding.to_dict()
        data["steps"] = list((plan.data if plan else {}).get("steps") or [])
        data["estimate"] = str((plan.data if plan else {}).get("estimate", ""))
        data["rollback"] = str((plan.data if plan else {}).get("rollback", ""))
        data["cycle"] = cycle
        return [data]

    def test_plans(
        self,
        results: Sequence[AgentResult],
        *,
        issue_id: str,
        refined_at: str = "",
    ) -> List[Dict[str, Any]]:
        """Normalise the designed tests.

        ``refined_at`` is the stamp of the finding this round produced.  It rides
        on every plan because the ledger retires a previous generation by
        comparing stamps -- plans carry fresh ids each round, so without it the
        reducer cannot tell a new generation from another entry in the same one.
        """
        design = _first(results, "test_design")
        out: List[Dict[str, Any]] = []
        for raw in (design.data.get("tests") if design else []) or []:
            if not isinstance(raw, Mapping):
                continue
            plan = TestPlan(
                id=new_id("plan"),
                issue_id=issue_id,
                covers=[issue_id],
                kind=str(raw.get("kind", "unit")),
                name=str(raw.get("name", "")),
                path=str(raw.get("path", "")),
                given=str(raw.get("given", "")),
                when=str(raw.get("when", "")),
                then=str(raw.get("then", "")),
                regression_of=str(raw.get("regression_of", "")),
            )
            record = plan.to_dict()
            if refined_at:
                record["refined_at"] = refined_at
            out.append(record)
        return out

    def reviews(self, results: Sequence[AgentResult], *, cycle: int, issue_id: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for result in results:
            review = Review(
                id=new_id("review"),
                issue_id=issue_id,
                lens=result.lens,
                verdict=result.verdict if result.ok else "abstain",
                confidence=as_float(result.data.get("confidence"), 0.5),
                blocking=_normalise_blocking(result.data.get("blocking"), issue_id),
                notes=result.note() or result.error,
                cycle=cycle,
            )
            out.append(review.to_dict())
        return out

    def votes(self, results: Sequence[AgentResult]) -> Dict[str, str]:
        return {r.role: (r.verdict if r.ok else "abstain") for r in results}

    def checks(self, results: Sequence[AgentResult], *, phase: str, cycle: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for result in results:
            rows.append(
                {
                    "id": new_id("check"),
                    "phase": phase,
                    "role": result.role,
                    "lens": result.lens,
                    "mode": result.mode,
                    "ok": result.ok,
                    "verdict": result.verdict if result.ok else "error",
                    "confidence": as_float(result.data.get("confidence"), 0.5),
                    "note": result.note() or result.error,
                    "cycle": cycle,
                    "tokens": result.tokens,
                    "latency_ms": result.latency_ms,
                    "fingerprint": result.fingerprint,
                }
            )
        return rows


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _first(results: Sequence[AgentResult], role: str) -> Optional[AgentResult]:
    for result in results:
        if result.role == role and result.ok:
            return result
    return None


def _normalise_blocking(raw: Any, issue_id: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if isinstance(raw, str):
        raw = [{"reason": raw}]
    for item in raw or []:
        if isinstance(item, Mapping):
            out.append({"issue_id": str(item.get("issue_id") or issue_id), "reason": str(item.get("reason") or item)})
        else:
            out.append({"issue_id": issue_id, "reason": str(item)})
    return out


def summary_table(results: Sequence[AgentResult]) -> str:
    """Compact console/report view: one row per agent."""
    lines = ["| agent | mode | ok | verdict | tokens | latency | note |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        note = (r.note() or r.error or "").replace("|", "/").replace("\n", " ")
        lines.append(
            "| %s | %s | %s | %s | %d | %.0fms | %s |"
            % (r.role, r.mode, "yes" if r.ok else "no", r.verdict, r.tokens, r.latency_ms, note[:120])
        )
    return "\n".join(lines)


def agent_catalog() -> List[Dict[str, Any]]:
    from .prompts import all_specs

    return [
        {
            "role": spec.role,
            "phase": spec.phase,
            "lens": spec.lens,
            "mode": spec.mode,
            "temperature": spec.temperature,
            "description": spec.description,
        }
        for spec in all_specs()
    ]
