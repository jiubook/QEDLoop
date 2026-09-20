# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Orchestration: wire the crew, the graph, tracing and reporting together.

This is the only module that knows about all the others.  Everything below it
stays a pure function of the state bus, which is what makes a run replayable
from ``trace.jsonl`` plus the recorded ``candidate/`` tree.

Layout of a run directory::

    runs/<run_id>/
        REPORT.md        human-readable report
        state.json       reportable slice of the state bus (no file contents)
        trace.jsonl      every graph and agent event, append-only
        candidate/       the exact tree Phase 5 verified (only files that changed)
        run.meta.json    pointers and the resolved policy
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import report as report_mod
from .config import load_text_file, resolve_path
from .core import Codebase, FileRecord, decode_source, diff_summary, new_id, sha_bytes, to_lf, utc_now
from .crew import Crew
from .graph import GraphError, StepLimitExceeded
from .llm import LLMProvider, make_provider, normalize_channels
from .phases import build_loop_graph, listener_box
from .policy import Policy
from .sandbox import SKIP_DIRS, parse_failing_tests, run_pytest
from .state import initial_state

DEFAULT_INCLUDE = ("*.py",)


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


def load_brief(path: Any, base: Optional[str] = None) -> Tuple[str, str]:
    """Read the per-target project constraints named by ``run.brief``.

    Returns ``(text, resolved path)``: the text rides on the bus and goes into
    every agent's system message, the path is what a report needs to say which
    rules a run worked under.  Keeping both means a reader never has to guess
    whether the file has changed since.
    """
    if not path:
        return "", ""
    source = resolve_path(path, base)
    return load_text_file(source, what="brief"), str(source)


@dataclass
class RunConfig:
    target: str = "."
    out_dir: str = "runs"
    include: Sequence[str] = DEFAULT_INCLUDE
    exclude: Sequence[str] = ()
    max_iterations: int = 3
    max_steps: int = 64
    provider: str = "auto"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    channels: Dict[str, Any] = field(default_factory=dict)
    llm_timeout: Optional[float] = None   # None = use the channel's own timeout
    brief: str = ""                       # project constraints injected into every agent
    brief_source: str = ""                # file they were read from (for the report)
    cache: bool = True
    token_budget: int = 0
    policy: Policy = field(default_factory=Policy)
    run_tests: bool = True
    test_timeout: float = 120.0
    #: Optional static check of the target repository, run against the candidate
    #: tree.  Green tests cannot see a lint rule, so without this the loop can
    #: hand back a patch that its own CI rejects.  Placeholders: ``{python}`` and
    #: ``{files}`` -- see ``sandbox.run_lint``.
    lint_command: str = ""
    keep_workspaces: bool = False
    allow_multi_cycle: bool = True
    dry_run: bool = False
    quiet: bool = False
    run_id: str = ""

    def resolved_run_id(self) -> str:
        return self.run_id or "%s-%s" % (time.strftime("%Y%m%d-%H%M%S"), new_id("run").split("-")[-1])

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, brief_base: Optional[str] = None, **overrides: Any) -> "RunConfig":
        """Build from a merged config mapping (file + CLI), ignoring unknowns.

        ``brief`` is the one field that names a *file* rather than a value: the
        mapping carries its path and this method reads the text, so the YAML
        key, the ``--brief`` flag and a hand-built ``RunConfig`` cannot drift
        into three different meanings.  ``brief_base`` is the config file's
        directory, which relative brief paths resolve against.
        """
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload: Dict[str, Any] = {
            key: value for key, value in (data or {}).items() if key in known and value is not None
        }
        policy_data = dict(data.get("gate") or data.get("policy") or {})
        for key in ("quality_target", "min_approvals", "max_self_loops", "min_iterations",
                    "plateau_epsilon", "max_no_progress", "require_tests"):
            if data.get(key) is not None:
                policy_data[key] = data[key]
        payload["policy"] = Policy.from_mapping(policy_data)
        payload["brief"], payload["brief_source"] = load_brief(data.get("brief"), brief_base)
        for key, value in overrides.items():
            if value is None or key not in known:
                continue
            payload[key] = value
        payload.setdefault("target", ".")
        payload["channels"] = normalize_channels(data.get("providers") or data.get("channels") or {})
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "out_dir": self.out_dir,
            "include": list(self.include),
            "exclude": list(self.exclude),
            "max_iterations": self.max_iterations,
            "max_steps": self.max_steps,
            "provider": self.provider,
            "model": self.model,
            "channels": sorted(self.channels),
            # The constraints themselves are not repeated here: the run state
            # carries the text, and what a reader of the report needs is which
            # file the run was governed by and how big it was.
            "brief": {"source": self.brief_source, "chars": len(self.brief)},
            "cache": self.cache,
            "token_budget": self.token_budget,
            "policy": self.policy.to_dict(),
            "run_tests": self.run_tests,
            "test_timeout": self.test_timeout,
            "lint_command": self.lint_command,
            "keep_workspaces": self.keep_workspaces,
            "allow_multi_cycle": self.allow_multi_cycle,
            "dry_run": self.dry_run,
        }


@dataclass
class RunResult:
    run_id: str
    state: Dict[str, Any]
    report: Optional[report_mod.RunReport] = None
    run_dir: Path = Path(".")
    trace_path: Optional[Path] = None
    candidate_dir: Optional[Path] = None
    exit_code: int = 0
    error: str = ""

    @property
    def status(self) -> str:
        return str(self.state.get("status", "unknown"))

    def baseline(self) -> Dict[str, str]:
        return {f["path"]: f["content"] for f in (self.state.get("codebase") or {}).get("files", [])}

    def changed_files(self) -> List[str]:
        base = self.baseline()
        working = self.state.get("working_code") or {}
        return sorted(p for p in set(base) | set(working) if base.get(p) != working.get(p))

    def diff(self) -> str:
        return diff_summary(self.baseline(), self.state.get("working_code") or {})

    def summary_lines(self) -> List[str]:
        lines = [
            "run              : %s" % self.run_id,
            "status           : %s" % self.status,
            "reason           : %s" % (self.state.get("status_reason") or ""),
            "changed files    : %s" % (", ".join(self.changed_files()) or "(none)"),
            "run directory    : %s" % self.run_dir,
        ]
        if self.report and self.report.markdown_path:
            lines.append("report           : %s" % self.report.markdown_path)
        if self.error:
            lines.append("error            : %s" % self.error)
        return lines


# --------------------------------------------------------------------------- #
# codebase loading
# --------------------------------------------------------------------------- #


def load_codebase(root: str | Path, patterns: Sequence[str], exclude: Sequence[str] = ()) -> Codebase:
    base = Path(root).resolve()
    if not base.is_dir():
        raise NotADirectoryError("target directory not found: %s" % base)
    exclude_set = [e.replace("\\", "/").strip("/") for e in exclude if e]
    files: List[FileRecord] = []
    for pattern in patterns or DEFAULT_INCLUDE:
        for path in sorted(base.rglob(pattern)):
            if not path.is_file():
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            rel = path.relative_to(base).as_posix()
            if any(rel == e or rel.startswith(e + "/") or Path(rel).match(e) for e in exclude_set):
                continue
            try:
                content = decode_source(path.read_bytes())
            except OSError:
                continue
            if content is None:
                continue
            files.append(FileRecord(path=rel, content=content))
    return Codebase(root=str(base), files=files)


# --------------------------------------------------------------------------- #
# trace
# --------------------------------------------------------------------------- #


class RunTrace:
    """Append-only JSONL of graph and agent events, optionally echoed live."""

    def __init__(self, path: Path, *, echo: bool = True) -> None:
        self.path = Path(path)
        self.echo = echo
        self.events = 0
        self.warnings: List[str] = []
        self._started = time.time()

    def __call__(self, event: Mapping[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("ts", time.time())
        payload["elapsed_s"] = round(time.time() - self._started, 3)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            self.warnings.append("trace write failed: %s" % exc)
        self.events += 1
        if self.echo:
            print(format_event(payload), flush=True)

    def __enter__(self) -> "RunTrace":
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")
        except OSError as exc:
            self.warnings.append("could not reset trace: %s" % exc)
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


#: Fields a live agent line prefers, best first.  Only the first few present are
#: printed, which keeps a line readable while still naming what changed.  The
#: order is "what a reader wants first": the verdict, the objection, then what the
#: phase produced, then the bookkeeping -- ``blocking``/``qa_rejected`` count
#: objections, and a count is all a one-line view can honestly show of a list.
_AGENT_DETAIL_ORDER = (
    "decision", "reason", "blocking", "qa_rejected", "quality", "status",
    "new_rows", "issues", "issue_ids", "issue_id", "targets", "agents", "files",
    "patch_id", "acceptance", "tests", "votes", "components", "reported",
    "deduped", "raw_reports", "note", "error",
)
_AGENT_DETAIL_LIMIT = 3
_DELTA_BRIEF_LIMIT = 4
_DETAIL_WIDTH = 64


def _clip_value(value: Any, width: int = _DETAIL_WIDTH) -> str:
    """One field, one line: sizes for collections, clipped text otherwise."""
    if isinstance(value, (list, tuple, set)):
        return str(len(value))
    text = json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, Mapping) else str(value)
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "\u2026"


def format_event(event: Mapping[str, Any]) -> str:
    """One trace event as one line a human can follow while the run is live.

    Producing this without lying took two fixes: the engine used to report a node
    only *before* it ran (so every line said ``ok 0ms -> `` regardless of what
    happened), and the generic field picker below looked for keys no ``agent.done``
    event carries (so most crew lines printed blank).
    """
    kind = str(event.get("type"))
    if kind == "node":
        node = str(event.get("node"))
        where = "step=%-2s cycle=%-2s" % (event.get("step"), event.get("cycle", 0))
        # ``duration_ms`` is the fallback for traces written before node events
        # carried an ``event`` key.
        if str(event.get("event") or "") != "done" and "duration_ms" not in event:
            return "[node ] %-9s %-6s %s" % (node, "start", where)
        delta = list((event.get("delta") or {}).items())
        brief = " ".join("%s=%s" % (key, value) for key, value in delta[:_DELTA_BRIEF_LIMIT])
        if len(delta) > _DELTA_BRIEF_LIMIT:
            brief += " (+%d more)" % (len(delta) - _DELTA_BRIEF_LIMIT)
        return "[node ] %-9s %-6s %s %-4s %7sms -> %-9s %s" % (
            node, "done", where,
            "ok" if event.get("ok", True) else "FAIL",
            int(event.get("duration_ms") or 0), event.get("next", ""), brief,
        )
    if kind == "agent":
        phase = str(event.get("phase", ""))
        name = str(event.get("event", ""))
        if name == "agent.done":
            return "[agent] %-9s %-22s %-20s %-8s %6s tok %7sms %s" % (
                phase, name, str(event.get("role", "")),
                str(event.get("verdict", "")) if event.get("ok", True) else "FAIL",
                int(event.get("tokens") or 0), int(event.get("latency_ms") or 0),
                _clip_value(event.get("note"), 80) if event.get("note") else "",
            )
        parts = []
        for key in _AGENT_DETAIL_ORDER:
            value = event.get(key)
            if value in (None, "", [], {}):
                continue
            parts.append("%s=%s" % (key, _clip_value(value)))
            if len(parts) >= _AGENT_DETAIL_LIMIT:
                break
        return "[agent] %-9s %-22s %s" % (phase, name, "  ".join(parts))
    if kind == "run":
        detail = event.get("reason") or event.get("run_id") or event.get("note") or ""
        result = event.get("result") or {}
        if not detail and result:
            detail = "%s passed, %s failed, %s errors (ran=%s)" % (
                result.get("passed", "?"), result.get("failed", "?"),
                result.get("errors", "?"), result.get("ran"),
            )
        state = str(event.get("state") or "")
        if state:
            detail = "%s: %s" % (state, detail) if detail else state
        return "[run  ] %-8s %s" % (event.get("event", ""), str(detail)[:130])
    return "[event] %s" % json.dumps({k: v for k, v in event.items() if k not in ("ts", "elapsed_s")}, default=str)[:160]


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #


def run_loop(
    config: RunConfig,
    *,
    provider: Optional[LLMProvider] = None,
    echo: Optional[bool] = None,
) -> RunResult:
    run_id = config.resolved_run_id()
    run_dir = Path(config.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "trace.jsonl"
    candidate_dir = run_dir / "candidate"

    codebase = load_codebase(config.target, config.include, config.exclude)
    if not codebase.files:
        state = {"status": "error", "status_reason": "no source files matched %s under %s" % (list(config.include), config.target)}
        return RunResult(run_id=run_id, state=state, run_dir=run_dir, exit_code=2, error="empty codebase")

    llm = provider or make_provider(
        config.provider,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        cache=config.cache,
        timeout=config.llm_timeout,
        channels=config.channels,
    )
    crew = Crew(llm, budget_tokens=config.token_budget)
    policy = config.policy

    state = initial_state(
        codebase.to_dict(),
        codebase.root,
        max_iterations=config.max_iterations,
        extra={
            "run_id": run_id,
            "policy": policy.to_dict(),
            "model_info": {
                "channel": getattr(llm, "channel", ""),
                "provider": getattr(llm, "name", ""),
                "model": getattr(llm, "model", ""),
                "base_url": getattr(llm, "base_url", "") or "",
            },
            "run_tests": config.run_tests,
            "test_timeout": config.test_timeout,
            "lint_command": config.lint_command,
            "keep_workspaces": config.keep_workspaces,
            "allow_multi_cycle": config.allow_multi_cycle,
            "brief": config.brief,
            "brief_source": config.brief_source,
            "trace_path": str(trace_path),
            "started_at": utc_now(),
        },
    )

    show = (not config.quiet) if echo is None else echo
    error = ""
    interrupted = False
    with RunTrace(trace_path, echo=show) as trace:
        # Build the phases with a live listener box: agent events then land in
        # trace.jsonl next to the node events, and one file explains a whole run.
        graph = build_loop_graph(crew, policy, box=listener_box(trace), max_steps=config.max_steps)
        state["graph_mermaid"] = graph.mermaid()
        trace({
            "type": "run", "event": "start", "run_id": run_id, "target": codebase.root,
            "files": len(codebase.files), "revision": codebase.revision,
            "provider": getattr(llm, "name", "?"), "channel": getattr(llm, "channel", "?"),
            "model": getattr(llm, "model", "?"),
            "policy": policy.to_dict(),
        })
        try:
            if config.run_tests:
                # Measuring the baseline once, before any agent runs, gives discovery
                # something honest to reason from ("the suite is green -- find what it
                # does not cover") and gives Phase 5 a like-for-like reference.
                #
                # It is announced *before* it starts, and it happens inside the
                # trace, because copying a repository and running its whole suite is
                # minutes of silence on a real target -- and silence is what makes
                # an operator reach for Ctrl+C on a run that is working fine.
                trace({"type": "run", "event": "baseline", "state": "measuring",
                       "note": "copying the target repository and running its suite"})
                state["baseline_tests"] = measure_baseline(codebase, config)
                trace({"type": "run", "event": "baseline", "state": "done",
                       "result": state["baseline_tests"]})
            state = graph.run(state, listener=trace)
        except KeyboardInterrupt:
            # Ctrl+C is the operator's stop button, and it is not an error: it is
            # a decision.  Letting it unwind would leave a run directory with a
            # trace and nothing else -- no ledger, no report, no way to see how
            # far it got, which is the worst moment to lose the record.  Recording
            # it and falling through to the normal artifact writing costs a few
            # seconds and keeps the run auditable.
            interrupted = True
            error = "KeyboardInterrupt: interrupted by the operator"
            state["status"] = "human_review"
            state["status_reason"] = "interrupted by the operator after %d node(s); %s" % (
                sum(int(v) for v in (state.get("node_visits") or {}).values()),
                str(state.get("status_reason") or "") or "nothing was verified",
            )
        except (StepLimitExceeded, GraphError) as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
            state["status"] = "escalated"
            state["status_reason"] = error
        except Exception as exc:  # keep the partial run for diagnosis
            error = "%s: %s" % (type(exc).__name__, exc)
            state["status"] = "error"
            state["status_reason"] = error
        state["agent_usage"] = usage_of(crew)
        state["finished_at"] = utc_now()
        trace({
            "type": "run", "event": "finish", "run_id": run_id, "status": state.get("status"),
            "reason": state.get("status_reason"), "usage": state["agent_usage"], "error": error,
        })

    result = RunResult(run_id=run_id, state=state, run_dir=run_dir, trace_path=trace_path,
                       candidate_dir=candidate_dir, error=error)
    result.report = report_mod.write_report(run_dir, state, config=config.to_dict())
    if not interrupted:
        keep_candidate(result)
    # An interrupted run deliberately gets no candidate tree: ``apply`` keys off
    # candidate.manifest.json, so the operator's stop button cannot leave behind
    # a tree that ``apply --allow-unverified`` would write onto the target.
    # Being verified is what makes a candidate appliable, and an interrupt is
    # exactly the case where nothing was verified.
    result.exit_code = {"converged": 0, "continue": 0, "human_review": 1, "escalated": 3, "error": 4}.get(
        str(state.get("status")), 1
    )

    if not config.quiet:
        print("")
        for line in result.report.summary_lines() if result.report else result.summary_lines():
            print(line)
    return result


def measure_baseline(codebase: Codebase, config: "RunConfig") -> Dict[str, Any]:
    """Run the target's suite once, against the untouched repository."""
    report = run_pytest(
        codebase.as_map(),
        target_root=codebase.root,
        timeout=config.test_timeout,
        keep=config.keep_workspaces,
    )
    return {
        "passed": report.passed,
        "failed": report.failed,
        "errors": report.errors,
        "skipped": report.skipped,
        "total": report.total,
        "green": report.green,
        "ran": report.ran,
        "failing": parse_failing_tests(report.stdout + "\n" + report.stderr)[:25],
        "error": report.error,
        "duration_s": round(report.duration_s, 2),
    }


def usage_of(crew: Crew) -> Dict[str, Any]:
    per_role: Dict[str, Dict[str, int]] = {}
    for result in crew.results:
        row = per_role.setdefault(result.role, {"calls": 0, "tokens": 0, "failed": 0})
        row["calls"] += 1
        row["tokens"] += int(result.tokens)
        row["failed"] += 0 if result.ok else 1
    return {
        "calls": crew.calls,
        "tokens": crew.tokens_used,
        # Real split, not "all of it was the prompt": input and output tokens are
        # priced differently by every vendor, and a run's cost is unreadable
        # without it.
        "prompt_tokens": crew.prompt_tokens_used,
        "completion_tokens": crew.completion_tokens_used,
        "per_role": per_role,
        "agents_run": len(crew.results),
        "agents_failed": len([r for r in crew.results if not r.ok]),
    }


# --------------------------------------------------------------------------- #
# run directory artifacts
# --------------------------------------------------------------------------- #


def detect_newline(data: bytes) -> str:
    """The line-ending convention a file already uses on disk.

    CRLF wins when both appear: a file that is mostly CRLF with a stray LF is
    still a CRLF file, and rewriting it the other way would touch every line.
    """
    crlf = data.count(b"\r\n")
    lone_lf = data.count(b"\n") - crlf
    if crlf and crlf >= lone_lf:
        return "\r\n"
    if lone_lf:
        return "\n"
    return "\r\n" if crlf else "\n"


def with_newline(text: str, convention: str) -> str:
    """Re-express ``text`` (LF) using ``convention`` without doubling carriage returns.

    Normalises first even for the LF case: a caller handing in text that still
    carries CRLF should get LF back, not its own input unchanged.
    """
    if convention == "\n":
        return to_lf(text)
    return to_lf(text).replace("\n", "\r\n")


def keep_candidate(result: RunResult) -> List[str]:
    """Persist the exact file contents Phase 5 verified, so ``apply`` never
    has to re-invent a patch from model text."""
    base = result.baseline()
    working = result.state.get("working_code") or {}
    changed = [p for p in sorted(set(base) | set(working)) if base.get(p) != working.get(p)]
    if not changed:
        return []
    root = Path(result.run_dir) / "candidate"
    for rel in changed:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        # Kept as the LF form on purpose -- ``baseline_sha`` below and
        # ``apply_run``'s comparison are both taken over it -- and written with
        # ``newline=""`` so the write translates nothing.  The body carries the
        # target's convention (CRLF on a CRLF checkout) and ``write_text``'s
        # default translation turned every "\r\n" into "\r\r\n", which reads
        # back as an extra blank line between every line.
        target.write_text(to_lf(str(working.get(rel, ""))), encoding="utf-8", newline="")
    manifest = {
        "run_id": result.run_id,
        "status": result.status,
        "candidate_sha": sha_bytes("".join(sorted(str(working.get(p, "")) for p in changed)).encode("utf-8")),
        "verified": result.status == "converged",
        "files": changed,
        # Line endings are content to a byte comparison and noise to a reader, so
        # every recorded baseline is the LF form of the text.  ``apply_run``
        # compares the same way; otherwise a CRLF checkout would fail this check
        # for every file and the run would refuse to write anything.
        "baseline_sha": {rel: sha_bytes(to_lf(str(base.get(rel, ""))).encode("utf-8")) for rel in changed},
    }
    (Path(result.run_dir) / "candidate.manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # The diff is the only place a human sees the patch before ``apply`` writes
    # it, so it is normalised to LF too.  ``diff_summary`` splits text that
    # carries the target's CRLF, so its hunk bodies end CRLF while difflib's
    # ``@@`` headers end LF -- mixed endings that the default translation then
    # doubled on one side only.  Written unconditionally: a dry run needs the
    # diff to review, and ``apply`` is refused for a run that did not converge.
    (Path(result.run_dir) / "candidate.diff").write_text(
        to_lf(result.diff()), encoding="utf-8", newline=""
    )
    return changed


def load_run(run_dir: str | Path) -> Dict[str, Any]:
    path = Path(run_dir) / "state.json"
    if not path.is_file():
        raise FileNotFoundError("no state.json in %s (pass the run directory, not its parent)" % run_dir)
    return json.loads(path.read_text(encoding="utf-8"))


def apply_run(
    run_dir: str | Path,
    *,
    target: Optional[str] = None,
    dry_run: bool = False,
    backup: bool = True,
    allow_unverified: bool = False,
) -> Dict[str, Any]:
    """Write the verified candidate tree back onto the target repository."""
    state = load_run(run_dir)
    root = Path(target or state.get("target_root") or ".").resolve()
    candidate_root = Path(run_dir) / "candidate"
    manifest_path = Path(run_dir) / "candidate.manifest.json"

    if not candidate_root.is_dir() or not manifest_path.is_file():
        return {"ok": False, "applied": [], "skipped": [], "dry_run": dry_run,
                "reason": "this run produced no candidate tree"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("verified") and not allow_unverified:
        return {
            "ok": False,
            "applied": [],
            "skipped": [],
            "dry_run": dry_run,
            "reason": "the run did not converge (status=%s); re-run with allow_unverified to apply anyway" % manifest.get("status"),
        }

    applied: List[str] = []
    skipped: List[Dict[str, str]] = []
    for rel in manifest.get("files") or []:
        source = candidate_root / rel
        destination = root / rel
        if not source.is_file():
            skipped.append({"path": rel, "reason": "missing from the candidate tree"})
            continue
        if not destination.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            new_text = source.read_text(encoding="utf-8")
            if not dry_run:
                # Deliberately *without* ``newline=""``, unlike the update branch
                # below.  The candidate tree is kept in LF, and a brand-new file
                # has no destination convention to inherit, so the platform
                # default is the right guess: CRLF on Windows, which is what the
                # CRLF checkouts this runs against expect.  Passing "" here
                # would drop an LF file into a CRLF repository.
                destination.write_text(new_text, encoding="utf-8")
            applied.append(rel)
            continue

        # Read and write with the *destination's* line endings.  ``read_text``
        # defaults to universal newlines and ``write_text`` to the platform
        # default, which on Windows silently rewrites a CRLF file as LF: git then
        # reports it modified with an empty diff.  Measured on a CRLF target with
        # ``core.autocrlf=true`` -- the file showed as ``M`` while ``git diff
        # --stat`` was empty.  The comparison is newline-insensitive because the
        # candidate tree is produced in LF (``keep_candidate`` writes it that
        # way), while ``baseline_sha`` is taken over the same LF text.
        convention = detect_newline(destination.read_bytes())
        new_text = source.read_text(encoding="utf-8")
        old_text = destination.read_text(encoding="utf-8", newline=convention)
        expected = (manifest.get("baseline_sha") or {}).get(rel)
        if expected and sha_bytes(to_lf(old_text).encode("utf-8")) != expected and not dry_run:
            skipped.append({"path": rel, "reason": "target changed since the run; not overwriting"})
            continue
        if new_text == to_lf(old_text):
            skipped.append({"path": rel, "reason": "already identical"})
            continue
        if not dry_run:
            if backup and destination.is_file():
                backup_path = destination.with_suffix(destination.suffix + ".qedloop.bak")
                backup_path.write_text(old_text, encoding="utf-8", newline="")
            destination.write_text(with_newline(new_text, convention), encoding="utf-8", newline="")
        applied.append(rel)

    return {
        "ok": bool(applied),
        "applied": applied,
        "skipped": skipped,
        "dry_run": dry_run,
        "target": str(root),
        "run_dir": str(run_dir),
        "status": state.get("status"),
    }
