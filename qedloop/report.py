# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Run reporting: a JSON state dump plus a human-readable Markdown report.

The Markdown report is the artifact a reviewer reads, so it leads with the
decision and the evidence, and only then shows the ledgers.  No live objects are
serialised -- everything that reaches disk goes through the explicit state view
built here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .core import clip, metrics_of, recap, tally, to_lf, utc_now
from .graph import END
from .phases import LOOP_EDGES
from .policy import Policy

# --------------------------------------------------------------------------- #
# what a report contains
# --------------------------------------------------------------------------- #

STATE_LEDGERS = (
    "issues", "findings", "test_plans", "reviews", "patches", "changes",
    "checks", "verifications", "qa_votes", "cycles", "quality_history",
)

STATE_SCALARS = ("model_info", "baseline_tests")


def collect_changed_files(state: Mapping[str, Any]) -> List[str]:
    """Files whose content the run changed, derived from the recorded patches.

    ``state.json`` deliberately does not carry file contents, so the report and
    the CLI derive the changed set from the change ledger rather than from the
    candidate tree.
    """
    changed: List[str] = []
    for change in state.get("changes") or []:
        if not change.get("ok"):
            continue
        for path in change.get("files") or []:
            if path not in changed:
                changed.append(str(path))
    return sorted(changed)


def state_view(state: Mapping[str, Any], *, include_code: bool = False) -> Dict[str, Any]:
    """Explicitly pick the reportable slice of the run state."""
    codebase = state.get("codebase") or {}
    view: Dict[str, Any] = {
        "run_id": state.get("run_id"),
        "target_root": state.get("target_root"),
        "status": state.get("status"),
        "status_reason": state.get("status_reason"),
        "cycle": state.get("cycle"),
        "max_iterations": state.get("max_iterations"),
        "self_loop_count": state.get("self_loop_count"),
        "node_visits": state.get("node_visits") or {},
        "policy": (state.get("policy") or {}),
        "codebase": {
            "root": codebase.get("root"),
            "revision": codebase.get("revision"),
            "files": [
                {"path": f.get("path"), "lines": f.get("lines"), "sha": f.get("sha"), "is_test": f.get("is_test")}
                for f in codebase.get("files", [])
            ],
        },
        "evidence": state.get("evidence") or {},
        "baseline_tests": state.get("baseline_tests") or {},
        "agent_usage": state.get("agent_usage") or {},
        "model": state.get("model_info") or {},
        # Which rules governed the run, not the rules themselves: the text is on
        # the bus and in every prompt, and state.json stays readable.
        "brief": {
            "source": state.get("brief_source") or "",
            "chars": len(str(state.get("brief") or "")),
        },
        "trace_path": state.get("trace_path"),
        "changed_files": collect_changed_files(state),
    }
    for name in STATE_LEDGERS:
        view[name] = list(state.get(name) or [])
    for name in STATE_SCALARS:
        view[name] = state.get(name) or {}
    if include_code:
        view["working_code"] = dict(state.get("working_code") or {})
    return view


@dataclass
class RunReport:
    state: Dict[str, Any]
    config: Dict[str, Any] = field(default_factory=dict)
    markdown_path: Optional[Path] = None
    json_path: Optional[Path] = None

    @property
    def status(self) -> str:
        return str(self.state.get("status", "unknown"))

    def summary_lines(self) -> List[str]:
        history = self.state.get("quality_history") or []
        last = history[-1] if history else {}
        lines = [
            "status           : %s" % self.status,
            "reason           : %s" % (self.state.get("status_reason") or ""),
            "cycles run       : %s" % (len(history) or self.state.get("cycle", 0)),
            "final quality    : %s (target %s)" % (
                (last.get("quality") if last else None),
                (self.state.get("policy") or {}).get("quality_target"),
            ),
            "issues           : %s" % tally(self.state.get("issues") or [], "status"),
        ]
        # A QA objection blocks convergence and withholds the verified credit, so
        # it has to be visible without opening the JSON: the run that motivated
        # this reported a full score and three verify credits while every QA lens
        # was rejecting the patch.
        rejected = sorted(
            str(role)
            for role, verdict in (last.get("qa_verdicts") or {}).items()
            if str(verdict) == "reject"
        )
        if rejected:
            lines.append("qa objections    : %s rejected the candidate tree" % ", ".join(rejected))
        lines += [
            "changed files    : %d" % len(collect_changed_files(self.state)),
            "agent calls      : %s" % (self.state.get("agent_usage") or {}).get("calls", 0),
            "tokens used      : %s" % (self.state.get("agent_usage") or {}).get("tokens", 0),
            "markdown         : %s" % (self.markdown_path or "-"),
            "json             : %s" % (self.json_path or "-"),
        ]
        return lines

    def to_markdown(self) -> str:
        state = self.state
        codebase = state.get("codebase") or {}
        evidence = state.get("evidence") or {}
        history = state.get("quality_history") or []
        issues = state.get("issues") or []
        changes = state.get("changes") or []
        usage = state.get("agent_usage") or {}
        policy = Policy.from_mapping(state.get("policy") or {})

        out: List[str] = []
        add = out.append

        add("# Self-iterating development loop -- run report")
        add("")
        add("**Run** `%s`  |  **Target** `%s`  |  **Status** `%s`" % (state.get("run_id"), state.get("target_root"), self.status))
        add("")
        add("> %s" % (state.get("status_reason") or "no reason recorded"))
        add("")

        # --- executive summary -------------------------------------------- #
        add("## 1. Summary")
        add("")
        add("| metric | value |")
        add("| --- | --- |")
        add("| base revision | `%s` |" % codebase.get("revision"))
        add("| files iterated | %d |" % len(codebase.get("files") or []))
        model = state.get("model_info") or {}
        if model:
            add("| model channel | `%s` (%s / `%s`) |" % (
                model.get("channel") or model.get("provider") or "-",
                model.get("provider") or "-",
                model.get("model") or "-",
            ))
            if model.get("base_url"):
                add("| endpoint | `%s` |" % model["base_url"])
        baseline = state.get("baseline_tests") or {}
        if baseline:
            add("| baseline suite | %s (%s passed, %s failed, %s errors) |" % (
                "green" if baseline.get("green") else "RED",
                baseline.get("passed"), baseline.get("failed"), baseline.get("errors"),
            ))
        if state.get("brief_source"):
            add("| project constraints | `%s` (%d chars) |" % (
                state["brief_source"], len(str(state.get("brief") or "")),
            ))
        add("| cycles completed | %d of %s |" % (len(history), state.get("max_iterations")))
        add("| review->refine loops | %s |" % state.get("self_loop_count"))
        add("| issues found | %d (%s) |" % (len(issues), tally(issues, "severity")))
        add("| issue status | %s |" % tally(issues, "status"))
        last_row = history[-1] if history else {}
        qa_rejects = sorted(
            str(role)
            for role, verdict in (last_row.get("qa_verdicts") or {}).items()
            if str(verdict) == "reject"
        )
        if qa_rejects:
            # Both a blocker and a withhold of the verified credit, so it belongs
            # in the summary table rather than only in the raw ledger.
            add("| **QA objections** | **%s rejected the candidate tree** |" % ", ".join(qa_rejects))
        add("| patches proposed | %d |" % len(state.get("patches") or []))
        add("| changes applied to candidate tree | %d |" % len(changes))
        add("| agent calls | %s |" % usage.get("calls", 0))
        if int(usage.get("agents_failed") or 0):
            # A run whose crew could not answer must say so in the summary, not
            # only in the phase tables: "0 issues found" and "0 agents replied"
            # look identical from the outside otherwise.
            add("| **agents failed** | **%s of %s** (see the phase tables for the errors) |" % (
                usage.get("agents_failed"), usage.get("agents_run") or usage.get("agents_failed"),
            ))
        add("| tokens | %s (prompt %s / completion %s) |" % (
            usage.get("tokens", 0), usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)))
        add("")
        if history:
            add("### Quality per cycle")
            add("")
            # Two weighted components first, then the three hard requirements:
            # the table should show which numbers produced the score and which
            # conditions merely had to hold.  Reviewer verdicts are requirements,
            # not points -- the votes themselves are listed per round in section 3
            # and per agent in section 6.
            add("| cycle | quality | tests | static | compiles | suite green | review ok | status | reason |")
            add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for row in history:
                components = row.get("components") or {}
                add("| %s | %.2f | %.2f | %.2f | %s | %s | %s | %s | %s |" % (
                    row.get("cycle"),
                    float(row.get("quality") or 0.0),
                    float(components.get("tests") or 0.0),
                    float(components.get("static") or 0.0),
                    "yes" if row.get("compiles") else "no",
                    "yes" if row.get("tests_green") else "no",
                    "yes" if row.get("review_ok") else "no",
                    row.get("status"),
                    str(row.get("reason") or "").replace("|", "/")[:90],
                ))
            add("")

        # --- defects ------------------------------------------------------- #
        add("## 2. Issue ledger")
        add("")
        if issues:
            add("| id | severity | status | lens | title | files | evidence |")
            add("| --- | --- | --- | --- | --- | --- | --- |")
            for issue in issues:
                add("| `%s` | %s | %s | %s | %s | %s | %s |" % (
                    issue.get("id"),
                    issue.get("severity"),
                    issue.get("status"),
                    issue.get("lens"),
                    str(issue.get("title") or "").replace("|", "/"),
                    ", ".join("`%s`" % f for f in (issue.get("files") or [])[:2]) or "-",
                    clip(str(issue.get("evidence") or "").replace("\n", " ").replace("|", "/"), 90),
                ))
        else:
            add("_No issue was reported in this run._")
        add("")

        # --- per cycle ----------------------------------------------------- #
        add("## 3. What happened, phase by phase")
        add("")
        reviews = state.get("reviews") or []
        checks = state.get("checks") or []
        for cycle in sorted({int(r.get("cycle") or 0) for r in checks} | {0}):
            add("### Cycle %d" % cycle)
            add("")
            rows = [r for r in reviews if int(r.get("cycle") or 0) == cycle]
            if rows:
                add("Phase 3 review votes: %s" % ", ".join("`%s=%s`" % (r.get("lens"), r.get("verdict")) for r in rows))
                blocking = [b for r in rows for b in (r.get("blocking") or [])]
                if blocking:
                    add("")
                    add("Blocking findings raised: %d" % len(blocking))
                    for item in blocking[:5]:
                        add("- `%s` %s" % (item.get("lens"), clip(str(item.get("reason") or ""), 200)))
                add("")
            gate = [c for c in checks if int(c.get("cycle") or 0) == cycle and c.get("role") == "gate"]
            for row in gate:
                add("- gate `%s` -> **%s** (%s)" % (row.get("phase"), row.get("verdict"), clip(str(row.get("note") or ""), 160)))
            add("")

        # --- changes ------------------------------------------------------- #
        add("## 4. Candidate changes")
        add("")
        if changes:
            for change in changes:
                add("### `%s` -- %s" % (change.get("patch_id"), change.get("issue_id")))
                add("")
                add("- strategy: `%s`" % change.get("strategy"))
                add("- files: %s" % (", ".join("`%s`" % f for f in change.get("files") or []) or "-"))
                add("- applied cleanly: %s" % ("yes" if change.get("ok") else "no"))
                if change.get("errors"):
                    add("- apply errors: %s" % "; ".join(str(e) for e in change["errors"][:4]))
                add("- expected effect: %s" % (change.get("expected_effect") or "-"))
                add("")
                if change.get("diff"):
                    add("```diff")
                    add(clip(str(change["diff"]), 4000).rstrip())
                    add("```")
                    add("")
        else:
            add("_No patch was produced._")
            add("")

        # --- evidence ------------------------------------------------------ #
        add("## 5. Verification evidence (measured, not asserted)")
        add("")
        static = evidence.get("static") or {}
        tests = evidence.get("tests") or {}
        if static:
            add("- compile: %s (%d files)" % ("ok" if static.get("compiles") else "FAILED", static.get("compiled_files") or 0))
            add("- declared defect markers: %s before -> %s after" % (static.get("markers_before"), static.get("markers_after")))
            for marker in static.get("markers_resolved") or []:
                add("  - resolved `%s` (%s:%s)" % (marker.get("name"), marker.get("path"), marker.get("line")))
            for marker in static.get("markers_introduced") or []:
                add("  - **introduced** `%s` (%s:%s)" % (marker.get("name"), marker.get("path"), marker.get("line")))
        if tests:
            before = tests.get("before") or {}
            after = tests.get("after") or {}
            regression = tests.get("regression") or {}
            add("")
            add("| suite | passed | failed | errors | total | green |")
            add("| --- | --- | --- | --- | --- | --- |")
            for label, row in (("before (baseline)", before), ("after (candidate)", after)):
                add("| %s | %s | %s | %s | %s | %s |" % (
                    label, row.get("passed", 0), row.get("failed", 0), row.get("errors", 0), row.get("total", 0),
                    "yes" if row.get("green") else "no"))
            add("")
            add("- fixed failures: %s  |  new failures: %s  |  delta passed: %s" % (
                regression.get("fixed"), regression.get("new_failures"), regression.get("delta_passed")))
            if after.get("error"):
                add("- sandbox note: %s" % after["error"])
            if after.get("stdout_tail"):
                add("")
                add("```text")
                add(clip(str(after["stdout_tail"]), 1500).rstrip())
                add("```")
        lint = evidence.get("lint") or {}
        if lint.get("configured"):
            # Printed whether it passed or not, and with the command: a reader has
            # to be able to tell a scoped check from a whole-repository one, since
            # only the former can be attributed to this cycle's patch.
            add("")
            add("- lint: %s (`%s`)" % (
                "ok" if lint.get("ok") else "FAILED",
                " ".join(str(part) for part in (lint.get("command") or [])),
            ))
            if lint.get("error"):
                add("- lint note: %s" % lint["error"])
            if not lint.get("ok") and lint.get("stdout_tail"):
                add("")
                add("```text")
                add(clip(str(lint["stdout_tail"]), 1200).rstrip())
                add("```")
        if not static and not tests:
            add("_No measurement was taken (no candidate tree was produced).__")
        add("")

        # --- agents -------------------------------------------------------- #
        add("## 6. Agent contributions")
        add("")
        add("| agent | phase | mode | verdict | ok | tokens | latency | note |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in checks:
            add("| `%s` | %s | %s | %s | %s | %s | %sms | %s |" % (
                row.get("role"), row.get("phase"), row.get("mode"), row.get("verdict"),
                "yes" if row.get("ok") else "no", row.get("tokens", 0), row.get("latency_ms", 0),
                clip(str(row.get("note") or "").replace("|", "/").replace("\n", " "), 110),
            ))
        add("")
        per_role = usage.get("per_role") or {}
        if per_role:
            add("| role | calls | tokens |")
            add("| --- | --- | --- |")
            for role in sorted(per_role):
                stats = per_role[role]
                add("| `%s` | %s | %s |" % (role, stats.get("calls", 0), stats.get("tokens", 0)))
            add("")

        # --- graph --------------------------------------------------------- #
        add("## 7. Loop topology")
        add("")
        add("| from | to | when |")
        add("| --- | --- | --- |")
        for edge in LOOP_EDGES:
            add("| `%s` | `%s` | %s |" % (edge["from"], edge["to"], edge["when"]))
        add("")
        if state.get("graph_mermaid"):
            add("```mermaid")
            add(str(state["graph_mermaid"]).rstrip())
            add("```")
            add("")

        # --- reproduce ----------------------------------------------------- #
        add("## 8. Reproduce")
        add("")
        add("```powershell")
        add("python run.py run --target %s --max-iterations %s --out runs" % (state.get("target_root"), state.get("max_iterations")))
        add("python run.py apply --run runs/<run_id> --dry-run   # inspect the accepted patch")
        add("python run.py apply --run runs/<run_id>             # write it to the target")
        add("```")
        add("")
        add("---")
        add("_Generated by qedloop at %s. Policy: %s_" % (utc_now(), json.dumps(policy.to_dict(), sort_keys=True)))
        add("")
        return "\n".join(out)


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return target


def write_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # The report is a reader artifact, so it is written as pure LF.  Section 4
    # embeds the candidate diff, and those lines are built from text that carries
    # the target's CRLF -- so they end "\r\n" while every other line is joined
    # with "\n".  ``write_text``'s platform default then doubled one side into
    # "\r\r\n", which renders as a blank line and turns every diff line into two,
    # in the one artifact a human is meant to read before ``apply``.
    target.write_text(to_lf(text), encoding="utf-8", newline="")
    return target


def write_report(
    out_dir: str | Path,
    state: Dict[str, Any],
    *,
    config: Optional[Mapping[str, Any]] = None,
    include_code: bool = False,
) -> RunReport:
    base = Path(out_dir)
    run_id = str(state.get("run_id") or "run")
    report = RunReport(state=state, config=dict(config or {}))
    report.json_path = write_json(base / "state.json", state_view(state, include_code=include_code))
    report.markdown_path = write_text(base / "REPORT.md", report.to_markdown())
    trace = state.get("trace_path")
    if trace:
        write_json(base / "run.meta.json", {
            "run_id": run_id,
            "status": state.get("status"),
            "trace": str(trace),
            "markdown": str(report.markdown_path),
            "json": str(report.json_path),
            "policy": state.get("policy") or {},
        })
    return report
