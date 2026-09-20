# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""CLI contract tests: what the user sees, and what an exit code means.

The commands are the only part of the framework a user interacts with directly,
so their silence is a feature only when it is deliberate.  A run that prints
nothing for five minutes is indistinguishable from a hang, and a run that
finished with nothing to say is indistinguishable from one that never started.
"""

from __future__ import annotations

from pathlib import Path

from qedloop.cli import build_parser, cmd_run
from qedloop.orchestrator import RunResult, format_event


def test_the_live_node_line_reports_the_result_not_just_the_departure():
    """Measured before this: every node line read ``ok 0ms -> `` whatever the
    node did, because the engine only ever reported a node *before* it ran."""
    start = format_event({"type": "node", "event": "start", "node": "discover", "step": 1, "cycle": 0})
    assert "start" in start and "step=1" in start and "->" not in start

    done = format_event({
        "type": "node", "event": "done", "node": "qa", "step": 5, "cycle": 0, "visit": 1,
        "ok": True, "duration_ms": 9231.4, "next": "__end__",
        "delta": {"evidence": "4 keys", "quality_history": "1 item"},
    })
    assert "done" in done and "9231" in done and "-> __end__" in done
    assert "evidence=4 keys" in done, done


def test_a_node_that_failed_is_not_printed_as_a_success():
    line = format_event({
        "type": "node", "event": "done", "node": "patch", "step": 4, "cycle": 0,
        "ok": False, "duration_ms": 12.0, "error": "ValueError: nope",
    })
    assert "FAIL" in line, line


def test_a_node_event_from_an_older_trace_still_reads_correctly():
    """traces are append-only evidence: the reader has to keep working on files
    written before node events carried an ``event`` key."""
    line = format_event({"type": "node", "node": "patch", "step": 4, "cycle": 2, "duration_ms": 5.0, "ok": True})
    assert "done" in line and "step=4" in line


def test_every_crew_member_line_carries_its_vote_and_cost():
    """``agent.done`` carries role/lens/verdict/tokens.  The field picker looked
    for keys this event does not have, so most crew lines printed blank -- who
    spoke, how they voted and what it cost were invisible."""
    line = format_event({
        "type": "agent", "event": "agent.done", "phase": "review", "cycle": 0,
        "role": "correctness_review", "lens": "correctness", "mode": "vote",
        "ok": True, "verdict": "reject", "tokens": 25222, "latency_ms": 70912.0,
        "note": "the plan names two different paths for the same test",
    })
    assert "correctness_review" in line and "reject" in line
    assert "25222 tok" in line and "70912" in line
    assert "two different paths" in line, line


def test_a_crew_member_that_failed_says_so_instead_of_abstaining():
    line = format_event({
        "type": "agent", "event": "agent.done", "phase": "discover", "cycle": 0,
        "role": "discover_behavior", "ok": False, "verdict": "abstain",
        "tokens": 0, "latency_ms": 3.0, "note": "",
    })
    assert "FAIL" in line, line


def test_a_review_verdict_line_names_the_decision_and_the_objection():
    """The one line that says whether the reviewers let the plan through."""
    line = format_event({
        "type": "agent", "event": "phase3.verdict", "phase": "review", "cycle": 0,
        "decision": "refine", "reason": "rejected by correctness",
        "votes": {"architecture": "approve", "correctness": "reject"},
        "blocking": [{"lens": "correctness", "issue_id": "ISS-0006", "reason": "two paths"}],
    })
    assert "decision=refine" in line and "rejected by correctness" in line
    assert "blocking=1" in line, line


def _run_args(tmp_path: Path, *extra: str):
    return build_parser().parse_args(
        ["run", "--target", str(tmp_path), "--provider", "mock", "--single-cycle", *extra]
    )


def _fake_run_loop(captured_run_dir: Path, status: str, reason: str):
    def _run_loop(config):
        run_dir = captured_run_dir / config.run_id
        return RunResult(
            run_id=config.run_id,
            state={"status": status, "status_reason": reason, "working_code": {}, "codebase": {}},
            run_dir=run_dir,
            exit_code=0,
        )

    return _run_loop


def test_a_quiet_run_announces_itself_and_reports_the_result(tmp_path, capsys, monkeypatch):
    """`--quiet` silences the live trace, not the run itself."""
    monkeypatch.setattr(
        "qedloop.cli.run_loop",
        _fake_run_loop(tmp_path / "runs", "converged", "quality 1.00 >= target 0.80"),
    )

    assert cmd_run(_run_args(tmp_path, "--quiet")) == 0
    out = capsys.readouterr().out

    assert "trace.jsonl" in out, "the user must be told where to watch progress"
    assert "converged" in out, "and the outcome must be reported, not just the exit code"
    assert "quality 1.00" in out


def test_a_loud_run_does_not_duplicate_the_summary(tmp_path, capsys, monkeypatch):
    """Without ``--quiet`` the framework already prints the summary once."""
    monkeypatch.setattr(
        "qedloop.cli.run_loop",
        _fake_run_loop(tmp_path / "runs", "converged", "quality 1.00 >= target 0.80"),
    )

    assert cmd_run(_run_args(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "running :" not in out, "no banner when the live trace is on"
    assert out == "", "run_loop owns the output in the loud path"


def test_a_failed_run_points_at_the_report(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        "qedloop.cli.run_loop",
        _fake_run_loop(tmp_path / "runs", "human_review", "two cycles closed nothing"),
    )

    assert cmd_run(_run_args(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "next: inspect" in out and "REPORT.md" in out


def test_qa_objections_are_reported_not_buried(tmp_path):
    """A rejected candidate tree must appear in the summary the user reads.

    The run that motivated this reported "final quality 1.0" and three verify
    credits while all three QA lenses were rejecting the patch; the objection
    was only discoverable by opening state.json.
    """
    from qedloop.report import RunReport

    report = RunReport(
        state={
            "run_id": "RID",
            "status": "human_review",
            "status_reason": "single-cycle mode stopped the loop; QA rejected the candidate tree (edge_case_qa)",
            "quality_history": [
                {
                    "quality": 1.0,
                    "qa_verdicts": {"static_qa": "reject", "edge_case_qa": "reject", "test_sandbox": "approve"},
                }
            ],
            "issues": [],
            "policy": {"quality_target": 0.8},
            "agent_usage": {},
        },
    )
    lines = "\n".join(report.summary_lines())
    assert "qa objections" in lines
    assert "edge_case_qa" in lines and "static_qa" in lines
    assert "test_sandbox" not in lines, "an approving lens is not an objection"


def test_no_qa_line_when_nobody_objected(tmp_path):
    from qedloop.report import RunReport

    report = RunReport(
        state={
            "run_id": "RID",
            "status": "converged",
            "status_reason": "quality 1.00 >= target 0.80",
            "quality_history": [{"quality": 1.0, "qa_verdicts": {"static_qa": "approve"}}],
            "issues": [],
            "policy": {"quality_target": 0.8},
            "agent_usage": {},
        },
    )
    assert "qa objections" not in "\n".join(report.summary_lines())
