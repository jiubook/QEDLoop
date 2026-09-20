# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""End-to-end loop tests driven by the deterministic offline provider.

These are the tests that prove the *closed loop* works: discovery, refinement,
review, patch and verification all wired to one state bus, with a real pytest
run deciding convergence.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from qedloop.llm import LLMError, LLMProvider, MockProvider
from qedloop.orchestrator import RunConfig, apply_run, load_codebase, run_loop
from qedloop.policy import Policy
from qedloop.report import state_view
from qedloop.sandbox import pytest_available, seed_from_directory

pytestmark = pytest.mark.skipif(not pytest_available(), reason="the loop needs pytest to verify a candidate tree")


def _config(demo_copy: Path, tmp_path: Path, **kw) -> RunConfig:
    defaults = dict(
        target=str(demo_copy),
        out_dir=str(tmp_path / "runs"),
        max_iterations=4,
        policy=Policy(quality_target=0.8),
        run_tests=True,
        test_timeout=120.0,
        quiet=True,
        run_id="e2e",
    )
    defaults.update(kw)
    return RunConfig(**defaults)


def _suite_is_blocked(result) -> bool:
    tests = ((result.state.get("evidence") or {}).get("tests") or {})
    after = tests.get("after") or {}
    return not tests.get("ran") and "could not start pytest" in str(after.get("error", ""))


def test_loop_converges_on_the_demo_fixture(demo_copy, tmp_path):
    result = run_loop(_config(demo_copy, tmp_path), provider=MockProvider())
    if _suite_is_blocked(result):
        pytest.skip("host sandbox refused to start pytest")

    assert result.status == "converged", result.state.get("status_reason")
    assert result.exit_code == 0

    # the ledger accumulated every declared defect and closed them all
    issues = result.state["issues"]
    assert len(issues) == 3, [i["title"] for i in issues]
    assert {i["bug_marker"] for i in issues} == {
        "stats/mean_zero",
        "mathx/clamp_inverted",
        "text/title_case_off_by_one",
    }
    assert all(i["status"] == "verified" for i in issues)
    assert all(i["seen_count"] >= 1 for i in issues)

    # every phase ran, and the loop needed three cycles to clear three defects
    assert set(result.state["node_visits"]) == {"discover", "refine", "review", "patch", "qa"}
    assert result.state["node_visits"]["qa"] >= 1
    assert len(result.state["quality_history"]) >= 1

    # the final quality row reflects a green suite and an approving reviewer
    final = result.state["quality_history"][-1]
    assert final["tests_green"] is True
    assert final["review_ok"] is True
    assert final["quality"] >= 0.8

    # measured evidence, not model claims
    tests = result.state["evidence"]["tests"]
    assert tests["before"]["green"] is False, "the fixture must start red"
    assert tests["after"]["green"] is True
    assert tests["regression"]["fixed"] >= 3
    static = result.state["evidence"]["static"]
    assert static["markers_after"] == 0
    assert static["compiles"] is True

    # all three source files were repaired in the candidate tree
    assert result.changed_files() == [
        "src/tinylib/mathx.py",
        "src/tinylib/stats.py",
        "src/tinylib/text.py",
    ]


def test_every_agent_lens_contributed(demo_copy, tmp_path):
    result = run_loop(_config(demo_copy, tmp_path), provider=MockProvider())
    if _suite_is_blocked(result):
        pytest.skip("host sandbox refused to start pytest")
    roles = {row["role"] for row in result.state["checks"]}
    for role in (
        "discover_archaeology", "discover_behavior", "discover_security",
        "refine_synthesize", "plan_split", "test_design",
        "architecture_review", "correctness_review", "risk_review",
        "patch_generate", "patch_refactor", "patch_reconcile",
        "static_qa", "test_sandbox", "edge_case_qa",
    ):
        assert role in roles, "agent %s never ran" % role


def test_run_artifacts_are_written(demo_copy, tmp_path):
    result = run_loop(_config(demo_copy, tmp_path), provider=MockProvider())
    run_dir = result.run_dir
    for name in ("REPORT.md", "state.json", "trace.jsonl", "candidate.manifest.json", "candidate.diff"):
        assert (run_dir / name).is_file(), "missing artifact %s" % name
    manifest = json.loads((run_dir / "candidate.manifest.json").read_text(encoding="utf-8"))
    assert manifest["verified"] is True
    assert manifest["files"] == result.changed_files()
    for rel in manifest["files"]:
        assert (run_dir / "candidate" / rel).is_file()
    report = (run_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "Self-iterating development loop" in report
    assert "| cycle | quality |" in report
    trace_lines = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(event.get("type") == "node" for event in trace_lines)
    assert any(event.get("type") == "agent" for event in trace_lines)


def test_state_json_does_not_embed_file_contents(demo_copy, tmp_path):
    result = run_loop(_config(demo_copy, tmp_path), provider=MockProvider())
    payload = json.loads((result.run_dir / "state.json").read_text(encoding="utf-8"))
    assert "working_code" not in payload, "state.json must stay a reportable slice, not a copy of the repo"
    assert payload["issues"] and payload["cycles"]


def test_single_cycle_mode_hands_off_instead_of_looping(demo_copy, tmp_path):
    config = _config(demo_copy, tmp_path, allow_multi_cycle=False, run_tests=False, policy=Policy(quality_target=0.8))
    result = run_loop(config, provider=MockProvider())
    assert result.status == "human_review"
    assert "single-cycle" in result.state["status_reason"]
    assert len(result.state["quality_history"]) == 1
    assert len(result.changed_files()) == 1, "one cycle fixes exactly the issue it scoped to"


def test_a_clean_repository_converges_immediately_without_calling_agents(tmp_path):
    target = tmp_path / "clean"
    (target / "src").mkdir(parents=True)
    (target / "src" / "ok.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (target / "tests").mkdir()
    (target / "tests" / "test_ok.py").write_text("from src.ok import f\n\n\ndef test_f():\n    assert f() == 1\n", encoding="utf-8")

    result = run_loop(
        RunConfig(target=str(target), out_dir=str(tmp_path / "runs"), quiet=True, run_id="clean"),
        provider=MockProvider(),
    )
    assert result.status == "converged"
    assert result.state["issues"] == [], "a repository with no declared defect must report none"
    assert result.state["agent_usage"]["calls"] == 3, "discovery runs once, then the loop stops"


def test_a_configured_lint_gate_keeps_a_green_run_from_converging(demo_copy, tmp_path):
    """A patch can pass every test and still be rejected by the target's own CI.

    Reproduces EER-Ai: 507 tests green while ruff refused the patch over a
    standard-library import that belonged in a ``TYPE_CHECKING`` block.  A green
    suite cannot see a rule, so the loop has to be told how to ask.

    The stand-in checker has no extension on purpose: a ``.py`` helper would
    itself become part of the repository the loop reads and patches, which would
    make this test measure something other than the gate.
    """
    (demo_copy / "lint_check").write_text(
        "import sys\n"
        "files = sys.argv[1:]\n"
        "for path in files:\n"
        "    print('%s: pretend rule violation' % path)\n"
        "sys.exit(1 if files else 0)\n",
        encoding="utf-8",
    )
    result = run_loop(
        _config(demo_copy, tmp_path, max_iterations=1, lint_command="{python} lint_check {files}"),
        provider=MockProvider(),
    )
    if _suite_is_blocked(result):
        pytest.skip("host sandbox refused to start pytest")

    lint = (result.state.get("evidence") or {}).get("lint") or {}
    assert lint.get("configured"), "the gate has to reach Phase 5 through the graph"
    assert lint.get("exit_code") == 1, lint
    changed = result.changed_files()
    assert changed, "precondition: the cycle changed something"
    assert all(path in (lint.get("command") or []) for path in changed), (
        "the check must be scoped to the files this cycle touched: %s" % lint.get("command")
    )
    assert result.status != "converged", "green tests must not outvote a configured gate"
    assert "lint did not pass" in str(result.state.get("status_reason") or ""), (
        result.state.get("status_reason")
    )


def test_apply_run_writes_the_verified_tree_onto_a_copy(demo_copy, tmp_path):
    result = run_loop(_config(demo_copy, tmp_path), provider=MockProvider())
    if _suite_is_blocked(result):
        pytest.skip("host sandbox refused to start pytest")

    fresh = tmp_path / "fresh_target"
    shutil.copytree(demo_copy, fresh)

    preview = apply_run(result.run_dir, target=str(fresh), dry_run=True)
    assert preview["dry_run"] and sorted(preview["applied"]) == result.changed_files()
    untouched = (fresh / "src" / "tinylib" / "stats.py").read_text(encoding="utf-8")
    assert "return 0.0" in untouched, "dry run must not write"

    outcome = apply_run(result.run_dir, target=str(fresh))
    assert outcome["ok"] and sorted(outcome["applied"]) == result.changed_files()
    for rel in result.changed_files():
        assert (fresh / rel).read_text(encoding="utf-8") == result.state["working_code"][rel]
        assert (fresh / (rel + ".qedloop.bak")).is_file(), "an overwritten file keeps a backup"


def test_apply_preserves_the_targets_line_endings(demo_copy, tmp_path):
    """Writing a patch must not rewrite every line of a CRLF checkout.

    Reproduces the shape of a real target (``core.autocrlf=true``, no
    ``.gitattributes``, so the working tree is CRLF): the candidate tree is kept
    in LF, and writing it back verbatim turns the whole file into an LF file.
    Measured on EER-Ai: git reported the file modified with an *empty* diff.

    This test used to assert only ``crlf > 0 and lone_lf == 0``, which
    "\\r\\r\\n" satisfies -- so it passed while every edited file was gaining a
    blank line between every line.  It now checks the doubling and the line
    count as well.
    """
    repo = tmp_path / "crlf_repo"
    shutil.copytree(demo_copy, repo)
    for path in repo.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        path.write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))

    result = run_loop(_config(repo, tmp_path), provider=MockProvider())
    if _suite_is_blocked(result):
        pytest.skip("host sandbox refused to start pytest")
    assert result.status == "converged", result.state.get("status_reason")

    outcome = apply_run(result.run_dir, target=str(repo))
    assert outcome["ok"], outcome
    changed = [rel for rel in result.changed_files() if rel.endswith(".py")]
    assert changed, "precondition: the run changed at least one python file"
    for rel in changed:
        body = (repo / rel).read_bytes()
        crlf = body.count(b"\r\n")
        lone_lf = body.count(b"\n") - crlf
        assert crlf > 0 and lone_lf == 0, (
            "%s came back with %d CRLF and %d lone LF; the target is a CRLF checkout" % (rel, crlf, lone_lf)
        )
        assert b"\r\r\n" not in body, (
            "%s came back with a doubled carriage return; each one reads as an extra blank line" % rel
        )
        assert (repo / rel).read_text(encoding="utf-8").count("\n") == (
            result.state["working_code"][rel].count("\n")
        ), "%s changed its line structure on the way to the target" % rel

        # The persisted candidate tree is what ``apply`` copies from, so it must
        # be clean too -- and it is deliberately kept as LF.
        candidate = Path(result.run_dir) / "candidate" / rel
        candidate_body = candidate.read_bytes()
        assert b"\r" not in candidate_body, "%s must be kept as LF" % rel

    diff = (Path(result.run_dir) / "candidate.diff").read_bytes()
    assert b"\r" not in diff, "the reviewed diff is a review artifact and is kept as LF"

    # Section 4 of the report embeds that same diff, so the report is written as
    # LF too -- it is the artifact a human actually reads before ``apply``.
    report = (Path(result.run_dir) / "REPORT.md").read_bytes()
    assert b"\r" not in report, "the report embeds the diff and is kept as LF"


def test_the_newline_helpers_agree_with_each_other():
    """Detect, re-express, and compare: three small rules that must not drift."""
    from qedloop.orchestrator import detect_newline, to_lf, with_newline

    assert detect_newline(b"a\r\nb\r\n") == "\r\n"
    assert detect_newline(b"a\nb\n") == "\n"
    assert detect_newline(b"a\nb\nc\r\n") == "\n", "the majority wins"
    assert detect_newline(b"a\r\nb\r\nc\n") == "\r\n"
    assert to_lf("a\r\nb\n") == "a\nb\n"
    assert with_newline("a\nb\n", "\r\n") == "a\r\nb\r\n"
    assert with_newline("a\r\nb\r\n", "\r\n") == "a\r\nb\r\n", "no doubled carriage returns"
    assert with_newline("a\r\nb\n", "\n") == "a\nb\n"


def test_the_baseline_is_announced_before_it_starts(demo_copy, tmp_path):
    """The first thing a real run does is copy the target and run its whole
    suite, which is minutes of silence on a big repository -- and silence is what
    makes an operator reach for Ctrl+C on a run that is working fine.  The
    announcement has to come *before* the measurement, and it has to be in the
    trace, so a run interrupted in those minutes still says where it was.
    """
    result = run_loop(_config(demo_copy, tmp_path), provider=MockProvider())
    events = [
        json.loads(line)
        for line in (result.run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baseline = [e for e in events if e.get("event") == "baseline"]
    assert [e["state"] for e in baseline] == ["measuring", "done"], baseline
    assert "copying the target" in str(baseline[0].get("note"))

    first_node = next(index for index, e in enumerate(events) if e.get("type") == "node")
    assert events.index(baseline[0]) < events.index(baseline[1]) < first_node, (
        "the baseline is measured, and reported, before any agent runs"
    )
    assert baseline[1]["result"]["passed"] > 0, baseline[1]["result"]


def test_a_plan_whose_own_commands_collect_nothing_is_rejected_without_calling_an_agent():
    """The plan check runs before the reviewers, and costs nothing.

    Three of four review rounds in a measured run went to contradictions inside
    the plan's own text -- a ``-k`` filter that selected no planned test, a step
    naming a test the planned-test list had named differently, a new test module
    duplicating an existing one.  Those need no judgement, so they must not buy
    three model calls; and the objection still has to reach the next refinement
    round, or the same plan comes back.
    """
    from qedloop.crew import Crew
    from qedloop.phases.review import make_review_node

    crew = Crew(MockProvider())
    state = {
        "cycle": 0,
        "self_loop_count": 0,
        "target_issue_id": "ISS-0001",
        "issues": [{"id": "ISS-0001", "title": "the shutdown branch is never exercised", "status": "open"}],
        "findings": [
            {
                "id": "FND-0001",
                "issue_id": "ISS-0001",
                "decision": "act",
                "steps": [{"order": 1, "action": "Run `pytest tests/unit/test_svc.py -k not_running` and record the failure"}],
            }
        ],
        "test_plans": [
            {
                "id": "PLN-0001",
                "issue_id": "ISS-0001",
                "covers": ["ISS-0001"],
                "name": "test_exit_application_skips_stop_scan_when_scanner_idle",
                "path": "tests/unit/test_svc.py",
            }
        ],
        "working_code": {"tests/unit/test_svc.py": "def test_existing():\n    pass\n"},
        "reviews": [],
    }

    delta = make_review_node(crew, Policy())(state)

    assert crew.calls == 0, "a contradiction in the plan must not buy three model calls"
    assert delta["review_decision"] == "refine", delta["review_reason"]
    assert delta["reviews"][0]["lens"] == "plan-consistency", delta["reviews"][0]
    assert "-k not_running" in delta["reviews"][0]["blocking"][0]["reason"], delta["reviews"][0]
    assert delta["review_blocking"], "the objection has to reach the next refinement round"
    assert delta["self_loop_count"] == 1, "it is charged to the same back-edge budget as a rejection"


class _UncollectablePlanProvider(MockProvider):
    """A planning lens that names a test its own run step cannot collect.

    This is the measured shape: the step runs ``-k not_running`` while the test it
    is supposed to exercise is called something else entirely, so the step that
    proves the change is load-bearing would observe nothing.
    """

    def complete(self, messages, **kwargs):
        if self._role(messages) == "plan_split":
            return self._json(
                {
                    "steps": [
                        {
                            "order": 1,
                            "action": "Run `pytest tests -k definitely_not_a_test_name` and record the failure",
                            "files": ["tests/test_stats.py"],
                        }
                    ],
                    "estimate": "XS",
                    "rollback": "revert the file",
                }
            )
        return super().complete(messages, **kwargs)


def test_an_inconsistent_plan_never_reaches_the_patch_node(demo_copy, tmp_path):
    """The whole point of checking the plan: nothing is implemented from it.

    The convergence test above is the other half of this one -- it runs the same
    mock crew, whose plans *are* consistent, and reaches `converged`.  Together
    they say the check fires on a contradiction and stays quiet otherwise.
    """
    result = run_loop(_config(demo_copy, tmp_path, run_tests=False), provider=_UncollectablePlanProvider())

    assert result.status == "human_review", result.state.get("status_reason")
    assert "refine budget" in str(result.state.get("status_reason")), result.state.get("status_reason")
    lenses = {review.get("lens") for review in result.state.get("reviews") or []}
    assert lenses == {"plan-consistency"}, lenses
    assert not (result.state.get("patches") or []), "an inconsistent plan must never be implemented"

    outcome = apply_run(result.run_dir, target=str(demo_copy), dry_run=True)
    assert not outcome.get("ok") and "no candidate tree" in str(outcome.get("reason")), outcome


def test_apply_refuses_a_run_that_did_not_converge(demo_copy, tmp_path):
    result = run_loop(
        _config(demo_copy, tmp_path, run_tests=False, allow_multi_cycle=False),
        provider=MockProvider(),
    )
    assert result.status != "converged"
    outcome = apply_run(result.run_dir, target=str(demo_copy), dry_run=True)
    assert outcome["ok"] is False and "did not converge" in outcome["reason"]


class _InterruptingProvider(MockProvider):
    """The operator pressing Ctrl+C, landing on a crew call.

    That is where the wall clock goes -- a cycle is minutes of model latency --
    so that is where a human waiting on a run presses the key.  Raised from a QA
    lens on purpose: by then the patch has already been written into the
    candidate tree, which is the case that must never end up appliable.
    """

    def complete(self, messages, **kwargs):
        if self._role(messages) in ("static_qa", "test_sandbox", "edge_case_qa"):
            raise KeyboardInterrupt
        return super().complete(messages, **kwargs)


def test_an_interrupted_run_stays_readable_and_can_never_be_applied(demo_copy, tmp_path):
    """Ctrl+C must leave an audit trail, and must not leave a patch behind.

    Before this, an interrupt unwound the whole orchestrator: the run directory
    kept a trace and nothing else -- no ledger, no report, no way to see how far
    it got, which is the worst possible moment to lose the record.  The opposite
    mistake would be worse: a candidate tree that ``apply --allow-unverified``
    would happily write onto the repository.
    """
    result = run_loop(_config(demo_copy, tmp_path), provider=_InterruptingProvider())

    assert result.error.startswith("KeyboardInterrupt"), result.error
    assert result.status == "human_review", result.state.get("status_reason")
    assert result.exit_code == 1, "an interrupted run is not a success"
    assert "interrupted by the operator" in result.state["status_reason"]
    assert result.changed_files(), "precondition: it stopped after a patch had been applied"

    assert (result.run_dir / "state.json").is_file(), "the ledger has to survive the interrupt"
    report = (result.run_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "interrupted by the operator" in report, "and the report has to say what happened"

    preview = apply_run(result.run_dir, target=str(demo_copy), dry_run=True)
    assert not preview.get("ok") and "no candidate tree" in str(preview.get("reason")), preview

    forced = apply_run(result.run_dir, target=str(demo_copy), allow_unverified=True)
    assert not forced.get("ok"), "the stop button must not become an appliable patch: %s" % forced
    assert "no candidate tree" in str(forced.get("reason")), forced


def test_the_fixture_on_disk_is_never_modified(demo_copy, tmp_path):
    before = seed_from_directory(demo_copy)
    result = run_loop(_config(demo_copy, tmp_path), provider=MockProvider())
    assert result.changed_files(), "the run must have produced a candidate"
    assert seed_from_directory(demo_copy) == before, "running the loop must not touch the target repository"


class _QaRejectingProvider(MockProvider):
    """A crew that measures a green suite and still refuses the patch.

    This is the real shape of the failure: the candidate kept the suite green
    (so the blended score said 1.00) while every QA lens rejected the change for
    not meeting the acceptance contract.  Nothing in the measured components or
    the reviewer votes can see that difference -- only a lens reading the
    artifact can.
    """

    def complete(self, messages, **kwargs):
        if self._role(messages) in ("static_qa", "test_sandbox", "edge_case_qa"):
            return self._json(
                {
                    "verdict": "reject",
                    "observations": "the patch rewrites the assertion instead of fixing the behaviour",
                    "confidence": 0.8,
                }
            )
        return super().complete(messages, **kwargs)


def test_a_qa_rejection_prevents_convergence_end_to_end(demo_copy, tmp_path):
    """The whole path: QA objection -> gate -> status -> issues -> report.

    Measured on a real run before this gate existed: the loop reported quality
    1.00, a green suite, three approvals and three *verified* issues while all
    three QA lenses had rejected the candidate tree.  A run must not be able to
    declare victory over its own reviewers.
    """
    result = run_loop(_config(demo_copy, tmp_path, max_iterations=2), provider=_QaRejectingProvider())
    if _suite_is_blocked(result):
        pytest.skip("host sandbox refused to start pytest")

    assert result.status != "converged", "QA rejected the tree; the loop cannot claim success"
    assert result.exit_code != 0, "and the exit code has to carry that"
    assert "QA rejected" in result.state["status_reason"]

    verified = [i["id"] for i in result.state["issues"] if i.get("status") == "verified"]
    assert verified == [], "a rejected candidate tree must not earn verify credits: %s" % verified

    report = (result.run_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "QA objections" in report, "the objection must be in the summary table, not only in the JSON"


class _BrokenProvider(LLMProvider):
    """Every call fails, the way an expired key or a wrong model id fails."""

    name = "broken"
    model = "broken"

    def complete(self, messages, *, temperature=0.2, max_tokens=1200):
        raise LLMError("no API key: set OPENAI_API_KEY/LLM_API_KEY or pass --api-key")


def test_a_crew_that_cannot_answer_is_not_a_clean_repository(tmp_path):
    """The most dangerous false negative the loop can produce.

    If every discovery agent fails, "0 issues found" and "0 agents replied" look
    identical from the outside -- and "converged" would tell a user whose key
    just expired that their repository is clean.  A scan nobody performed is not
    a finding, so it must not end the run as one.
    """
    target = tmp_path / "repo"
    (target / "src").mkdir(parents=True)
    (target / "src" / "ok.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (target / "tests").mkdir()
    (target / "tests" / "test_ok.py").write_text(
        "from src.ok import f\n\n\ndef test_f():\n    assert f() == 1\n", encoding="utf-8"
    )

    result = run_loop(
        RunConfig(target=str(target), out_dir=str(tmp_path / "runs"), quiet=True, run_id="broken", run_tests=False),
        provider=_BrokenProvider(),
    )

    assert result.status == "error", "an unanswered scan must not be reported as convergence"
    assert "discovery agents failed" in result.state["status_reason"]
    assert result.exit_code != 0, "a run that measured nothing must not exit 0"
    assert result.state["agent_usage"]["agents_failed"] == 3

    report = (result.run_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "agents failed" in report, "the summary must show it, not only the phase tables"
    assert "**Status** `error`" in report


def test_codebase_loading_respects_globs_and_skips_caches(demo_copy):
    codebase = load_codebase(str(demo_copy), ("*.py",))
    paths = codebase.manifest
    assert "src/tinylib/mathx.py" in paths
    assert not [p for p in paths if "__pycache__" in p]
    assert codebase.revision and len(codebase.revision) == 12

    only_tests = load_codebase(str(demo_copy), ("test_*.py",))
    assert list(only_tests.manifest) == ["tests/test_tinylib.py"]


# --------------------------------------------------------------------------- #
# per-target project constraints (run.brief)
# --------------------------------------------------------------------------- #


class _CapturingProvider(MockProvider):
    """A mock that keeps what it was sent, so a prompt can be asserted on."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list = []

    def complete(self, messages, **kw):
        self.seen.append("\n".join(m.content for m in messages))
        return super().complete(messages, **kw)


def test_brief_is_read_relative_to_the_config_that_names_it(tmp_path):
    """``brief: brief.md`` means the file next to the config, not next to the CWD."""
    from qedloop.config import ConfigError
    from qedloop.orchestrator import RunConfig

    config_dir = tmp_path / "targets" / "demo"
    config_dir.mkdir(parents=True)
    expected = "Never touch generated files."
    (config_dir / "brief.md").write_text(expected + "\n", encoding="utf-8")

    config = RunConfig.from_mapping(
        {"target": str(tmp_path), "brief": "brief.md"},
        brief_base=str(config_dir),
    )
    assert config.brief == expected, "the trailing newline is not part of the brief"
    assert config.brief_source == str(config_dir / "brief.md")
    assert config.to_dict()["brief"] == {"source": str(config_dir / "brief.md"), "chars": len(expected)}

    with pytest.raises(ConfigError) as excinfo:
        RunConfig.from_mapping({"brief": "missing.md"}, brief_base=str(config_dir))
    assert "brief" in str(excinfo.value), "a brief that cannot be read must not be ignored"


def test_project_constraints_reach_the_provider_and_the_report(tmp_path):
    """The whole path: config -> state bus -> system message -> provider."""
    target = tmp_path / "clean"
    (target / "src").mkdir(parents=True)
    (target / "src" / "ok.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (target / "tests").mkdir()
    (target / "tests" / "test_ok.py").write_text(
        "from src.ok import f\n\n\ndef test_f():\n    assert f() == 1\n", encoding="utf-8"
    )

    brief = "Windows only. Do not add dependencies."
    provider = _CapturingProvider()
    result = run_loop(
        RunConfig(
            target=str(target),
            out_dir=str(tmp_path / "runs"),
            brief=brief,
            brief_source=str(tmp_path / "brief.md"),
            quiet=True,
            run_id="brief",
        ),
        provider=provider,
    )

    assert result.status == "converged"
    assert provider.seen, "the agents must have been called"
    for prompt in provider.seen:
        assert "<project_constraints>" in prompt and brief in prompt, prompt[:200]

    assert result.state["brief"] == brief
    assert state_view(result.state)["brief"] == {"source": str(tmp_path / "brief.md"), "chars": len(brief)}
