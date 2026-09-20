# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Gate tests: the two decisions that steer the loop, with no LLM involved."""

from __future__ import annotations

import pytest

from qedloop.policy import Policy, block_reasons, convergence, is_plateau, review_route, score_cycle


def _review(lens, verdict, reason=""):
    return {"lens": lens, "verdict": verdict, "blocking": ([{"issue_id": "ISS-0001", "reason": reason}] if reason else [])}


def _lint(*, ok, configured=True, ran=True, exit_code=1, command=("ruff", "check"), error=""):
    return {
        "configured": configured,
        "ran": ran,
        "ok": ok,
        "command": list(command),
        "exit_code": exit_code,
        "error": error,
    }


def _evidence(*, ran=True, passed=3, failed=0, errors=0, new_failures=0, compiles=True, introduced=0, lint=None):
    evidence = {
        "static": {
            "compiles": compiles,
            "markers_before": 1,
            "markers_after": 0 if compiles else 1,
            "markers_resolved": [{"name": "demo/marker"}] if compiles else [],
            "markers_introduced": [{"name": "new"}] * introduced,
            "syntax_errors": [] if compiles else [{"path": "a.py", "line": 1, "message": "bad"}],
        },
        "tests": {
            "ran": ran,
            "after": {"passed": passed, "failed": failed, "errors": errors, "total": passed + failed + errors, "green": failed == 0 and errors == 0},
            "before": {"passed": 1, "failed": 3, "errors": 0, "total": 4, "green": False},
            "regression": {"new_failures": new_failures, "fixed": 3 - failed, "before_green": False, "after_green": failed == 0, "delta_passed": passed - 1},
        },
    }
    if lint is not None:
        evidence["lint"] = lint
    return evidence


# --------------------------------------------------------------------------- #
# Phase 3 gate
# --------------------------------------------------------------------------- #


def test_review_gate_approves_with_enough_approvals():
    policy = Policy()
    route = review_route([_review("architecture", "approve"), _review("correctness", "approve")], self_loop_count=0, policy=policy)
    assert route["decision"] == "approve"


def test_review_gate_sends_rejection_back_to_refinement():
    policy = Policy()
    route = review_route(
        [_review("architecture", "approve"), _review("correctness", "reject", "the fix is not verifiable")],
        self_loop_count=0,
        policy=policy,
    )
    assert route["decision"] == "refine"
    assert route["blocking"][0]["reason"] == "the fix is not verifiable"


def test_review_gate_stops_once_the_refine_budget_is_spent():
    """The stop message must not read like there is budget left.

    A measured run printed "review rejected 2 time(s) and the refine budget (3) is
    spent": the 2 was the number of lenses objecting *in that round*, while three
    rounds had actually been rejected and the budget -- which is counted across
    the whole run, not per cycle -- was gone.  It read as if a try remained.
    """
    policy = Policy(max_self_loops=2)
    route = review_route(
        [_review("correctness", "reject", "no"), _review("risk-and-regression", "reject", "also no")],
        self_loop_count=2,
        policy=policy,
    )
    assert route["decision"] == "stop"
    assert "2 lens(es)" in route["reason"], route["reason"]
    assert "across the whole run" in route["reason"], route["reason"]


def test_review_gate_refines_when_nobody_voted():
    policy = Policy()
    route = review_route([_review("architecture", "abstain")], self_loop_count=0, policy=policy)
    assert route["decision"] == "refine"
    assert "abstained" in route["reason"]


def test_review_gate_requires_the_configured_number_of_approvals():
    policy = Policy(min_approvals=2)
    assert review_route([_review("architecture", "approve")], self_loop_count=0, policy=policy)["decision"] == "refine"
    assert review_route(
        [_review("architecture", "approve"), _review("correctness", "approve")], self_loop_count=0, policy=policy
    )["decision"] == "approve"


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def test_score_is_one_when_everything_is_green():
    policy = Policy()
    score = score_cycle(
        reviews=[_review("architecture", "approve"), _review("correctness", "approve")],
        evidence=_evidence(),
        checks=[],
        policy=policy,
    )
    assert score["quality"] == pytest.approx(1.0)
    assert score["tests_green"] and score["review_ok"] and score["compiles"]


def test_score_reports_compiles_false_when_the_tree_is_broken():
    score = score_cycle(reviews=[_review("a", "approve")], evidence=_evidence(compiles=False), checks=[], policy=Policy())
    assert score["compiles"] is False
    assert score["components"]["static"] == 0.0


def test_score_drops_when_the_suite_is_red():
    policy = Policy()
    score = score_cycle(
        reviews=[_review("architecture", "approve")],
        evidence=_evidence(passed=1, failed=2),
        checks=[],
        policy=policy,
    )
    assert score["quality"] < policy.quality_target
    assert score["tests_green"] is False


def test_new_failures_are_penalised_even_with_a_green_absolute_result():
    policy = Policy()
    clean = score_cycle(reviews=[_review("a", "approve")], evidence=_evidence(), checks=[], policy=policy)
    regressed = score_cycle(reviews=[_review("a", "approve")], evidence=_evidence(new_failures=1), checks=[], policy=policy)
    assert regressed["quality"] < clean["quality"]
    assert regressed["tests_green"] is False


def test_syntax_errors_and_new_markers_reduce_the_static_component():
    policy = Policy()
    broken = score_cycle(reviews=[_review("a", "approve")], evidence=_evidence(compiles=False), checks=[], policy=policy)
    assert broken["components"]["static"] == 0.0
    marked = score_cycle(reviews=[_review("a", "approve")], evidence=_evidence(introduced=1), checks=[], policy=policy)
    assert marked["components"]["static"] < 1.0


def test_missing_test_evidence_cannot_converge_when_tests_are_required():
    policy = Policy(require_tests=True)
    score = score_cycle(reviews=[_review("a", "approve")], evidence=_evidence(ran=False), checks=[], policy=policy)
    assert score["tests_green"] is False


def test_a_configured_lint_gate_blocks_convergence_and_names_the_command():
    """Green tests cannot see a lint rule, and "lint failed" is not actionable.

    Reproduces EER-Ai: the crew's patch passed 507 tests and was rejected by the
    target's own ruff configuration (`TC003`).  Nothing in the compile status or
    the suite can see that, so without this gate the run reports a clean
    convergence and fails CI the moment a human commits it.
    """
    policy = Policy(quality_target=0.8)
    score = score_cycle(
        reviews=[_review("a", "approve")],
        evidence=_evidence(lint=_lint(ok=False, command=("{python}", "-m", "ruff", "check", "src/a.py"))),
        checks=[],
        policy=policy,
    )

    assert score["quality"] == pytest.approx(1.0), "a requirement is not a weighted component"
    assert score["lint_ok"] is False
    reasons = block_reasons(score, policy, cycle=0)
    assert any("lint did not pass" in r and "ruff" in r for r in reasons), reasons
    assert _converge(score, policy=policy)["status"] != "converged"


def test_an_unconfigured_lint_gate_blocks_nothing():
    """The default has to be a no-op, so every existing target behaves as before."""
    policy = Policy(quality_target=0.8)
    score = score_cycle(reviews=[_review("a", "approve")], evidence=_evidence(), checks=[], policy=policy)

    assert score["lint_ok"] is True and score["lint"]["configured"] is False
    assert block_reasons(score, policy, cycle=0) == []
    assert _converge(score, policy=policy)["status"] == "converged"


def test_a_lint_gate_that_could_not_run_is_not_a_passing_gate():
    """A typo in the command must not silently disable the check."""
    policy = Policy(quality_target=0.8)
    score = score_cycle(
        reviews=[_review("a", "approve")],
        evidence=_evidence(lint=_lint(
            ok=False, ran=False, exit_code=None, command=("nosuchlinter",),
            error="could not start the lint command: [WinError 2]",
        )),
        checks=[],
        policy=policy,
    )

    assert score["lint_ok"] is False
    reasons = block_reasons(score, policy, cycle=0)
    assert any("nosuchlinter" in r for r in reasons), reasons
    assert _converge(score, policy=policy)["status"] != "converged"


def test_a_plan_refined_three_times_is_not_penalised_for_refining():
    """The score used to divide the whole review ledger, not the round that mattered.

    Reproduces a real run: twelve review rows over four rounds (6 approve / 6
    reject), the Phase 3 gate approved on the last round 3/3, the patch applied
    cleanly, the suite went 503 -> 507 passed -- and the reported quality was
    0.70 with review 0.00, so a single-cycle run handed a success over as if it
    had failed.  Making the reviewers reject a plan and then fixing it is the
    loop working, not a defect to be scored down.
    """
    policy = Policy()
    final_round = [_review(lens, "approve") for lens in ("architecture", "correctness", "risk-and-regression")]
    ledger = (
        [_review("architecture", "approve"), _review("correctness", "approve"),
         _review("risk-and-regression", "reject", "the fix is not verifiable")]
        + [_review(lens, "reject", "the plan contradicts itself")
           for lens in ("architecture", "correctness", "risk-and-regression")]
        + [_review("architecture", "approve"), _review("correctness", "reject", "wrong endpoint"),
           _review("risk-and-regression", "reject", "no rollback")]
        + final_round
    )

    first_pass = score_cycle(reviews=final_round, evidence=_evidence(), checks=[], policy=policy)
    refined = score_cycle(reviews=ledger, evidence=_evidence(), checks=[], policy=policy)

    assert refined["quality"] == pytest.approx(first_pass["quality"]), (
        "the score must not depend on how many review rounds the plan needed"
    )
    assert refined["quality"] >= policy.quality_target
    assert refined["review_ok"] is True
    assert (refined["approvals"], refined["rejections"]) == (6, 6), "the votes are still reported as counts"


def test_review_votes_are_a_requirement_not_a_weighted_component():
    """A hard requirement that is also weighted is counted twice.

    ``review_ok`` already gates convergence, so the votes must change the
    verdict and never the score -- otherwise the score can keep falling after
    the requirement is satisfied, which is how it came to contradict the gate it
    sits next to.
    """
    policy = Policy()
    assert policy.weights.get("review", 0.0) == 0.0, "precondition: review carries no weight"

    approved = score_cycle(reviews=[_review("a", "approve")], evidence=_evidence(), checks=[], policy=policy)
    nobody = score_cycle(reviews=[], evidence=_evidence(), checks=[], policy=policy)

    assert approved["quality"] == pytest.approx(nobody["quality"])
    assert approved["review_ok"] is True and nobody["review_ok"] is False
    assert "review" not in approved["components"]


# --------------------------------------------------------------------------- #
# Phase 5 gate
# --------------------------------------------------------------------------- #


def _converge(score, cycle=1, history=(), policy=None, max_iterations=3, allow_continue=True, no_progress=0):
    return convergence(
        score,
        cycle=cycle,
        max_iterations=max_iterations,
        history=list(history),
        policy=policy or Policy(),
        allow_continue=allow_continue,
        no_progress_cycles=no_progress,
    )


def test_convergence_requires_green_tests_and_an_approval():
    policy = Policy(quality_target=0.5)
    assert _converge({"quality": 0.9, "tests_green": False, "review_ok": True, "compiles": True}, policy=policy)["status"] == "continue"
    assert _converge({"quality": 0.9, "tests_green": True, "review_ok": False, "compiles": True}, policy=policy)["status"] == "continue"
    assert _converge({"quality": 0.9, "tests_green": True, "review_ok": True, "compiles": True}, policy=policy)["status"] == "converged"


def test_a_tree_that_does_not_compile_can_never_converge():
    """A candidate tree that fails to parse must not be reported as a success,
    however good the other numbers look."""
    policy = Policy(quality_target=0.5)
    decision = _converge({"quality": 1.0, "tests_green": True, "review_ok": True, "compiles": False}, policy=policy)
    assert decision["status"] == "continue"
    assert "does not compile" in decision["reason"]


def test_target_threshold_is_enforced():
    policy = Policy(quality_target=0.95)
    decision = _converge({"quality": 0.9, "tests_green": True, "review_ok": True}, policy=policy)
    assert decision["status"] == "continue" and "0.90" in decision["reason"]


def test_iteration_budget_escalates_instead_of_looping_forever():
    decision = _converge({"quality": 0.4, "tests_green": False, "review_ok": True}, cycle=3, max_iterations=3)
    assert decision["status"] == "escalated"


def test_the_iteration_budget_counts_finished_cycles_not_the_index():
    """``max_iterations: 3`` must buy three cycles, not four.

    ``cycle`` indexes the cycle that just finished, so comparing the *index*
    against a budget of *cycles* let one extra cycle run: a run that never
    converged printed "cycles completed | 4 of 3".  Same index-versus-count trap
    ``min_iterations`` fell into, in the other direction.
    """
    stalled = {"quality": 0.4, "tests_green": False, "review_ok": True}
    assert _converge(stalled, cycle=1, max_iterations=3)["status"] == "continue", "2 of 3 cycles done"
    assert _converge(stalled, cycle=2, max_iterations=3)["status"] == "escalated", "3 of 3 cycles done"


def test_a_quality_plateau_hands_off_to_a_human():
    history = [{"quality": 0.5}, {"quality": 0.505}, {"quality": 0.507}]
    assert is_plateau(history, Policy()) is True
    decision = _converge(
        {"quality": 0.507, "tests_green": False, "review_ok": True},
        cycle=3,
        history=history,
        max_iterations=10,
        no_progress=2,
    )
    assert decision["status"] == "human_review"
    assert "stuck" in decision["reason"]


def test_closing_an_issue_counts_as_progress_even_when_the_score_is_flat():
    history = [{"quality": 0.5}, {"quality": 0.505}, {"quality": 0.507}]
    decision = _converge(
        {"quality": 0.507, "tests_green": False, "review_ok": True},
        cycle=3,
        history=history,
        max_iterations=10,
        no_progress=0,
    )
    assert decision["status"] == "continue"
    assert is_plateau([{"quality": 0.3}, {"quality": 0.5}], Policy()) is False


def test_real_progress_does_not_trigger_the_plateau_rule():
    history = [{"quality": 0.3}, {"quality": 0.5}, {"quality": 0.7}]
    decision = _converge({"quality": 0.7, "tests_green": False, "review_ok": True}, cycle=3, history=history, max_iterations=10)
    assert decision["status"] == "continue"


def test_single_cycle_mode_stops_after_one_pass():
    decision = _converge({"quality": 0.4, "tests_green": True, "review_ok": True}, cycle=1, allow_continue=False)
    assert decision["status"] == "human_review"


def _perfect(**overrides):
    score = {
        "quality": 1.0,
        "compiles": True,
        "tests_green": True,
        "review_ok": True,
        "approvals": 3,
        "qa_verdicts": {},
    }
    score.update(overrides)
    return score


def test_a_qa_rejection_blocks_convergence():
    """A QA lens that saw the measured tree outranks a full blended score.

    Measured on a real run: quality 1.00 (tests 1.0, review 1.0, static 1.0, a
    green suite, a compiling tree, three approvals) while static_qa,
    test_sandbox and edge_case_qa all rejected the candidate patch.  The gate
    read only the score, so the run reported success and the user never saw the
    objection.
    """
    policy = Policy(quality_target=0.8)
    assert _converge(_perfect(qa_verdicts={"static_qa": "approve"}), policy=policy)["status"] == "converged", (
        "precondition: a perfect score still converges"
    )

    disputed = _perfect(qa_verdicts={"static_qa": "reject", "test_sandbox": "reject", "edge_case_qa": "reject"})
    decision = _converge(disputed, policy=policy)
    assert decision["status"] == "continue", "a rejected candidate tree must not converge"
    assert "QA rejected" in decision["reason"]
    assert "edge_case_qa" in decision["reason"], "the report has to name which lens objected"


def test_qa_abstentions_do_not_block():
    """Only a voiced objection stops the loop; a silent lens stays silent."""
    verdicts = {"static_qa": "abstain", "test_sandbox": "abstain", "edge_case_qa": "abstain"}
    assert _converge(_perfect(qa_verdicts=verdicts), policy=Policy(quality_target=0.8))["status"] == "converged"


def test_a_single_cycle_handoff_names_the_real_reason():
    """The hand-off used to hard-code "quality X below target Y" regardless.

    On the run that motivated this it printed the arithmetically impossible
    "quality 1.00 below target 0.80" while three QA agents were rejecting the
    patch, so the run was handed over with no stated objection.
    """
    disputed = _perfect(qa_verdicts={"static_qa": "reject", "test_sandbox": "reject", "edge_case_qa": "reject"})
    decision = _converge(disputed, cycle=0, policy=Policy(quality_target=0.8), allow_continue=False)
    assert decision["status"] == "human_review"
    assert "1.00 below target 0.80" not in decision["reason"], "a false statement is worse than a terse one"
    assert "QA rejected" in decision["reason"]


def test_the_escalation_reason_names_what_blocked_it():
    """ "Budget exhausted" is why the loop stopped, not why it failed."""
    disputed = _perfect(qa_verdicts={"edge_case_qa": "reject"})
    decision = _converge(disputed, cycle=3, max_iterations=3, policy=Policy(quality_target=0.8))
    assert decision["status"] == "escalated"
    assert "QA rejected" in decision["reason"], "an escalated run must carry the objection"
    assert "edge_case_qa" in decision["reason"]


def test_the_blocker_list_is_the_single_source_of_reasons():
    """Every branch must describe the same run the same way."""
    from qedloop.policy import block_reasons

    policy = Policy(quality_target=0.8)
    reasons = block_reasons(
        {"quality": 0.2, "compiles": False, "tests_green": False, "review_ok": False,
         "qa_verdicts": {"static_qa": "reject"}},
        policy,
        cycle=0,
    )
    assert "candidate tree does not compile" in reasons
    assert "test suite not green" in reasons
    assert "no reviewer approval" in reasons
    assert any(r.startswith("quality 0.20 < 0.80") for r in reasons)
    assert any(r.startswith("QA rejected") for r in reasons)
    assert block_reasons(_perfect(qa_verdicts={}), policy, cycle=0) == []


def test_min_iterations_counts_cycles_not_the_cycle_index():
    """``cycle`` indexes the cycle that just finished; ``min_iterations`` counts them.

    Comparing the two directly made the shipped default of 1 mean "at least two
    cycles".  Measured on a real run: one clean cycle, quality 1.00, a green
    suite, five reviewer approvals and all three QA lenses approving -- and the
    run was handed over with the reason "single-cycle mode stopped the loop; "
    and nothing after the semicolon.  The report's own "cycles completed: 1 of 3"
    already counted, so the report and the gate disagreed about how many cycles
    had run.
    """
    policy = Policy()
    assert policy.min_iterations == 1, "precondition: the shipped default"

    decision = _converge(_perfect(qa_verdicts={}), cycle=0, policy=policy, allow_continue=False)

    assert decision["status"] == "converged", decision["reason"]


def test_a_cycle_shortfall_is_named_instead_of_leaving_a_dangling_semicolon():
    """A hand-off with an empty blocker list names no reason at all.

    With the cycle count inside ``block_reasons``, "no blockers" and "converged"
    mean the same thing, so no non-converged branch can print an empty list.
    """
    policy = Policy(min_iterations=3)
    decision = _converge(_perfect(qa_verdicts={}), cycle=0, policy=policy, allow_continue=False)

    assert decision["status"] == "human_review"
    assert "1 of 3" in decision["reason"], "the hand-off has to name the shortfall"
    assert not decision["reason"].rstrip().endswith(";"), (
        "an empty blocker list prints a dangling semicolon: %r" % decision["reason"]
    )


def test_qa_rejection_withholds_the_verified_credit():
    """A patch that only rewrites an assertion must not be recorded as a fix.

    Reproduces a real run: the patch touched tests/integration/test_screenshot_api.py
    and nothing else, the suite stayed green (503 before, 503 after), and the
    issue was marked ``verified`` -- while edge_case_qa was rejecting it for not
    meeting the acceptance contract.  A green suite cannot see the difference
    between fixing behaviour and rewriting the assertion that checks it.
    """
    from qedloop.phases.qa import _resolved_issue_ids

    evidence = {
        "static": {"compiles": True, "markers_resolved": [], "syntax_errors": []},
        "tests": {
            "ran": True,
            "after": {"total": 503, "passed": 503, "failed": 0, "errors": 0},
            "regression": {"new_failures": 0},
            "failing_after": [],
            "repaired_tests": [],
        },
    }
    todo = [{"id": "ISS-0001", "bug_marker": ""}]

    assert _resolved_issue_ids(evidence, todo) == ["ISS-0001"], "precondition: a green suite credits it"
    assert _resolved_issue_ids(evidence, todo, qa_rejected=True) == [], (
        "a QA rejection must withhold the credit"
    )


def test_policy_round_trips_through_a_mapping():
    policy = Policy.from_mapping({"quality_target": 0.9, "weights": {"tests": 0.7, "static": 0.1}, "min_approvals": 2})
    assert policy.quality_target == 0.9 and policy.min_approvals == 2
    assert policy.weights == {"tests": 0.7, "static": 0.1}
    assert Policy.from_mapping(policy.to_dict()).to_dict() == policy.to_dict()


def test_a_retired_weight_in_a_config_file_is_dropped_not_applied():
    """A stale ``review:`` key must not come back in through ``loop.yml``.

    Keeping it would count it into the weight total while blending nothing, so
    the run would score lower than the report footer's own weights claim.
    """
    policy = Policy.from_mapping({"weights": {"tests": 0.5, "review": 0.9, "static": 0.5}})
    assert "review" not in policy.weights, "the retired key must not survive the load"
    assert policy.to_dict()["weights"] == {"tests": 0.5, "static": 0.5}
    assert Policy().to_dict()["weights"] == {"tests": 0.5, "static": 0.2}, "defaults stay unchanged"
