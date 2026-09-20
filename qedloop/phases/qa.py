# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Phase 5 -- QA verification and the convergence decision.

Order of operations matters and is deliberately *measurement first*:

1. the candidate tree is run through the target's real test suite in a scratch
   directory (``sandbox.verification_snapshot``), producing before/after evidence;
2. three QA agents interpret that evidence from three lenses and vote;
3. the deterministic gate (:func:`qedloop.policy.convergence`) combines the
   score with the votes and decides converge / continue / hand off.

The model never gets to declare a green suite; it can only interpret one.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Sequence

from ..core import new_id, utc_now
from ..crew import Crew
from ..policy import Policy, convergence, is_plateau, score_cycle
from ..prompts import QA_VOTING_ROLES, specs_for
from ..sandbox import failing_tests_for_marker, verification_snapshot
from .base import (
    ListenerBox,
    agent_rows,
    batch_issues,
    fresh_batch,
    listener_box,
    make_logger,
    tests_are_green,
    verified_issues,
)


def make_qa_node(crew: Crew, policy: Policy, box: ListenerBox | None = None):
    log = make_logger(box if box is not None else listener_box())

    def qa(state: Mapping[str, Any]) -> Dict[str, Any]:
        cycle = int(state.get("cycle", 0))
        base = dict(state.get("codebase_files") or {})
        if not base:
            base = {f["path"]: f["content"] for f in (state.get("codebase") or {}).get("files", [])}
        candidate = dict(state.get("working_code") or {})
        todo = fresh_batch(state, batch_issues(state))
        patch_error = str(state.get("patch_error") or "")
        max_iterations = int(state.get("max_iterations", 3))

        # --- 1. measurement -------------------------------------------------
        if candidate == base or not todo:
            evidence: Dict[str, Any] = {
                "static": {},
                "tests": {"ran": False, "error": patch_error or "candidate tree is identical to the baseline"},
                "measured_at": utc_now(),
            }
            no_candidate = True
        else:
            log("phase5.measure", {"phase": "qa", "cycle": cycle, "note": "running the target suite against the candidate tree"})
            evidence = verification_snapshot(
                base,
                candidate,
                baseline_root=state.get("target_root") or None,
                run_tests=bool(state.get("run_tests", True)),
                timeout=float(state.get("test_timeout", 120.0)),
                keep_workspaces=bool(state.get("keep_workspaces", False)),
                lint_command=str(state.get("lint_command") or ""),
            )
            evidence["measured_at"] = utc_now()
            no_candidate = False

        # --- 2. interpretation ---------------------------------------------
        interpreted = dict(state)
        interpreted["evidence"] = evidence
        interpreted["selected_patch"] = state.get("selected_patch") or {}
        interpreted["focus_ids"] = [str(todo[0]["id"])] if todo else []
        results: List[Any] = []
        if no_candidate:
            log("phase5.skip_agents", {"phase": "qa", "cycle": cycle, "reason": "no candidate tree to verify"})
        else:
            log(
                "phase5.start",
                {
                    "phase": "qa",
                    "cycle": cycle,
                    "agents": [s.role for s in specs_for("qa")],
                    "tests": {
                        "before": ((evidence.get("tests") or {}).get("before") or {}).get("total"),
                        "after": ((evidence.get("tests") or {}).get("after") or {}).get("total"),
                        "failed_before": ((evidence.get("tests") or {}).get("before") or {}).get("failed"),
                        "failed_after": ((evidence.get("tests") or {}).get("after") or {}).get("failed"),
                    },
                },
            )
            results = crew.run_phase("qa", interpreted)

        checks = agent_rows(results, phase="qa", cycle=cycle, listener=log)
        verdicts = {r.role: (r.verdict if r.ok else "error") for r in results}
        rejected_by_qa = sorted(role for role, verdict in verdicts.items() if verdict == "reject")

        # --- 3. gate --------------------------------------------------------
        score = score_cycle(
            reviews=state.get("reviews") or [],
            evidence=evidence,
            checks=checks,
            policy=policy,
        )
        score["qa_verdicts"] = verdicts
        score["cycle"] = cycle
        score["at"] = utc_now()

        # Which issues this cycle *proved* fixed.  Computed once, after the
        # verdicts, because a QA rejection withholds the credit: crediting an
        # issue while the crew is refusing the tree it was proved in is how a
        # patch that only rewrote an assertion got recorded as a fix.
        resolved_ids = _resolved_issue_ids(evidence, todo, qa_rejected=bool(rejected_by_qa))

        # Progress is "an issue actually closed", not "the score moved": with a
        # handful of defects the blended score can stay flat while each cycle
        # genuinely retires one of them.
        if resolved_ids:
            streak = 0
        elif is_plateau(state.get("quality_history") or [], policy):
            streak = int(state.get("no_progress_cycles") or 0) + 1
        else:
            streak = 0

        if no_candidate:
            decision = {
                "status": "human_review" if patch_error else "converged",
                "reason": patch_error or "nothing left to change: no open issue produced a candidate tree",
                "quality": 0.0,
            }
            score["quality"] = 0.0
        else:
            decision = convergence(
                score,
                cycle=cycle,
                max_iterations=max_iterations,
                history=state.get("quality_history") or [],
                policy=policy,
                allow_continue=bool(state.get("allow_multi_cycle", True)),
                no_progress_cycles=streak,
            )
        # Copy the verdict onto the history row so the persisted quality history
        # explains itself without cross-referencing the cycles ledger.
        score["status"] = decision["status"]
        score["reason"] = decision["reason"]

        log(
            "phase5.verdict",
            {
                "phase": "qa",
                "cycle": cycle,
                "decision": decision["status"],
                "reason": decision["reason"],
                "quality": score.get("quality"),
                "components": score.get("components"),
                "tests_green": score.get("tests_green"),
                "review_ok": score.get("review_ok"),
                "votes": verdicts,
                "qa_rejected": rejected_by_qa,
            },
        )

        delta: Dict[str, Any] = {
            "evidence": evidence,
            "checks": checks + [
                {
                    "id": new_id("check"),
                    "phase": "qa",
                    "role": "gate",
                    "ok": decision["status"] == "converged",
                    "verdict": decision["status"],
                    "cycle": cycle,
                    "note": decision["reason"],
                }
            ],
            "qa_votes": [
                {
                    "id": new_id("check"),
                    "role": role,
                    "verdict": verdict,
                    "cycle": cycle,
                    "phase": "qa",
                }
                for role, verdict in verdicts.items()
            ],
            "verifications": [
                {
                    "id": new_id("check"),
                    "issue_id": issue["id"],
                    "cycle": cycle,
                    "status": "verified" if issue["id"] in resolved_ids else "still_open",
                    "quality": score.get("quality"),
                }
                for issue in todo
            ],
            "quality_history": (state.get("quality_history") or []) + [score],
            "no_progress_cycles": streak,
            "cycles": [
                {
                    "id": new_id("cycle"),
                    "index": cycle,
                    "quality": score.get("quality"),
                    "components": score.get("components"),
                    "tests_green": score.get("tests_green"),
                    "review_ok": score.get("review_ok"),
                    "status": decision["status"],
                    "reason": decision["reason"],
                    "at": score["at"],
                }
            ],
            "status": decision["status"],
            "status_reason": decision["reason"],
        }

        if resolved_ids:
            delta["issues"] = [{"id": i, "status": "verified", "verified_cycle": cycle, "verified_at": utc_now()} for i in resolved_ids]
            delta["batch_verified"] = resolved_ids
            delta["needs_discovery"] = True
            log("phase5.verified", {"phase": "qa", "cycle": cycle, "issue_ids": resolved_ids})

        if decision["status"] == "continue":
            delta["cycle"] = cycle + 1
        return delta

    return qa


def _resolved_issue_ids(
    evidence: Mapping[str, Any],
    todo: Sequence[Mapping[str, Any]],
    qa_rejected: bool = False,
) -> List[str]:
    """Which issues this cycle *proved* fixed, from measurement only.

    A fix is credited only when two independent measurements agree:

    * the declared defect the issue reported is no longer present in the
      candidate tree (its marker is gone), **and**
    * the suite made real progress: at least one previously failing test now
      passes, and the patch introduced no new failure.

    Neither half is sufficient alone.  A marker can vanish in a refactor that
    changes nothing observable, and a green test can be a coincidence of the
    patch; together they are the strongest evidence available without a human.

    Deliberately *not* required: the whole suite being green.  With several
    independent defects in one repository, demanding a fully green suite before
    crediting any single fix would make the loop unable to make progress -- it
    would keep re-patching the same already-fixed file until the budget ran out.

    Issues that reported no declared marker need a green suite instead, and are
    flagged in the report as the weaker case.

    ``qa_rejected`` withholds every credit for this cycle.  A QA lens speaks only
    after the candidate tree has been measured, so its refusal is the one verdict
    that has seen the artifact; without this, "the suite is still green" was
    enough to mark an issue fixed even when the patch had merely rewritten an
    assertion and changed no behaviour at all.
    """
    if qa_rejected:
        return []

    static = evidence.get("static") or {}
    tests = evidence.get("tests") or {}
    after = tests.get("after") or {}
    regression = tests.get("regression") or {}

    resolved_markers = {str(m.get("key") or m.get("name")) for m in static.get("markers_resolved") or []}
    repaired = {str(t) for t in tests.get("repaired_tests") or []}
    still_failing = [str(t) for t in tests.get("failing_after") or []]

    ran = bool(tests.get("ran"))
    no_new_failures = int(regression.get("new_failures") or 0) == 0
    suite_green = ran and int(after.get("failed") or 0) == 0 and int(after.get("errors") or 0) == 0
    compiles = bool(static.get("compiles"))
    measured = bool(static) or ran

    if not measured or not compiles or not no_new_failures:
        return []

    out: List[str] = []
    for issue in todo:
        marker = str(issue.get("bug_marker") or "")
        if marker:
            if marker not in resolved_markers:
                continue
            if failing_tests_for_marker(marker, still_failing):
                continue
            if repaired or suite_green:
                out.append(str(issue.get("id")))
        elif suite_green:
            out.append(str(issue.get("id")))
    return out
