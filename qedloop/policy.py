# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Convergence policy: when to loop, when to stop.

Two independent decisions live here, and keeping them separate is what makes
the loop debuggable:

* **Phase 3 gate** -- may this *plan* be implemented?  (``review_route``)
* **Phase 5 gate** -- is the *result* good enough to stop?  (``convergence``)

Both are pure functions of the state bus, so they can be unit tested without an
LLM and replayed against a trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

# --------------------------------------------------------------------------- #
# tunables (mirrored by examples/loop.buggy_service.yml)
# --------------------------------------------------------------------------- #

MIN_APPROVALS = 1          # Phase 3: approvals needed to let the plan through
MAX_SELF_LOOPS = 3         # Phase 3: review -> refine back edges before forcing through
QUALITY_TARGET = 0.80      # Phase 5: stop threshold
MIN_ITERATIONS = 1         # never declare victory before this many cycles
PLATEAU_EPSILON = 0.02     # quality gain below this counts as a plateau
MAX_NO_PROGRESS = 2        # consecutive plateau cycles before hand-off

WEIGHTS = {"tests": 0.5, "static": 0.2}

# ``review`` was a weighted component (0.3) and is now a hard requirement only.
# Whether the reviewers approved already gates the loop through ``review_ok`` in
# :func:`convergence`, and a hard requirement must not also ride in the weighted
# score -- counting it twice is what let the score contradict the gate sitting
# next to it.  Measured on a real run: 12 review rows over four rounds (6
# approve / 6 reject), the Phase 3 gate approved on the last round 3/3, and the
# score still read review 0.00 because it divided the whole append-only ledger
# instead of the round that mattered.
#
# Scoring only the last round would not have fixed it: ``review_route`` returns
# ``refine`` whenever a rejection is voiced, so any cycle that reaches Phase 4
# has zero rejections in that round and the component is 1.0 by construction --
# a constant that only dilutes the measured evidence.
RETIRED_WEIGHTS = ("review",)
WEIGHTED_COMPONENTS = ("tests", "static")


@dataclass
class Policy:
    quality_target: float = QUALITY_TARGET
    min_approvals: int = MIN_APPROVALS
    max_self_loops: int = MAX_SELF_LOOPS
    min_iterations: int = MIN_ITERATIONS
    plateau_epsilon: float = PLATEAU_EPSILON
    max_no_progress: int = MAX_NO_PROGRESS
    weights: Dict[str, float] = field(default_factory=lambda: dict(WEIGHTS))
    require_tests: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "Policy":
        data = dict(data or {})
        weights = dict(WEIGHTS)
        weights.update(
            # A retired key must not come back in through a stale config file:
            # it would be counted into ``total_weight`` while blending nothing,
            # so the run would score lower than the report's own footer claims.
            # Dropping it here keeps ``to_dict()`` showing the weights that
            # actually applied.
            (str(k), float(v))
            for k, v in (data.get("weights") or {}).items()
            if str(k) not in RETIRED_WEIGHTS
        )
        policy = cls(
            quality_target=float(data.get("quality_target", QUALITY_TARGET)),
            min_approvals=int(data.get("min_approvals", MIN_APPROVALS)),
            max_self_loops=int(data.get("max_self_loops", MAX_SELF_LOOPS)),
            min_iterations=int(data.get("min_iterations", MIN_ITERATIONS)),
            plateau_epsilon=float(data.get("plateau_epsilon", PLATEAU_EPSILON)),
            max_no_progress=int(data.get("max_no_progress", MAX_NO_PROGRESS)),
            require_tests=bool(data.get("require_tests", True)),
            weights=weights,
        )
        return policy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality_target": self.quality_target,
            "min_approvals": self.min_approvals,
            "max_self_loops": self.max_self_loops,
            "min_iterations": self.min_iterations,
            "plateau_epsilon": self.plateau_epsilon,
            "max_no_progress": self.max_no_progress,
            "require_tests": self.require_tests,
            "weights": dict(self.weights),
        }


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_cycle(
    *,
    reviews: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    policy: Policy,
) -> Dict[str, Any]:
    """Blend measured evidence and static facts into 0..1.

    Three hard requirements are kept *outside* the score so a high style score
    can never paper over them:

    * the candidate tree compiles (``compiles``)
    * every test in the candidate tree passes (``tests_green``)
    * the plan cleared the Phase 3 gate (``review_ok``)

    All three are also required by :func:`convergence`, because "the candidate
    tree does not even parse" must never be reported as a converged run.  They
    are *only* requirements: a requirement that is also weighted is counted
    twice, and the weight keeps moving after the requirement is already
    satisfied (see ``WEIGHTS``).
    """
    tests = (evidence or {}).get("tests") or {}
    after = tests.get("after") or {}
    regression = tests.get("regression") or {}
    static = (evidence or {}).get("static") or {}

    if not tests.get("ran", False):
        test_component = 0.0
        tests_green = not policy.require_tests
    else:
        total = int(after.get("total") or 0)
        failing = int(after.get("failed") or 0) + int(after.get("errors") or 0)
        run_component = (int(after.get("passed") or 0) / total) if total else 0.0
        no_new_failures = int(regression.get("new_failures") or 0) == 0
        test_component = _clamp(run_component * (1.0 if no_new_failures else 0.5))
        tests_green = bool(total) and failing == 0 and no_new_failures

    # Votes are reported as counts and gate through ``review_ok``; they are not
    # blended into ``quality`` (see ``WEIGHTS``).
    approvals = sum(1 for r in reviews or [] if str(r.get("verdict")) == "approve")
    rejections = sum(1 for r in reviews or [] if str(r.get("verdict")) == "reject")
    review_ok = approvals >= policy.min_approvals

    if not static:
        static_component = 0.0
        compiles = False
    else:
        static_component = 1.0 if static.get("compiles") else 0.0
        static_component -= 0.15 * len(static.get("markers_introduced") or [])
        static_component = _clamp(static_component)
        compiles = bool(static.get("compiles"))

    weights = policy.weights
    total_weight = sum(weights.get(k, 0.0) for k in WEIGHTED_COMPONENTS) or 1.0
    quality = (
        test_component * weights.get("tests", 0.0)
        + static_component * weights.get("static", 0.0)
    ) / total_weight

    # The target's own static check, when the run configured one.  A requirement,
    # never a component: a rule violation is not a matter of degree, and a run
    # that could not evaluate a configured gate has not passed it.  Absent means
    # "not configured", which blocks nothing.
    lint = dict((evidence or {}).get("lint") or {})
    lint_ok = True if not lint.get("configured") else bool(lint.get("ok"))

    return {
        "quality": round(_clamp(quality), 4),
        "components": {
            "tests": round(test_component, 4),
            "static": round(static_component, 4),
        },
        "lint_ok": bool(lint_ok),
        # Compact on purpose: this rides into ``quality_history``, and what a
        # reason string needs is which command ran and how it ended, not its
        # output (that stays in ``evidence``).
        "lint": {
            "configured": bool(lint.get("configured")),
            "ok": bool(lint.get("ok")),
            "command": [str(part) for part in (lint.get("command") or [])],
            "exit_code": lint.get("exit_code"),
            "error": str(lint.get("error") or ""),
        },
        "approvals": approvals,
        "rejections": rejections,
        "abstentions": len([r for r in (reviews or []) if str(r.get("verdict")) == "abstain"]),
        "compiles": compiles,
        "tests_green": bool(tests_green),
        "review_ok": bool(review_ok),
        "markers_resolved": len(static.get("markers_resolved") or []),
        "markers_introduced": len(static.get("markers_introduced") or []),
        "qa_verdicts": {str(c.get("role")): str(c.get("verdict")) for c in (checks or [])},
    }


# --------------------------------------------------------------------------- #
# Phase 3 gate
# --------------------------------------------------------------------------- #


def review_route(
    reviews: Sequence[Mapping[str, Any]],
    *,
    self_loop_count: int,
    policy: Policy,
) -> Dict[str, Any]:
    """``approve`` -> Phase 4, ``refine`` -> back to Phase 2, ``stop`` -> END."""
    approvals = [r for r in reviews or [] if str(r.get("verdict")) == "approve"]
    rejections = [r for r in reviews or [] if str(r.get("verdict")) == "reject"]
    abstentions = [r for r in reviews or [] if str(r.get("verdict")) == "abstain"]

    if rejections:
        # A voiced objection always wins over a bare approval: silence is not
        # consent, and an unrebutted finding must be answered before patching.
        if self_loop_count >= policy.max_self_loops:
            # ``len(rejections)`` is how many lenses objected *in this round*, and
            # the budget it exhausted was spent across the whole run --
            # ``self_loop_count`` is never reset.  The old wording ("rejected 2
            # time(s) and the refine budget (3) is spent") read like one attempt
            # remained, while three rounds had in fact been rejected.
            return {
                "decision": "stop",
                "reason": "this round was rejected by %d lens(es); the run's refine budget "
                          "(%d review->refine back edges, counted across the whole run) is spent"
                % (len(rejections), policy.max_self_loops),
                "blocking": _blocking(rejections),
            }
        return {
            "decision": "refine",
            "reason": "rejected by %s" % ", ".join(str(r.get("lens")) for r in rejections),
            "blocking": _blocking(rejections),
        }
    if len(approvals) >= policy.min_approvals:
        return {
            "decision": "approve",
            "reason": "%d/%d reviewers approved (need %d)" % (len(approvals), len(reviews), policy.min_approvals),
            "blocking": [],
        }
    return {
        "decision": "refine",
        "reason": "no reviewer approved (%d abstained): insufficient information to proceed" % len(abstentions),
        "blocking": [],
    }


def _blocking(rejections: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for review in rejections:
        for item in review.get("blocking") or []:
            out.append({"lens": review.get("lens"), **dict(item)})
        if not review.get("blocking") and review.get("notes"):
            out.append({"lens": review.get("lens"), "reason": str(review.get("notes"))[:300]})
    return out


# --------------------------------------------------------------------------- #
# Phase 5 gate
# --------------------------------------------------------------------------- #


def block_reasons(score: Mapping[str, Any], policy: Policy, *, cycle: int) -> List[str]:
    """Every reason this cycle may not converge, in a fixed order.

    Kept separate from :func:`convergence` because more than one branch needs it.
    The single-cycle hand-off used to hard-code "quality X below target Y" as its
    reason, which printed the arithmetically impossible "quality 1.00 below
    target 0.80" while three QA agents were rejecting the candidate tree -- the
    real objection was never named.  Everything that blocks a cycle belongs in
    this one list so no branch can describe the run differently from another.

    ``cycle`` is a required argument rather than a key read out of ``score``: the
    cycle count is itself a condition of convergence (``min_iterations``), and a
    caller that forgot to pass it would silently drop one reason -- which is how
    a hand-off came to print "single-cycle mode stopped the loop; " with nothing
    after the semicolon.  With the shortfall in this list, "no blockers" and
    "converged" mean the same thing, so every non-converged branch has something
    to say.
    """
    reasons: List[str] = []
    if not bool(score.get("compiles")):
        reasons.append("candidate tree does not compile")
    if not bool(score.get("tests_green")):
        reasons.append("test suite not green")
    lint = score.get("lint") or {}
    if lint.get("configured") and not bool(score.get("lint_ok", True)):
        # Named down to the command: the crew cannot act on "lint failed", and
        # invariant: every convergence condition has to be attributable here.
        detail = str(lint.get("error") or ("exit %s" % lint.get("exit_code")))
        reasons.append(
            "target lint did not pass (%s -> %s)"
            % (" ".join(str(part) for part in (lint.get("command") or ["?"])), detail)
        )
    if not bool(score.get("review_ok")):
        reasons.append("no reviewer approval")
    if float(score.get("quality") or 0.0) < policy.quality_target:
        reasons.append("quality %.2f < %.2f" % (float(score.get("quality") or 0.0), policy.quality_target))
    rejects = sorted(
        str(role)
        for role, verdict in (score.get("qa_verdicts") or {}).items()
        if str(verdict) == "reject"
    )
    if rejects:
        # A QA lens that examined the measured candidate tree and refused it
        # outranks any blended score: the score is computed from a green suite,
        # three approving reviewers and a clean compile, none of which can see
        # that the patch fails its own acceptance contract.  Measured on one
        # run: quality 1.00 with static_qa, test_sandbox and edge_case_qa all
        # rejecting, reported to the user as a full score.
        reasons.append("QA rejected the candidate tree (%s)" % ", ".join(rejects))
    # ``cycle`` indexes the cycle that just finished (0-based), while
    # ``min_iterations`` counts cycles.  Comparing the two directly made the
    # shipped default of 1 mean "at least two cycles": measured on a real run,
    # one clean cycle scored 1.00 with a green suite, five reviewer approvals and
    # all three QA lenses approving, and the run was still handed over with the
    # reason "single-cycle mode stopped the loop; " and nothing after it.
    cycles_done = int(cycle) + 1
    if cycles_done < policy.min_iterations:
        reasons.append(
            "only %d of %d required cycle(s) completed" % (cycles_done, policy.min_iterations)
        )
    return reasons


def convergence(
    score: Mapping[str, Any],
    *,
    cycle: int,
    max_iterations: int,
    history: Sequence[Mapping[str, Any]],
    policy: Policy,
    allow_continue: bool = True,
    no_progress_cycles: int = 0,
) -> Dict[str, Any]:
    """Decide ``converged`` / ``continue`` / ``human_review`` / ``escalated``.

    ``no_progress_cycles`` is the caller's count of consecutive cycles that
    closed no issue *and* moved quality by less than the epsilon.  A cycle that
    verifies a defect is progress even when the blended score barely moves, so
    the plateau rule is driven by that counter rather than by the score alone.
    """
    quality = float(score.get("quality") or 0.0)
    tests_green = bool(score.get("tests_green"))
    review_ok = bool(score.get("review_ok"))
    compiles = bool(score.get("compiles"))
    lint_ok = bool(score.get("lint_ok", True))
    blockers = block_reasons(score, policy, cycle=cycle)

    # ``cycle`` indexes the cycle that just finished, so the number of finished
    # cycles is one more than the index.  ``min_iterations`` counts cycles, and
    # the report does too ("cycles completed" is ``len(history)``); comparing the
    # index against the count is what made the default of 1 mean two cycles.
    cycles_done = cycle + 1

    if (
        not blockers
        and cycles_done >= policy.min_iterations
        and tests_green
        and review_ok
        and compiles
        and lint_ok
        and quality >= policy.quality_target
    ):
        return {
            "status": "converged",
            "reason": "quality %.2f >= target %.2f with a compiling tree, a green suite and %d approvals"
            % (quality, policy.quality_target, int(score.get("approvals") or 0)),
            "quality": quality,
        }

    if cycles_done >= max_iterations:
        # The blocker list rides along here too: "budget exhausted" explains why
        # the loop stopped, never why it failed, and a user handed an escalated
        # run needs the objection that kept it from converging.
        #
        # ``cycles_done`` and not ``cycle``: the budget counts cycles while
        # ``cycle`` indexes the one that just finished, so comparing the index
        # bought the loop one cycle it was not granted -- ``max_iterations: 3``
        # executed four, and the report printed "cycles completed | 4 of 3".
        # Exactly the trap ``min_iterations`` above fell into; both now count
        # finished cycles.
        return {
            "status": "escalated",
            "reason": "iteration budget exhausted at quality %.2f (target %.2f), compiles=%s tests_green=%s; %s"
            % (quality, policy.quality_target, compiles, tests_green, "; ".join(blockers)),
            "quality": quality,
        }

    if no_progress_cycles >= policy.max_no_progress:
        return {
            "status": "human_review",
            "reason": "%d consecutive cycles closed no issue and moved quality by less than %.2f; the crew is stuck"
            % (no_progress_cycles, policy.plateau_epsilon),
            "quality": quality,
        }

    if not allow_continue:
        return {
            "status": "human_review",
            "reason": "single-cycle mode stopped the loop; " + "; ".join(blockers),
            "quality": quality,
        }

    return {
        "status": "continue",
        "reason": "; ".join(blockers) or "quality below target",
        "quality": quality,
    }


def is_plateau(history: Sequence[Mapping[str, Any]], policy: Policy) -> bool:
    """True when the last two quality readings are within ``plateau_epsilon``."""
    values = [float(row.get("quality") or 0.0) for row in history or []]
    if len(values) < 2:
        return False
    return abs(values[-1] - values[-2]) < policy.plateau_epsilon
