# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Phase 3 -- code review through three lenses, plus the plan gate.

Architecture and correctness reviewers *vote*; the risk reviewer contributes a
note and the blast radius.  The gate itself is deterministic
(:func:`qedloop.policy.review_route`) and produces the loop drawn in the
architecture diagram:

``review --reject--> refine``  and  ``review --approve--> patch``
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from ..core import new_id
from ..crew import Crew
from ..graph import END
from ..plans import check_plan
from ..policy import review_route
from ..prompts import plans_for, specs_for
from .base import (
    ListenerBox,
    agent_rows,
    batch_issues,
    fresh_batch,
    latest_finding,
    listener_box,
    make_logger,
)


def make_review_node(crew: Crew, policy: Any, box: ListenerBox | None = None):
    log = make_logger(box if box is not None else listener_box())

    def review(state: Mapping[str, Any]) -> Dict[str, Any]:
        cycle = int(state.get("cycle", 0))
        self_loops = int(state.get("self_loop_count", 0))
        todo = fresh_batch(state, batch_issues(state))
        checks: List[Dict[str, Any]] = []
        reviews: List[Dict[str, Any]] = []
        gate_notes: List[Dict[str, Any]] = []

        log(
            "phase3.start",
            {
                "phase": "review",
                "cycle": cycle,
                "agents": [s.role for s in specs_for("review")],
                "targets": [i["id"] for i in todo],
                "attempt": self_loops + 1,
            },
        )

        for issue in todo:
            issue_id = str(issue["id"])
            # Mechanical checks first.  A plan whose own commands cannot collect
            # the tests it names is rejected without spending three model calls
            # and a whole graph step on it -- a measured run spent three of four
            # review rounds on exactly that.  The objection is raised as a review
            # row from the `plan-consistency` lens, so `refine_prompt` sees it in
            # its blockers next round and `review_route` charges it to the back
            # edge budget like any other rejection: no second budget to tune, and
            # no way for a broken plan to ping-pong for free.
            problems = check_plan(
                latest_finding(state, issue_id),
                plans_for(state, issue_id),
                code=state.get("working_code") or {},
            )
            if problems:
                rows = [
                    {
                        "id": new_id("review"),
                        "issue_id": issue_id,
                        "lens": "plan-consistency",
                        "verdict": "reject",
                        "confidence": 1.0,
                        "blocking": [{"issue_id": issue_id, "reason": problem} for problem in problems],
                        "notes": "checked mechanically against the plan and the candidate tree; no agent was called",
                        "cycle": cycle,
                    }
                ]
                log(
                    "phase3.plan_inconsistent",
                    {
                        "phase": "review",
                        "cycle": cycle,
                        "issue_id": issue_id,
                        "problems": problems[:4],
                        "agents_skipped": len(specs_for("review")),
                    },
                )
            else:
                results = crew.run_phase("review", state, focus=issue)
                checks.extend(agent_rows(results, phase="review", cycle=cycle, listener=log))
                rows = crew.reviews(results, cycle=cycle, issue_id=issue_id)

            reviews.extend(rows)

            route = review_route(rows, self_loop_count=self_loops, policy=policy)
            # Route on the batch as a whole: any rejection sends the cycle back to
            # refinement, because a half-reviewed plan must not be implemented.
            if route["decision"] != "approve":
                decision = route
                break
            gate_notes.append({"issue_id": issue_id, **route})
        else:
            decision = {
                "decision": "approve",
                "reason": "every reviewed issue cleared the gate",
                "blocking": [],
            }

        log(
            "phase3.verdict",
            {
                "phase": "review",
                "cycle": cycle,
                "decision": decision["decision"],
                "reason": decision["reason"],
                "blocking": decision.get("blocking", [])[:4],
                "votes": {r.get("lens"): r.get("verdict") for r in reviews},
            },
        )

        delta: Dict[str, Any] = {
            "reviews": reviews,
            "checks": checks + [
                {
                    "id": new_id("check"),
                    "phase": "review",
                    "role": "gate",
                    "ok": decision["decision"] == "approve",
                    "verdict": decision["decision"],
                    "cycle": cycle,
                    "note": decision["reason"],
                }
            ],
            "review_decision": decision["decision"],
            "review_reason": decision["reason"],
            "review_blocking": decision.get("blocking", []),
        }
        if decision["decision"] == "refine":
            delta["self_loop_count"] = self_loops + 1
        if decision["decision"] == "stop":
            delta["status"] = "human_review"
            delta["status_reason"] = decision["reason"]
        if not todo:
            delta["status"] = "human_review"
            delta["status_reason"] = "nothing to review"
        return delta

    return review
