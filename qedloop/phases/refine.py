# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Phase 2 -- requirement refinement.

Each issue in the batch gets one pass with three agents: one states the
requirement so it can be proven false, one decomposes it into ordered steps, and
one designs the test that will settle it.  The refined requirement is the object
Phase 3 reviews and Phase 4 implements, so it is stored as its own ledger row
keyed by ``issue_id`` (a re-refinement supersedes the previous version).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from ..core import new_id
from ..crew import Crew
from ..prompts import specs_for
from .base import ListenerBox, agent_rows, batch_issues, fresh_batch, listener_box, make_logger

#: The refine phase runs in two passes.  The synthesis states the requirement
#: (decision, acceptance criteria, non-goals) and must land before the two
#: planning lenses, because those criteria are binding constraints on their
#: output.  Kept as ordered tuples rather than inline literals so the split is
#: visible from the call site.
SYNTHESIS_ROLES = ("refine_synthesize",)
PLANNING_ROLES = ("plan_split", "test_design")


def make_refine_node(crew: Crew, policy: Any = None, box: ListenerBox | None = None):
    log = make_logger(box if box is not None else listener_box())

    def refine(state: Mapping[str, Any]) -> Dict[str, Any]:
        cycle = int(state.get("cycle", 0))
        todo = fresh_batch(state, batch_issues(state))
        log(
            "phase2.start",
            {
                "phase": "refine",
                "cycle": cycle,
                "agents": [s.role for s in specs_for("refine")],
                "targets": [i["id"] for i in todo],
                "self_loop": int(state.get("self_loop_count", 0)),
            },
        )

        findings: List[Dict[str, Any]] = []
        plans: List[Dict[str, Any]] = []
        checks: List[Dict[str, Any]] = []
        focus_map: Dict[str, str] = {}

        for issue in todo:
            results = crew.run_phase("refine", state, focus=issue, roles=SYNTHESIS_ROLES)
            synthesis = next((r for r in results if r.role == "refine_synthesize" and r.ok), None)
            log(
                "phase2.synthesis",
                {
                    "phase": "refine",
                    "cycle": cycle,
                    "issue_id": issue["id"],
                    "ok": synthesis is not None,
                },
            )
            # The planning lenses run after the synthesis, not beside it.  They
            # have to see the acceptance criteria and non-goals it just wrote:
            # running them in parallel is how the round produced a plan that
            # violated a requirement stated in the same breath, which the review
            # then rejected -- correctly -- three rounds running.
            results += crew.run_phase(
                "refine",
                state,
                focus=issue,
                roles=PLANNING_ROLES,
                synthesis=(synthesis.data if synthesis is not None else None),
            )
            checks.extend(agent_rows(results, phase="refine", cycle=cycle, listener=log))
            produced = crew.findings(results, issue_id=str(issue["id"]), cycle=cycle)
            findings.extend(produced)
            produced_plans = crew.test_plans(
                results,
                issue_id=str(issue["id"]),
                # The stamp ties this round's plans to this round's requirement,
                # which is what lets the ledger retire the previous generation.
                refined_at=str(produced[0].get("refined_at", "")) if produced else "",
            )
            plans.extend(produced_plans)
            if produced:
                focus_map[str(issue["id"])] = str(produced[0]["id"])
            log(
                "phase2.refined",
                {
                    "phase": "refine",
                    "cycle": cycle,
                    "issue_id": issue["id"],
                    "decision": (produced[0].get("decision") if produced else "unknown"),
                    "acceptance": len(produced[0].get("acceptance", [])) if produced else 0,
                    "tests": len(produced_plans),
                },
            )

        delta: Dict[str, Any] = {
            "findings": findings,
            "test_plans": plans,
            "refine_votes": checks,
            "checks": checks,
        }
        if not todo:
            delta["checks"] = checks + [
                {
                    "id": new_id("check"),
                    "phase": "refine",
                    "role": "gate",
                    "ok": True,
                    "verdict": "skip",
                    "cycle": cycle,
                    "note": "nothing refined: batch is empty or every issue is deferred/rejected",
                }
            ]
            delta["status"] = "human_review"
            delta["status_reason"] = "no actionable issue survived refinement"
        return delta

    return refine
