# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Phase 4 -- produce the change.

Two agents write competing patches (minimal fix vs. remove-the-defect-class), a
third reconciles them, and then the deterministic reconciler in
:func:`qedloop.sandbox.apply_ops` applies the winning ops to a *candidate tree*
held on the state bus.  Nothing touches the real filesystem here: the candidate
only becomes an applied change after Phase 5 accepts it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from ..core import ChangeProposal, diff_summary, new_id, strip_marker
from ..crew import AgentTask, Crew
from ..prompts import find_spec, specs_for
from ..sandbox import apply_ops
from .base import ListenerBox, agent_rows, batch_issues, fresh_batch, listener_box, make_logger

MAX_OP_ATTEMPTS = 2


def make_patch_node(crew: Crew, policy: Any = None, box: ListenerBox | None = None):
    log = make_logger(box if box is not None else listener_box())

    def patch(state: Mapping[str, Any]) -> Dict[str, Any]:
        cycle = int(state.get("cycle", 0))
        todo = fresh_batch(state, batch_issues(state))
        if state.get("review_decision") != "approve" or not todo:
            log(
                "phase4.blocked",
                {
                    "phase": "patch",
                    "cycle": cycle,
                    "reason": "no approved plan" if state.get("review_decision") != "approve" else "nothing in the batch",
                },
            )
            return {
                "checks": [
                    {
                        "id": new_id("check"),
                        "phase": "patch",
                        "role": "gate",
                        "ok": False,
                        "verdict": "blocked",
                        "cycle": cycle,
                        "note": "patch phase reached without an approved plan",
                    }
                ],
                "patch_error": "no approved plan to implement",
            }

        issue = todo[0]
        generators = [s.role for s in specs_for("patch") if s.mode != "synthesize"]
        log(
            "phase4.start",
            {
                "phase": "patch",
                "cycle": cycle,
                "agents": generators,
                "issue_id": issue["id"],
            },
        )

        checks: List[Dict[str, Any]] = []
        candidates: List[Dict[str, Any]] = []
        attempts = 0
        # The reconcile lens is a tool this node calls *with candidates* (see
        # _reconcile), not a member of the fan-out: asking it to merge nothing
        # costs a call and invites an answer about a context it was never given
        # ("no candidate patches ... were present in the context") -- which is
        # also what used to crash the run on a non-numeric confidence.

        while attempts < MAX_OP_ATTEMPTS:
            attempts += 1
            results = crew.run_phase("patch", state, focus=issue, roles=generators)
            checks.extend(agent_rows(results, phase="patch", cycle=cycle, listener=log))
            candidates = _collect(crew, results, issue)
            if any(c.get("ops") for c in candidates):
                break
            log(
                "phase4.no_ops",
                {
                    "phase": "patch",
                    "cycle": cycle,
                    "attempt": attempts,
                    "note": "no anchored edit was produced; asking again",
                },
            )

        if not candidates or not any(c.get("ops") for c in candidates):
            note = "the crew could not express a fix as an anchored edit"
            log("phase4.failed", {"phase": "patch", "cycle": cycle, "note": note})
            return {
                "patches": candidates,
                "checks": checks,
                "patch_error": note,
                "status": "human_review",
                "status_reason": note,
            }

        chosen, reconcile_note = _reconcile(crew, state, candidates)
        proposal = ChangeProposal.from_dict(chosen)
        proposal.issue_ids = [str(issue["id"])]
        if not proposal.id:
            proposal.id = new_id("patch")

        if reconcile_note:
            checks.append(
                {
                    "id": new_id("check"),
                    "phase": "patch",
                    "role": "patch_reconcile",
                    "ok": True,
                    "verdict": str(reconcile_note.get("pick", "A")),
                    "cycle": cycle,
                    "note": str(reconcile_note.get("reason", "")),
                }
            )

        working_code = dict(state.get("working_code") or {})
        applied_paths = [str(op.path) for op in proposal.ops]
        result = apply_ops(working_code, [op.to_dict() for op in proposal.ops])
        candidate = result.code
        stripped: List[str] = []
        if result.ok:
            # A fixed defect must not keep advertising itself: retract the
            # declaration so the next verification pass reads the tree honestly.
            marker = str(proposal.bug_marker or issue.get("bug_marker") or "")
            cleaned = strip_marker(candidate, marker, applied_paths)
            stripped = sorted(p for p in cleaned if cleaned.get(p) != candidate.get(p))
            candidate = cleaned
        log(
            "phase4.applied" if result.ok else "phase4.apply_failed",
            {
                "phase": "patch",
                "cycle": cycle,
                "patch_id": proposal.id,
                "strategy": proposal.strategy,
                "files": sorted({str(a.get("path")) for a in result.applied}) if result.ok else [],
                "markers_retracted": stripped,
                "errors": result.errors[:4],
            },
        )

        patch_record = proposal.to_dict()
        patch_record["issue_id"] = str(issue["id"])
        patch_record["cycle"] = cycle
        patch_record["apply"] = result.to_dict()
        patch_record["accepted"] = False

        change_record = {
            "id": new_id("change"),
            "patch_id": proposal.id,
            "issue_id": str(issue["id"]),
            "cycle": cycle,
            "strategy": proposal.strategy,
            "files": sorted({str(a.get("path")) for a in result.applied}) if result.ok else [],
            "markers_retracted": stripped,
            "ok": result.ok,
            "errors": result.errors,
            "rationale": proposal.rationale,
            "expected_effect": proposal.expected_effect,
            "diff": diff_summary(working_code, candidate) if result.ok else "",
        }

        delta: Dict[str, Any] = {
            "patches": [patch_record],
            "changes": [change_record],
            "selected_patch": patch_record,
            "checks": checks,
        }
        if result.ok:
            delta["working_code"] = candidate
        else:
            delta["patch_error"] = "; ".join(result.errors)
            delta["status"] = "human_review"
            delta["status_reason"] = "candidate patch could not be applied cleanly: %s" % "; ".join(result.errors[:2])
        return delta

    return patch


def _collect(crew: Crew, results: List[Any], issue: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for result in results:
        spec = find_spec(result.role)
        if spec.phase != "patch" or spec.mode == "synthesize" or not result.ok:
            continue
        ops = [op for op in (result.data.get("ops") or []) if isinstance(op, Mapping)]
        proposal = ChangeProposal.from_dict(
            {
                "id": new_id("patch"),
                "issue_ids": [str(issue["id"])],
                "strategy": "minimal" if spec.lens == "minimal-fix" else "refactor",
                "ops": ops,
                "rationale": str(result.data.get("rationale", "")),
                "risk": str(result.data.get("risk", "medium")),
                "expected_effect": str(result.data.get("expected_effect", "")),
                "bug_marker": str(result.data.get("bug_marker", "")),
            }
        )
        row = proposal.to_dict()
        row["agent"] = result.role
        row["lens"] = result.lens
        out.append(row)
    return out


def _reconcile(crew: Crew, state: Mapping[str, Any], candidates: List[Dict[str, Any]]):
    """Ask the reconciliation agent, but keep a deterministic fallback."""
    usable = [c for c in candidates if c.get("ops")]
    fallback = usable[0]
    if len(usable) < 2:
        return fallback, {}
    result = crew.run_one(_reconcile_task(state, usable))
    if not result.ok:
        return fallback, {"pick": "A", "reason": "reconciliation agent unavailable; fell back to the minimal patch"}
    pick = str(result.data.get("pick", "A")).upper()
    data = result.data
    if pick == "B":
        chosen = usable[1]
        chosen = dict(chosen)
        chosen["strategy"] = str(data.get("strategy") or "refactor")
        return chosen, data
    return fallback, data


def _reconcile_task(state: Mapping[str, Any], candidates: List[Dict[str, Any]]):
    spec = find_spec("patch_reconcile")
    summary = [
        {
            "candidate": "A" if index == 0 else "B",
            "agent": c.get("agent"),
            "strategy": c.get("strategy"),
            "risk": c.get("risk"),
            "rationale": c.get("rationale"),
            "ops": [
                {"path": op["path"], "search": op["search"][:160], "replace": op["replace"][:160]}
                for op in c.get("ops", [])
            ],
        }
        for index, c in enumerate(candidates[:2])
    ]
    return AgentTask(spec, state, {"candidates": summary})
