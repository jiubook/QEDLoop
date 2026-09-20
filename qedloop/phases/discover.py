# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Phase 1 -- issue discovery.

Three agents read the same tree through three different lenses and return
independent batches.  This node then does the work that makes the rest of the
loop cheap:

* **de-duplication** -- lenses routinely describe one defect three times; a
  sighting is merged onto the canonical row instead of creating a near-copy.
* **target selection** -- the cycle is scoped to exactly one issue, so every
  later verdict and test result is attributable to it.
* **fixed-point detection** -- if nothing new was found and nothing is left
  open, the loop really is done, and saying so is more useful than another
  cycle over the same tree.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..core import as_float, new_id, recap
from ..crew import Crew
from ..prompts import specs_for
from .base import ListenerBox, agent_rows, listener_box, make_logger, verified_issues

MAX_CANDIDATES_PER_AGENT = 3
SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def make_discover_node(crew: Crew, policy: Any = None, box: ListenerBox | None = None):
    log = make_logger(box if box is not None else listener_box())

    def discover(state: Mapping[str, Any]) -> Dict[str, Any]:
        cycle = int(state.get("cycle", 0))
        if not state.get("needs_discovery", True):
            # The current target has not reached a verdict yet; re-scanning now
            # would only re-report the issue being worked on.
            log("phase1.skip", {"phase": "discover", "cycle": cycle, "reason": "current target not finished/verified"})
            return {
                "checks": [
                    {
                        "id": new_id("check"),
                        "phase": "discover",
                        "role": "gate",
                        "ok": True,
                        "verdict": "skip",
                        "cycle": cycle,
                        "note": "discovery deferred until the current target is verified",
                    }
                ]
            }

        done = verified_issues(state)
        log(
            "phase1.start",
            {
                "phase": "discover",
                "cycle": cycle,
                "agents": [s.role for s in specs_for("discover")],
                "already_verified": len(done),
            },
        )

        results = crew.run_phase("discover", state)
        raw = crew.issues(results, cycle=cycle)
        reported, fresh = _dedupe(raw, state.get("issues") or [])
        candidates = _cap(reported)
        target = _pick_target(candidates, state)
        checks = agent_rows(results, phase="discover", cycle=cycle, listener=log)

        answered = [r for r in results if getattr(r, "ok", False)]
        unreachable = [str(getattr(r, "error", "") or "no output") for r in results if not getattr(r, "ok", False)]

        if target is None and not answered:
            # Nobody answered, so "nothing was found" is not a finding: it is a
            # run that measured nothing.  Reporting it as a fixed point would
            # turn a wrong key, a retired model id or a network outage into
            # "your repository is clean" -- the one lie this loop must never
            # tell, because the user cannot see it from the outside.
            reason = "%d/%d discovery agents failed, so nothing was scanned: %s" % (
                len(unreachable), len(results), (unreachable or ["no agent was scheduled"])[0][:200],
            )
            log("phase1.unreachable", {"phase": "discover", "cycle": cycle, "reason": reason,
                                       "failed": len(results)})
            return {
                "needs_discovery": False,
                "status": "error",
                "status_reason": reason,
                "checks": checks
                + [
                    {
                        "id": new_id("check"),
                        "phase": "discover",
                        "role": "gate",
                        "ok": False,
                        "verdict": "error",
                        "cycle": cycle,
                        "note": reason,
                    }
                ],
            }

        if target is None:
            # Fixed point: nothing to work on.  Reached either because the tree
            # is genuinely clean or because everything reported so far has been
            # verified by measurement.
            note = (
                "no issue was reported and no previously reported issue is still open"
                if not candidates
                else "every reported issue has been verified; nothing left to work on"
            )
            if unreachable:
                # A partial crew is a weaker fixed point, not a silent one.
                note += " (%d/%d lenses failed to answer)" % (len(unreachable), len(results))
            log("phase1.settled", {"phase": "discover", "cycle": cycle, "note": note, "reported": len(candidates)})
            return {
                "issue_candidates": fresh,
                "issues": fresh,
                "batch_issue_ids": [str(c["id"]) for c in candidates],
                "target_issue_id": "",
                "focus_ids": [],
                "needs_discovery": False,
                "status": "converged",
                "status_reason": note,
                "checks": checks
                + [
                    {
                        "id": new_id("check"),
                        "phase": "discover",
                        "role": "gate",
                        "ok": not unreachable,
                        "verdict": "converged" if not unreachable else "degraded",
                        "cycle": cycle,
                        "note": note,
                    }
                ],
            }

        log(
            "phase1.found",
            {
                "phase": "discover",
                "cycle": cycle,
                "raw_reports": len(raw),
                "deduped": len(candidates),
                "new_rows": len(fresh),
                "target": target["id"],
                "issues": [
                    {"id": c["id"], "title": c["title"], "severity": c["severity"], "lenses": c.get("lenses")}
                    for c in candidates
                ],
            },
        )
        return {
            "issue_candidates": fresh,
            "issues": candidates,          # reducer merges/dedupes into the ledger
            "batch_issue_ids": [str(c["id"]) for c in candidates],
            "target_issue_id": str(target["id"]),
            "focus_ids": [str(target["id"])],
            "batch_verified": [],
            "needs_discovery": False,
            "checks": checks,
        }

    return discover


# --------------------------------------------------------------------------- #
# ordering, identity, de-duplication
# --------------------------------------------------------------------------- #


def _cap(candidates: Sequence[Mapping[str, Any]], limit: int = MAX_CANDIDATES_PER_AGENT * 3) -> List[Dict[str, Any]]:
    """Order the batch: severity first, then reporter confidence, then stable id."""
    ordered = sorted(
        (dict(c) for c in candidates),
        key=lambda c: (
            SEVERITY_RANK.get(str(c.get("severity", "medium")).lower(), 2),
            -as_float(c.get("confidence"), 0.0),
            str(c.get("id")),
        ),
    )
    return ordered[:limit]


def issue_signature(issue: Mapping[str, Any]) -> str:
    """Identity of a *defect*, not of a report about it.

    A declared marker is the strongest identity the code offers.  Without one,
    the normalised title plus the first file is the best available proxy.
    """
    marker = str(issue.get("bug_marker") or "").strip().lower()
    if marker:
        return "marker::%s" % marker
    files = [str(f).replace("\\", "/").lower() for f in (issue.get("files") or [])]
    title = " ".join(str(issue.get("title") or "").lower().split())
    return "title::%s::%s" % (title, files[0] if files else "")


def _dedupe(
    candidates: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collapse reports of one defect into its canonical row.

    Returns ``(reported, fresh)``:

    * ``reported`` -- one canonical row per distinct defect seen this pass.  A
      re-sighting of a known issue comes back carrying the *known* assignment,
      so the loop keeps working on the row it already owns.
    * ``fresh`` -- only the rows that are not in the ledger yet, i.e. what the
      reducer should be handed as new.

    Independent agreement is treated as evidence: each extra lens raises the
    row's confidence and extends its ``lenses`` list, and the strongest severity
    reported by any lens wins.
    """
    known: Dict[str, Dict[str, Any]] = {}
    for row in ledger:
        known.setdefault(issue_signature(row), dict(row))

    seen: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    fresh: List[Dict[str, Any]] = []

    for candidate in candidates:
        signature = issue_signature(candidate)
        if signature in seen:
            _merge_sighting(seen[signature], candidate)
            continue
        if signature in known:
            row = dict(known[signature])
            row.setdefault("lenses", [row.get("lens", "")])
            _merge_sighting(row, candidate)
            seen[signature] = row
            order.append(signature)
            continue
        row = dict(candidate)
        row.setdefault("lenses", [row.get("lens", "")])
        row.setdefault("seen_count", 1)
        seen[signature] = row
        order.append(signature)
        fresh.append(row)

    return [seen[s] for s in order], fresh


def _merge_sighting(row: Dict[str, Any], candidate: Mapping[str, Any]) -> None:
    lenses = list(row.get("lenses") or ([row["lens"]] if row.get("lens") else []))
    if candidate.get("lens") and candidate["lens"] not in lenses:
        lenses.append(candidate["lens"])
    row["lenses"] = lenses
    if lenses and not row.get("lens"):
        row["lens"] = lenses[0]
    row["confidence"] = round(min(0.99, as_float(row.get("confidence"), 0.5) + 0.05), 3)
    if SEVERITY_RANK.get(str(candidate.get("severity")), 9) < SEVERITY_RANK.get(str(row.get("severity")), 9):
        row["severity"] = candidate["severity"]


def _pick_target(candidates: Sequence[Mapping[str, Any]], state: Mapping[str, Any]) -> Dict[str, Any] | None:
    """The highest-priority open issue that has not already been verified.

    Severity first, then confidence, then the stable id -- so replaying the same
    tree targets the same issue and a run stays reproducible.
    """
    verified = {str(i.get("id")) for i in (state.get("issues") or []) if str(i.get("status")) == "verified"}
    eligible = [dict(c) for c in candidates if str(c.get("id")) not in verified]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda c: (
            SEVERITY_RANK.get(str(c.get("severity", "medium")).lower(), 2),
            -as_float(c.get("confidence"), 0.0),
            str(c.get("id")),
        ),
    )[0]


def summarize_batch(state: Mapping[str, Any], limit: int = 10) -> str:
    return recap(
        list(state.get("issues") or [])[:limit],
        ("id", "severity", "title", "status", "seen_count"),
        limit=limit,
    )
