# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""State bus definition for the self-iterating development loop.

This is the contract every phase node reads and writes.  Channels behave like
LangGraph channels: each one either *replaces* its previous value or *reduces*
the incoming update into an accumulated ledger.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping

from .core import merge_dict, merge_ledger, Replace

__all__ = ["CHANNELS", "REDUCER_KINDS", "NODE_NAMES", "initial_state", "apply_channel"]

#: channel -> reducer.  ``None`` means "whole-value replace".
CHANNELS: Dict[str, Any] = {
    # --- what the loop knows about the world -------------------------------
    "codebase": None,          # Codebase dict (frozen snapshot at run start)
    "codebase_files": None,    # dict[path -> content]: the frozen baseline tree
    "target_root": None,       # str: repository under iteration
    "change_log": None,        # list[str]: applied file changes (append via node)
    "cwd": None,               # str: sandbox working directory
    # --- Phase 1: issue discovery -----------------------------------------
    "issue_candidates": merge_ledger("issues"),
    "batch_issue_ids": None,       # list[str]: every issue this discovery pass found
    "target_issue_id": None,       # str: the single issue this cycle works on
    "batch_verified": None,        # list[str]: issues verified green this batch
    # --- the shared ledger every phase reads ------------------------------
    "issues": merge_ledger("issues"),
    # --- Phase 2: requirement refinement ----------------------------------
    "findings": merge_ledger("findings"),
    "test_plans": merge_ledger("plans"),
    "refine_votes": merge_ledger("checks"),
    "focus_ids": None,             # list[str]: issues a scoped phase pass targets
    # --- Phase 3: code review ---------------------------------------------
    "reviews": merge_ledger("reviews"),
    "review_decision": None,       # approve | refine | stop
    "review_reason": None,
    "review_blocking": None,
    # --- Phase 4: patch synthesis -----------------------------------------
    "patches": merge_ledger("patches"),
    "changes": merge_ledger("changes"),
    "working_code": None,      # dict[path -> content]: candidate tree
    "selected_patch": None,    # the patch the cycle decided to apply
    "patch_error": None,
    # --- Phase 5: QA verification -----------------------------------------
    "checks": merge_ledger("checks"),
    "evidence": merge_dict,    # dict: measured facts (test runs, static checks)
    "baseline_tests": None,    # dict: the suite as measured before any change
    "verifications": merge_ledger("checks"),
    "qa_votes": merge_ledger("checks"),
    # --- convergence bookkeeping -------------------------------------------
    "cycles": merge_ledger("cycles"),
    "quality_history": None,   # list[dict]: one row per QA pass
    # --- control ------------------------------------------------------------
    "cycle": None,                  # int
    "max_iterations": None,         # int: phase-5 loop budget
    "self_loop_count": None,        # int: review -> refine back edges taken
    "no_progress_cycles": None,     # int: consecutive cycles that closed nothing
    "node_visits": None,            # dict[node -> int]
    "status": None,                 # running | converged | human_review | escalated
    "status_reason": None,          # str
    "stop_requested": None,         # bool
    "needs_discovery": None,        # bool: next cycle discovers a fresh batch
    # --- runtime settings (set once, read by the phases) --------------------
    "run_tests": None,              # bool: may QA execute the target's suite
    "test_timeout": None,           # float seconds per pytest invocation
    "keep_workspaces": None,        # bool: keep the scratch trees for inspection
    "allow_multi_cycle": None,      # bool: may the loop run more than one cycle
    "model_info": None,             # dict: channel / provider / model / base_url
    "brief": None,                  # str: per-target project constraints for every agent
    "brief_source": None,           # str: file the constraints were read from
}

#: Ledger channels, listed for reporting and for delta-vs-value validation.
REDUCER_KINDS: Dict[str, str] = {
    "issue_candidates": "issues",
    "issues": "issues",
    "findings": "findings",
    "test_plans": "plans",
    "refine_votes": "checks",
    "reviews": "reviews",
    "patches": "patches",
    "changes": "changes",
    "checks": "checks",
    "verifications": "checks",
    "qa_votes": "checks",
    "cycles": "cycles",
}

NODE_NAMES = ("discover", "refine", "review", "patch", "qa")


def initial_state(
    codebase: Mapping[str, Any],
    target_root: str,
    *,
    max_iterations: int = 3,
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build the run state.  Ledger channels start empty so reducers work."""
    files = {f["path"]: f["content"] for f in codebase.get("files", [])}
    state: Dict[str, Any] = {
        "codebase": dict(codebase),
        "codebase_files": dict(files),
        "target_root": target_root,
        "cwd": target_root,
        "change_log": [],
        "issue_candidates": [],
        "batch_issue_ids": [],
        "target_issue_id": "",
        "batch_verified": [],
        "issues": [],
        "findings": [],
        "test_plans": [],
        "refine_votes": [],
        "focus_ids": [],
        "reviews": [],
        "review_decision": "",
        "review_reason": "",
        "review_blocking": [],
        "patches": [],
        "changes": [],
        "working_code": dict(files),
        "selected_patch": {},
        "patch_error": "",
        "checks": [],
        "evidence": {},
        "baseline_tests": {},
        "verifications": [],
        "qa_votes": [],
        "cycles": [],
        "quality_history": [],
        "cycle": 0,
        "max_iterations": int(max_iterations),
        "self_loop_count": 0,
        "no_progress_cycles": 0,
        "node_visits": {},
        "status": "running",
        "status_reason": "",
        "stop_requested": False,
        "needs_discovery": True,
        "run_tests": True,
        "test_timeout": 120.0,
        "keep_workspaces": False,
        "allow_multi_cycle": True,
        "model_info": {},
        "brief": "",
        "brief_source": "",
    }
    if extra:
        state.update(dict(extra))
    return state


def apply_channel(state: MutableMapping[str, Any], name: str, reducer: Any, value: Any) -> Any:
    """Apply one channel update and write it back into ``state``."""
    if isinstance(value, Replace):
        state[name] = value.value
        return value.value
    if reducer is None:
        state[name] = value
        return value
    current = state.get(name)
    if current is None:
        # A ledger reducer always accumulates into a list; every other reducer
        # (merge_dict) treats None as "nothing yet".
        current = [] if getattr(reducer, "ledger_kind", None) else {}
    merged = reducer(current, value)
    state[name] = merged
    return merged
