# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""The five phases of the self-iterating loop, each a graph node factory."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from ..crew import Crew
from ..graph import END, START, CompiledGraph, StateGraph
from ..policy import Policy
from .base import ListenerBox, listener_box
from .discover import make_discover_node
from .patch import make_patch_node
from .qa import make_qa_node
from .refine import make_refine_node
from .review import make_review_node

__all__ = [
    "make_discover_node",
    "make_refine_node",
    "make_review_node",
    "make_patch_node",
    "make_qa_node",
    "build_loop_graph",
    "LOOP_EDGES",
    "listener_box",
]

#: (source, target, condition) -- printed in reports and rendered as a diagram.
LOOP_EDGES: List[Dict[str, str]] = [
    {"from": "START", "to": "discover", "when": "always"},
    {"from": "discover", "to": "refine", "when": "new issues found"},
    {"from": "discover", "to": "END", "when": "no new issue: the loop is at a fixed point"},
    {"from": "refine", "to": "review", "when": "requirement stated with acceptance criteria"},
    {"from": "review", "to": "refine", "when": "a reviewer rejects the plan (max_self_loops)"},
    {"from": "review", "to": "patch", "when": "approvals >= min_approvals"},
    {"from": "review", "to": "END", "when": "rejections persist past the refine budget"},
    {"from": "patch", "to": "qa", "when": "anchored ops applied to the candidate tree"},
    {"from": "qa", "to": "discover", "when": "quality < target and budget remains"},
    {"from": "qa", "to": "END", "when": "converged / escalated / human review"},
]


def build_loop_graph(
    crew: Crew,
    policy: Policy,
    *,
    box: ListenerBox | None = None,
    max_steps: int = 64,
) -> CompiledGraph:
    """Assemble the StateGraph exactly as drawn in docs/architecture.md.

    Agent events go through the shared ``box`` rather than a fixed listener, so
    a caller (the orchestrator) can attach a trace *after* building the graph.
    """
    listeners = box if box is not None else listener_box()
    graph = StateGraph("self_iterating_dev_loop")
    graph.add_node("discover", make_discover_node(crew, policy, listeners), max_visits=12)
    graph.add_node("refine", make_refine_node(crew, policy, listeners), max_visits=8)
    graph.add_node("review", make_review_node(crew, policy, listeners), max_visits=8)
    graph.add_node("patch", make_patch_node(crew, policy, listeners), max_visits=8)
    graph.add_node("qa", make_qa_node(crew, policy, listeners), max_visits=8)

    graph.add_edge(START, "discover")
    graph.add_conditional_edges(
        "discover",
        lambda state: "end" if _discovery_done(state) else "continue",
        {"continue": "refine", "end": END},
    )
    graph.add_edge("refine", "review")
    graph.add_conditional_edges(
        "review",
        lambda state: {"approve": "approve", "refine": "refine", "stop": "stop"}.get(
            str(state.get("review_decision") or "refine"), "refine"
        ),
        {"approve": "patch", "refine": "refine", "stop": END},
    )
    graph.add_edge("patch", "qa")
    graph.add_conditional_edges(
        "qa",
        lambda state: "loop" if str(state.get("status")) == "continue" else "done",
        {"loop": "discover", "done": END},
    )
    return graph.compile(max_steps=max_steps, default_max_visits=8)


def _discovery_done(state: Dict[str, Any]) -> bool:
    """END when the cycle has nothing to work on.

    A discovery pass that only re-sighted already-verified issues leaves the
    loop at a fixed point: there is no open target, so refining would be busy
    work.  The loop exits and reports instead.
    """
    if str(state.get("status")) not in ("running", "continue"):
        return True
    return not str(state.get("target_issue_id") or "")
