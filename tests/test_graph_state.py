# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Reducer tests: the state bus is the contract every phase relies on."""

from __future__ import annotations

import pytest

from qedloop.core import Replace, merge_dict, merge_ledger, new_id
from qedloop.graph import (
    END,
    START,
    GraphError,
    StateGraph,
    StepLimitExceeded,
    apply_updates,
    summarise_delta,
)
from qedloop.state import CHANNELS, initial_state


def _codebase():
    return {
        "root": "C:/demo",
        "revision": "abc123",
        "manifest": {"src/a.py": "111"},
        "files": [
            {"path": "src/a.py", "content": "x = 1\n", "lines": 2, "sha": "111", "is_test": False},
            {"path": "tests/test_a.py", "content": "def test_x():\n    pass\n", "lines": 2, "sha": "222", "is_test": True},
        ],
    }


# --------------------------------------------------------------------------- #
# reducers
# --------------------------------------------------------------------------- #


def test_issue_reducer_merges_by_id_instead_of_duplicating():
    reducer = merge_ledger("issues")
    first = reducer([], [{"id": "ISS-0001", "title": "a", "status": "open", "cycle": 0}])
    assert len(first) == 1 and first[0]["seen_count"] == 1

    second = reducer(first, [{"id": "ISS-0001", "title": "a", "status": "verified", "cycle": 1}])
    assert len(second) == 1, "the same issue must not enter the ledger twice"
    assert second[0]["status"] == "verified"
    assert second[0]["seen_count"] == 2


def test_issue_reducer_keeps_distinct_ids_separate():
    reducer = merge_ledger("issues")
    merged = reducer([], [{"id": "ISS-0001", "title": "a"}, {"id": "ISS-0002", "title": "b"}])
    assert [row["id"] for row in merged] == ["ISS-0001", "ISS-0002"]


def test_finding_reducer_supersedes_and_versions():
    reducer = merge_ledger("findings")
    first = reducer([], [{"id": "FND-0001", "issue_id": "ISS-0001", "summary": "v1", "version": 1}])
    second = reducer(first, [{"id": "FND-0001", "issue_id": "ISS-0001", "summary": "v2"}])
    assert len(second) == 1
    assert second[0]["version"] == 2
    assert second[0]["summary"] == "v2"
    assert second[0]["superseded"][0]["summary"] == "v1"


def test_a_new_plan_generation_retires_the_previous_one():
    """Plans are replaced per round; earlier generations must stop being served.

    Measured on a real run: four refine rounds left five plan rows for one
    issue, three of them naming the same test function, and the reviewers
    rejected every round for ambiguity that only existed in the ledger.  The
    retired rows stay for audit -- deleting them would make a replayed run
    diverge from the one that produced the report.
    """
    reducer = merge_ledger("plans")
    first = reducer([], [
        {"id": "PLN-0001", "issue_id": "ISS-0001", "name": "test_a", "refined_at": "T1"},
        {"id": "PLN-0002", "issue_id": "ISS-0001", "name": "test_b", "refined_at": "T1"},
    ])
    second = reducer(first, [{"id": "PLN-0003", "issue_id": "ISS-0001", "name": "test_a", "refined_at": "T2"}])

    assert len(second) == 3, "the audit trail keeps every generation"
    live = [p for p in second if not p.get("superseded_at")]
    assert [p["id"] for p in live] == ["PLN-0003"], "only the newest generation is in force"
    assert {p["superseded_at"] for p in second if p.get("superseded_at")} == {"T2"}


def test_retiring_plans_is_scoped_to_the_issue():
    """One issue's refinement must not retire another issue's plan set."""
    reducer = merge_ledger("plans")
    rows = reducer([], [{"id": "PLN-0001", "issue_id": "ISS-0001", "refined_at": "T1"}])
    rows = reducer(rows, [{"id": "PLN-0002", "issue_id": "ISS-0002", "refined_at": "T2"}])
    assert not rows[0].get("superseded_at"), "ISS-0001's plan is still in force"


def test_plans_without_a_generation_stamp_are_never_retired():
    """A plan that cannot say which round it belongs to is left alone."""
    reducer = merge_ledger("plans")
    rows = reducer([], [{"id": "PLN-0001", "issue_id": "ISS-0001"}])
    rows = reducer(rows, [{"id": "PLN-0002", "issue_id": "ISS-0001", "refined_at": "T2"}])
    assert not rows[0].get("superseded_at")


def test_append_only_ledgers_accumulate_duplicates():
    reducer = merge_ledger("checks")
    rows = reducer(reducer([], [{"id": "CHK-0001"}]), [{"id": "CHK-0001"}])
    assert len(rows) == 2, "checks are an append-only audit trail, not a set"


def test_merge_dict_is_a_shallow_union():
    assert merge_dict({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}


def test_ids_are_unique_and_prefixed():
    values = {new_id("issue") for _ in range(50)}
    assert len(values) == 50
    assert all(v.startswith("ISS-") for v in values)


# --------------------------------------------------------------------------- #
# updates
# --------------------------------------------------------------------------- #


def test_apply_updates_uses_the_channel_reducer():
    state = initial_state(_codebase(), "C:/demo")
    apply_updates(state, {"issues": [{"id": "ISS-0001", "title": "a"}]})
    apply_updates(state, {"issues": [{"id": "ISS-0001", "title": "a"}, {"id": "ISS-0002", "title": "b"}]})
    assert len(state["issues"]) == 2
    assert state["issues"][0]["seen_count"] == 2


def test_apply_updates_honours_replace_marker():
    state = initial_state(_codebase(), "C:/demo")
    apply_updates(state, {"working_code": Replace({"src/a.py": "x = 2\n"})})
    assert state["working_code"] == {"src/a.py": "x = 2\n"}


def test_apply_updates_skips_none_values():
    state = initial_state(_codebase(), "C:/demo")
    touched = apply_updates(state, {"evidence": None, "status": None})
    assert touched == []


def test_every_channel_is_declared():
    state = initial_state(_codebase(), "C:/demo")
    assert set(state) <= set(CHANNELS), "nodes may only write declared channels"


def test_a_failing_node_does_not_erase_the_run():
    """A node failure must leave the evidence of everything that ran before it.

    ``run`` copies the state, so an exception anywhere used to discard the whole
    run: the report for a run that had found two issues and gathered three
    approvals read "0 issues -- no issue was reported in this run", which is
    indistinguishable from a clean repository.  The tokens were already spent;
    the findings must survive.
    """
    def found(state):
        return {"issues": [{"id": "ISS-0001", "title": "a real finding"}]}

    def dies(state):
        raise ValueError("could not convert string to float: 'low'")

    graph = StateGraph("partial")
    graph.add_node("found", found)
    graph.add_node("dies", dies)
    graph.add_edge(START, "found")
    graph.add_edge("found", "dies")
    graph.add_edge("dies", END)

    state = initial_state(_codebase(), "C:/demo")
    with pytest.raises(ValueError):
        graph.compile().run(state)

    assert len(state["issues"]) == 1, "the work done before the failure must survive"
    assert state["node_visits"] == {"found": 1, "dies": 1}


def test_evidence_channel_merges():
    state = initial_state(_codebase(), "C:/demo")
    apply_updates(state, {"evidence": {"a": 1}})
    apply_updates(state, {"evidence": {"b": 2}})
    assert state["evidence"] == {"a": 1, "b": 2}


# --------------------------------------------------------------------------- #
# graph engine
# --------------------------------------------------------------------------- #


def test_graph_runs_nodes_in_order_and_ends():
    seen = []
    graph = StateGraph("t", channels={"trace": None, "n": None})
    graph.add_node("a", lambda s: {"n": 1})
    graph.add_node("b", lambda s: {"n": s["n"] + 1})
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)
    compiled = graph.compile()
    final = compiled.run({"trace": [], "n": 0}, listener=lambda e: seen.append(e["node"]))
    assert final["n"] == 2
    # Two events per node: "start", then "done" once the node body has returned.
    assert seen == ["a", "a", "b", "b"]


def test_the_listener_hears_a_node_start_and_its_result():
    """One event before the node runs, one after -- not two of the first.

    The live view is built from these, and a node is slow (an agent-backed one
    takes minutes), so "started" has to be visible before it finishes or a
    working run looks hung.  It used to be visible *only* before, which is how
    every line came to read ``ok 0ms -> `` no matter what the node did.
    """
    seen = []
    graph = StateGraph("t", channels={"n": None, "notes": None})
    graph.add_node("a", lambda s: {"n": 1, "notes": ["one", "two"]})
    graph.add_edge(START, "a")
    graph.add_edge("a", END)

    graph.compile().run({"n": 0}, listener=seen.append)

    assert [e["event"] for e in seen] == ["start", "done"]
    start, done = seen
    assert start["node"] == "a" and "duration_ms" not in start
    assert "started_at" not in start, "a wall-clock float is not trace material"
    assert done["ok"] is True and done["next"] == END
    assert isinstance(done["duration_ms"], float)
    assert done["delta"] == {"n": "1", "notes": "2 items"}, done["delta"]


def test_a_node_that_raises_is_reported_as_failed_and_not_as_never_run():
    """A node blowing up must reach the trace: an exception used to be yielded
    to a caller that threw the yield away, so the last thing a reader saw was
    the node *starting*."""
    seen = []
    graph = StateGraph("t", channels={"n": None})

    def boom(_state):
        raise ValueError("nope")

    graph.add_node("a", boom)
    graph.add_edge(START, "a")
    graph.add_edge("a", END)

    with pytest.raises(ValueError):
        graph.compile().run({"n": 0}, listener=seen.append)

    assert [e["event"] for e in seen] == ["start", "done"]
    assert seen[-1]["ok"] is False and "ValueError: nope" in seen[-1]["error"]


def test_a_large_channel_is_summarised_instead_of_copied_into_the_trace():
    """The ``qa`` node's evidence carries the candidate suite's stdout: logging
    it verbatim would put megabytes into a file meant to be read while the run is
    in flight, and ``trace.jsonl`` is what the live view reads."""
    summary = summarise_delta({
        "evidence": {"tests": {"stdout": "x" * 500_000}},
        "checks": [{"id": "c1"}, {"id": "c2"}],
        "status": "continue",
        "target_issue_id": "",
        "status_reason": "y" * 300,
    })
    assert summary == {
        "evidence": "1 key",
        "checks": "2 items",
        "status": "continue",
        "target_issue_id": "empty",
        "status_reason": "300 chars",
    }, summary


def test_conditional_edges_route_by_state():
    def router(state):
        return "up" if state["n"] < 2 else "down"

    graph = StateGraph("t", channels={"n": None})
    graph.add_node("loop", lambda s: {"n": s["n"] + 1})
    graph.add_node("exit", lambda s: {})
    graph.add_edge(START, "loop")
    graph.add_conditional_edges("loop", router, {"up": "loop", "down": "exit"})
    graph.add_edge("exit", END)
    assert graph.compile().run({"n": 0})["n"] == 2


def test_self_loop_is_bounded_by_max_visits():
    graph = StateGraph("t", channels={"n": None})
    graph.add_node("spin", lambda s: {"n": s["n"] + 1}, max_visits=3)
    graph.add_edge(START, "spin")
    graph.add_edge("spin", "spin")
    with pytest.raises(StepLimitExceeded):
        graph.compile().run({"n": 0})


def test_step_budget_is_enforced():
    graph = StateGraph("t", channels={"n": None})
    graph.add_node("spin", lambda s: {"n": s["n"] + 1}, max_visits=1000)
    graph.add_edge(START, "spin")
    graph.add_edge("spin", "spin")
    with pytest.raises(StepLimitExceeded):
        graph.compile(max_steps=5).run({"n": 0})


def test_unknown_edge_target_is_rejected_at_compile():
    graph = StateGraph("t", channels={"n": None})
    graph.add_node("a", lambda s: {})
    graph.add_edge(START, "a")
    graph.add_edge("a", "ghost")
    with pytest.raises(GraphError):
        graph.compile()


def test_router_returning_unknown_key_is_rejected():
    graph = StateGraph("t", channels={"n": None})
    graph.add_node("a", lambda s: {})
    graph.add_edge(START, "a")
    graph.add_conditional_edges("a", lambda s: "nope", {"ok": END})
    with pytest.raises(GraphError):
        graph.compile().run({"n": 0})


def test_node_visits_are_counted_in_state():
    graph = StateGraph("t", channels={"n": None, "node_visits": None})
    graph.add_node("a", lambda s: {"n": 1})
    graph.add_edge(START, "a")
    graph.add_edge("a", END)
    final = graph.compile().run({"n": 0, "node_visits": {}})
    assert final["node_visits"] == {"a": 1}


def test_mermaid_export_mentions_every_node():
    graph = StateGraph("t", channels={"n": None})
    for name in ("a", "b"):
        graph.add_node(name, lambda s: {})
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)
    diagram = graph.compile().mermaid()
    assert "flowchart TD" in diagram
    assert 'a["a"]' in diagram
    assert "DONE([END])" in diagram
