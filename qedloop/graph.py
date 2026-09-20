# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""A small, dependency-free StateGraph engine.

Mirrors the LangGraph model closely enough that the mental transfer is 1:1:

======================  =========================================================
LangGraph               qedloop
======================  =========================================================
``StateGraph(State)``   :class:`StateGraph` + ``CHANNELS`` in :mod:`qedloop.state`
annotated reducers      ledger reducers in :mod:`qedloop.core`
``add_node``            :meth:`StateGraph.add_node`
``add_edge``            :meth:`StateGraph.add_edge`
``add_conditional_edges``  :meth:`StateGraph.add_conditional_edges`
``START`` / ``END``     :data:`START` / :data:`END`
``compile().invoke``    :meth:`StateGraph.compile` -> :meth:`CompiledGraph.run`
``.stream()``           :meth:`CompiledGraph.stream`
recursion_limit         ``max_steps`` (unbounded edges) + per-node ``max_visits``
checkpointer            not implemented (see docs/architecture.md)
======================  =========================================================
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from . import state as state_mod
from .core import Replace

START = "__start__"
END = "__end__"

NodeFn = Callable[[Dict[str, Any]], Optional[Mapping[str, Any]]]
EdgeFn = Callable[[Dict[str, Any]], str]
Listener = Callable[[Dict[str, Any]], None]


class GraphError(RuntimeError):
    """Raised for malformed graphs (unknown edge target, dead end, ...)."""


class StepLimitExceeded(RuntimeError):
    """Raised when the loop guard trips: too many steps or node visits."""


# --------------------------------------------------------------------------- #
# channel application (shared with tests and the orchestrator)
# --------------------------------------------------------------------------- #


def apply_updates(state: Dict[str, Any], updates: Optional[Mapping[str, Any]]) -> List[str]:
    """Apply a node's returned delta to ``state``; return the touched channels."""
    if not updates:
        return []
    touched: List[str] = []
    for name, value in updates.items():
        reducer = state_mod.CHANNELS.get(name)
        if value is None and not isinstance(value, Replace):
            continue
        state_mod.apply_channel(state, name, reducer, value)
        touched.append(name)
    return touched


def unwrap(updates: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Strip ``Replace`` markers -- handy when persisting the final state."""
    out: Dict[str, Any] = {}
    for key, value in (updates or {}).items():
        out[key] = value.value if isinstance(value, Replace) else value
    return out


def _brief_text(text: str, limit: int = 40) -> str:
    """A string a reader can use: the value if it is short, its size if it is not.

    Verbatim-if-short matters because the interesting strings on the bus are tiny
    and self-describing (``status: continue``, ``target_issue_id: ISS-0006``),
    while the uninteresting ones are enormous and opaque (``status_reason``,
    ``working_code``).
    """
    flat = " ".join(text.split())
    if not flat:
        return "empty"
    return flat if len(flat) <= limit else "%d chars" % len(text)


def summarise_delta(delta: Any) -> Dict[str, str]:
    """Per-channel sizes of a node's delta, for anything that has to log it.

    The raw delta cannot be logged: the ``qa`` node's ``evidence`` carries the
    candidate suite's stdout, so one node would add hundreds of kilobytes to a
    file whose whole purpose is to be read *while the run is in flight*.  What a
    reader needs from a node event is which channels it wrote and how much, which
    is exactly what a size gives; the values themselves are already in the report
    and in ``state.json``.
    """
    out: Dict[str, str] = {}
    for name, value in (delta or {}).items():
        if isinstance(value, (list, tuple, set)):
            out[str(name)] = "%d item%s" % (len(value), "" if len(value) == 1 else "s")
        elif isinstance(value, Mapping):
            out[str(name)] = "%d key%s" % (len(value), "" if len(value) == 1 else "s")
        elif isinstance(value, str):
            out[str(name)] = _brief_text(value)
        else:
            out[str(name)] = str(value)
    return out


def for_listener(event: Mapping[str, Any]) -> Dict[str, Any]:
    """The copy a ``listener`` receives: no internal clock, no raw delta.

    A listener is where the live view and ``trace.jsonl`` come from, so what it
    gets has to be JSON-serialisable and small enough to keep for a whole run.
    ``started_at`` is a wall-clock float the trace already covers with ``ts`` and
    ``elapsed_s``; ``delta`` is summarised by :func:`summarise_delta`.
    """
    payload = {key: value for key, value in event.items() if key != "started_at"}
    if "delta" in payload:
        payload["delta"] = summarise_delta(payload["delta"])
    return payload


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #


class StateGraph:
    """Declare nodes and edges first, then :meth:`compile` into a runnable."""

    def __init__(self, name: str = "graph", channels: Optional[Mapping[str, Any]] = None) -> None:
        self.name = name
        self.channels: Dict[str, Any] = dict(channels or state_mod.CHANNELS)
        self.nodes: Dict[str, NodeFn] = {}
        self.edges: Dict[str, str] = {}
        self.branches: Dict[str, Tuple[EdgeFn, Dict[str, str]]] = {}
        self.max_visits: Dict[str, int] = {}

    # -- declaration ------------------------------------------------------- #
    def add_node(
        self,
        name: str,
        fn: NodeFn,
        *,
        max_visits: Optional[int] = None,
        replace: bool = False,
    ) -> "StateGraph":
        if name in (START, END):
            raise GraphError("%r is reserved" % name)
        if name in self.nodes and not replace:
            raise GraphError("node %r already exists" % name)
        if not callable(fn):
            raise GraphError("node %r must be callable" % name)
        self.nodes[name] = fn
        if max_visits is not None:
            self.max_visits[name] = int(max_visits)
        return self

    def add_edge(self, source: str, target: str) -> "StateGraph":
        if source in self.branches:
            raise GraphError("node %r already has a conditional branch" % source)
        self.edges[source] = target
        return self

    def add_conditional_edges(
        self,
        source: str,
        router: EdgeFn,
        targets: Sequence[str] | Mapping[str, str],
    ) -> "StateGraph":
        if source in self.edges:
            raise GraphError("node %r already has a static edge" % source)
        if isinstance(targets, Mapping):
            table = {str(k): str(v) for k, v in targets.items()}
        else:
            table = {str(t): str(t) for t in targets}
        self.branches[source] = (router, table)
        return self

    # -- validation -------------------------------------------------------- #
    def _validate(self) -> None:
        if not self.nodes:
            raise GraphError("graph has no nodes")
        for source, target in self.edges.items():
            for name in (source, target):
                if name not in self.nodes and name != END and name != START:
                    raise GraphError("edge %s -> %s references unknown node %r" % (source, target, name))
        for source, (_, table) in self.branches.items():
            if source not in self.nodes:
                raise GraphError("branch source %r is not a node" % source)
            if not table:
                raise GraphError("branch %r has no targets" % source)
            for key, target in table.items():
                if target not in self.nodes and target != END:
                    raise GraphError("branch %s -[%s]-> unknown node %r" % (source, key, target))
        if not self.outgoing(START):
            raise GraphError("START has no outgoing edge")

    def outgoing(self, node: str) -> List[str]:
        if node in self.branches:
            return sorted(set(self.branches[node][1].values()))
        if node in self.edges:
            return [self.edges[node]]
        return []

    def reachable(self) -> List[str]:
        seen: List[str] = []
        stack: List[str] = list(self.outgoing(START))
        while stack:
            node = stack.pop()
            if node in seen or node == END:
                continue
            seen.append(node)
            stack.extend(self.outgoing(node))
        return seen

    def describe(self) -> str:
        lines = ["graph %r -- %d nodes, %d channels" % (self.name, len(self.nodes), len(self.channels))]
        for name in self.nodes:
            if name in self.branches:
                router, table = self.branches[name]
                arrow = "?[%s] -> %s" % (getattr(router, "__name__", "router"), ", ".join(sorted(set(table.values()))))
            else:
                arrow = "-> %s" % self.edges.get(name, END)
            lines.append("  [%s] %s" % (name, arrow))
        return "\n".join(lines)

    # -- compilation -------------------------------------------------------- #
    def compile(self, *, max_steps: int = 64, default_max_visits: int = 8) -> "CompiledGraph":
        self._validate()
        limits = {name: default_max_visits for name in self.nodes}
        limits.update(self.max_visits)
        return CompiledGraph(self, max_steps=max_steps, max_visits=limits)


class CompiledGraph:
    def __init__(self, graph: StateGraph, *, max_steps: int, max_visits: Mapping[str, int]) -> None:
        self.graph = graph
        self.max_steps = int(max_steps)
        self.max_visits = dict(max_visits)

    # -- routing ------------------------------------------------------------ #
    def route(self, node: str, state: Mapping[str, Any]) -> str:
        if node in self.graph.branches:
            router, table = self.graph.branches[node]
            key = router(dict(state))
            if key not in table:
                raise GraphError("router %s returned %r; expected one of %s" % (node, key, sorted(table)))
            return table[key]
        return self.graph.edges.get(node, END)

    # -- execution --------------------------------------------------------- #
    def stream(self, state: Dict[str, Any], *, listener: Optional[Listener] = None) -> Iterator[Dict[str, Any]]:
        """Execute ``state`` in place, yielding one event per node execution.

        The caller's mapping object is mutated as nodes run, so a consumer that
        wants the final state can simply keep its own reference.

        Two events reach ``listener`` per node -- ``start`` before the node body
        runs and ``done`` after it returns or raises -- because a node is not
        fast: an agent-backed node takes minutes, and a view that only hears from
        a node once it has finished cannot tell "still working" from "hung".
        ``run()`` used to drop the yielded completion event on the floor, so the
        live view said ``ok 0ms -> `` for every node and ``trace.jsonl`` recorded
        no durations or routing at all.
        """
        work: Dict[str, Any] = state
        node = self.route(START, work)
        step = 0
        while node != END:
            step += 1
            if step > self.max_steps:
                raise StepLimitExceeded("step budget %d exhausted" % self.max_steps)
            visits = work.setdefault("node_visits", {})
            visits[node] = int(visits.get(node, 0)) + 1
            limit = self.max_visits.get(node, 8)
            if visits[node] > limit:
                raise StepLimitExceeded("node %r visited %d times (limit %d)" % (node, visits[node], limit))

            started = time.time()
            event: Dict[str, Any] = {
                "type": "node",
                "event": "start",
                "node": node,
                "step": step,
                "visit": visits[node],
                "cycle": work.get("cycle", 0),
                "started_at": started,
            }
            if listener:
                listener(for_listener(event))

            try:
                updates = self.graph.nodes[node](work)
            except Exception as exc:  # node failure is a first-class event
                event.update({"event": "done", "ok": False, "error": "%s: %s" % (type(exc).__name__, exc), "touched": []})
                event["duration_ms"] = round((time.time() - started) * 1000, 1)
                if listener:
                    listener(for_listener(event))
                yield {k: v for k, v in event.items() if k != "started_at"}
                raise

            touched = apply_updates(work, updates)
            event.update({"event": "done", "ok": True, "touched": touched, "delta": unwrap(updates)})
            event["duration_ms"] = round((time.time() - started) * 1000, 1)

            nxt = self.route(node, work)
            event["next"] = nxt
            if nxt == node:
                event["self_loop"] = True
            if listener:
                listener(for_listener(event))
            yield {k: v for k, v in event.items() if k != "started_at"}

            if work.get("stop_requested"):
                work["status"] = "human_review"
                work["status_reason"] = work.get("status_reason") or "stop requested by policy/operator"
                return
            node = nxt

    def run(self, state: Dict[str, Any], *, listener: Optional[Listener] = None) -> Dict[str, Any]:
        """Run to completion and return the final state (the same object as ``state``).

        Deliberately *not* a copy.  When a node raises, the caller keeps the
        partial state -- the issues already found, the reviews already cast, the
        tokens already paid for.  Copying here meant one malformed agent answer
        erased an entire run from the report: the run directory said "0 issues,
        no issue was reported" for a run that had found two and gathered three
        approvals, which is indistinguishable from a clean repository.
        """
        for _ in self.stream(state, listener=listener):
            pass
        return state
    # -- introspection ------------------------------------------------------ #
    def mermaid(self) -> str:
        """Render the compiled topology as a Mermaid flowchart."""
        lines = ["flowchart TD", "  START([START])", "  DONE([END])"]
        for name in self.graph.nodes:
            lines.append('  %s["%s"]' % (name, name))
        for entry in self.graph.outgoing(START):
            lines.append("  START --> %s" % entry)
        for name in self.graph.nodes:
            if name in self.graph.branches:
                for key, target in sorted(self.graph.branches[name][1].items()):
                    lines.append("  %s -->|%s| %s" % (name, key, "DONE" if target == END else target))
            else:
                target = self.graph.edges.get(name, END)
                lines.append("  %s --> %s" % (name, "DONE" if target == END else target))
        return "\n".join(lines)
