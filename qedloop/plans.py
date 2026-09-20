# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Mechanical checks on a refined plan, before three model calls judge it.

A plan is a document that tells the implementer what to run.  When its own
commands cannot collect the tests it names, that is a contradiction *inside the
document*, and no reviewer judgement is needed to find it.  One measured run
spent three of its four review rounds on exactly this class of defect:

* ``pytest tests/unit/test_system_service.py -k not_running`` against a planned
  test called ``test_exit_application_skips_stop_scan_when_scanner_idle`` -- the
  command collects nothing, so the "break it and watch it fail" step is vacuous;
* the same filter on the same file, one round later, collecting an *older* test
  called ``..._when_scanner_not_running`` instead: the step would have exercised
  the wrong test and passed, which is worse than collecting nothing;
* a step running ``::test_webui_sentinel_not_accessed_when_window_injected``
  while the planned-test list named the same test
  ``test_exit_application_with_injected_window_never_reads_webui_module``;
* a plan creating ``tests/unit/services/test_system_service.py`` while the
  workspace already carried ``tests/unit/test_system_service.py`` and no step
  said what should happen to the older module.

Everything here is a string or path comparison over the plan and the candidate
tree: no model call, no I/O, no clock.  The checks are deliberately conservative,
because a false positive sends a workable plan back for another round -- each one
fires only on a contradiction between two parts of the plan (or between the plan
and a file it never mentions), never on a plan that merely looks unusual.  What
they cannot see is judgement: nobody here can tell that a sentinel recording
every attribute access will record ``__path__`` first, because that needs the
measurement, and the lenses are still there for it.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

#: The planning lens quotes the commands it wants run; a bare ``pytest ...``
#: outside backticks is still a command, so both forms are collected.
_QUOTED = re.compile(r"`([^`\n]*)`")
_BARE = re.compile(r"\bpytest\s[^\n`;]*")
_DEF = re.compile(r"^[ \t]*def[ \t]+(test_[A-Za-z0-9_]+)", re.MULTILINE)
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: ``and``/``or``/``not`` are pytest's own operators, not fragments of a name.
_KEYWORDS = frozenset({"and", "or", "not"})


def _test_names(code: Mapping[str, str]) -> Set[str]:
    """Every test function the candidate tree already defines."""
    names: Set[str] = set()
    for text in (code or {}).values():
        names.update(_DEF.findall(str(text)))
    return names


def _normalise(path: str) -> str:
    return str(path).replace("\\", "/").lstrip("./")


def _commands(action: str) -> List[str]:
    """The ``pytest`` invocations a step asks for, deduplicated in order."""
    found = [chunk.strip() for chunk in _QUOTED.findall(action) if "pytest" in chunk]
    found += [chunk.strip() for chunk in _BARE.findall(action)]
    out: List[str] = []
    for command in found:
        if command and command not in out:
            out.append(command)
    return out


def _targets(command: str) -> Tuple[List[str], List[str], List[str]]:
    """``(node ids, -k expressions, .py paths)`` named by one command line."""
    parts = command.replace('"', " ").replace("'", " ").split()
    node_ids: List[str] = []
    filters: List[str] = []
    paths: List[str] = []
    index = 0
    while index < len(parts):
        part = parts[index]
        if part in ("-k", "--keyword") and index + 1 < len(parts):
            filters.append(parts[index + 1])
            index += 2
            continue
        if part.startswith("-k") and len(part) > 2:
            filters.append(part[2:])
            index += 1
            continue
        if "::" in part:
            # ``path.py::Class::test_x[param]`` -- pytest collects by the last
            # segment, and a parametrised id still names the same function.
            node_ids.append(part.split("::")[-1].split("[")[0])
            part = part.split("::")[0]
        if part.endswith(".py"):
            paths.append(_normalise(part))
        index += 1
    return node_ids, filters, paths


def check_plan(
    finding: Mapping[str, Any] | None,
    plans: Sequence[Mapping[str, Any]] | None,
    *,
    code: Mapping[str, str],
) -> List[str]:
    """Problems that stop the plan from proving what it claims.  Empty means ok.

    Returns human-readable reasons, ready to be raised the way a reviewer's
    objection is: each one names the step or the plan it came from, because the
    next refinement round has to act on it without seeing this function.
    """
    steps = [step for step in ((finding or {}).get("steps") or []) if isinstance(step, Mapping)]
    planned = [plan for plan in (plans or []) if isinstance(plan, Mapping)]
    known_paths = {_normalise(path) for path in (code or {})}
    planned_names = {str(plan.get("name") or "").strip() for plan in planned} - {""}
    known_names = _test_names(code) | planned_names
    problems: List[str] = []

    for index, step in enumerate(steps, start=1):
        action = str(step.get("action") or "")
        for command in _commands(action):
            node_ids, filters, paths = _targets(command)
            for name in node_ids:
                if name and name not in known_names:
                    problems.append(
                        "step %d runs `%s`, but no planned or existing test is called `%s`"
                        % (index, command, name)
                    )
            for expression in filters:
                tokens = {token for token in _TOKEN.findall(expression)} - _KEYWORDS
                if not tokens:
                    continue
                # pytest's -k matches substrings, so a token only has to appear
                # inside one known name.
                matched = {name for name in known_names if any(token in name for token in tokens)}
                if not matched:
                    problems.append(
                        "step %d filters with `-k %s`, which selects no planned or existing test "
                        "(planned: %s)" % (index, expression, ", ".join(sorted(planned_names)) or "none")
                    )
                    continue
                # Selecting *something* is not enough when the command names a
                # file this plan is adding tests to.  Measured: a falsification
                # step ran `-k not_running` on the very file it was adding
                # `test_..._when_scanner_idle` to, and the filter collected an
                # older, similarly named test instead -- so the step that proves
                # the change is load-bearing would have observed the wrong test
                # and passed.  Stated as a fact, with the way out: name the test.
                added_here = sorted(
                    str(plan.get("name"))
                    for plan in planned
                    if _normalise(plan.get("path") or "") in set(paths)
                )
                if added_here and not (matched & planned_names):
                    problems.append(
                        "step %d runs `-k %s` on %s, which selects none of the tests this plan adds there "
                        "(%s): it selects %s instead"
                        % (index, expression, ", ".join(sorted(set(paths))), ", ".join(added_here),
                           ", ".join(sorted(matched)[:3]))
                    )

    seen: Dict[Tuple[str, str], int] = {}
    for plan in planned:
        name = str(plan.get("name") or "").strip()
        path = _normalise(plan.get("path") or "")
        if name and not name.startswith("test_"):
            problems.append("planned test `%s` is not collected by pytest: names must start with `test_`" % name)
        key = (path, name)
        if name and key in seen:
            problems.append("`%s` is planned twice for %s" % (name, path or "a new file"))
        seen[key] = 1
        # A new test module sharing a basename with an existing one is the shape
        # that produced "either the existing flat module is orphaned or the
        # planned package is never populated".  Silent when any step mentions the
        # older file: a plan that migrates or replaces it has said what it means.
        if not path or path in known_paths or not path.rsplit("/", 1)[-1].startswith("test_"):
            continue
        twins = sorted(candidate for candidate in known_paths if candidate.rsplit("/", 1)[-1] == path.rsplit("/", 1)[-1])
        mentioned = any(twin in _normalise(str(step.get("action") or "")) for twin in twins for step in steps)
        if twins and not mentioned:
            problems.append(
                "the plan adds `%s` while `%s` already exists with the same module name, and no step "
                "says what happens to the older file" % (path, twins[0])
            )

    # A generation lists one row per test, so the same file-level collision is
    # found once per row.  The objections are read by a person next round, not
    # counted, so report each distinct one once, in the order it was found.
    unique: List[str] = []
    for problem in problems:
        if problem not in unique:
            unique.append(problem)
    return unique
