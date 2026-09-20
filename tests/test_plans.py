# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""The plan checks: contradictions a machine can see before a reviewer reads.

Every case here is a shape a measured run actually produced.  The point of the
checks is that they cost nothing -- a reviewer round is three model calls and
20-60 seconds, and three of four rounds in that run went to defects that are
visible by comparing the plan against itself.
"""

from __future__ import annotations

from qedloop.plans import check_plan

CODE = {
    "tests/unit/test_system_service.py": "def test_webui_sentinel_is_load_bearing():\n    pass\n",
    "src/app/system_service.py": "def exit_application():\n    pass\n",
}


def _finding(*actions):
    return {"issue_id": "ISS-0003", "steps": [{"order": i, "action": a} for i, a in enumerate(actions, start=1)]}


def test_a_k_filter_that_selects_nothing_is_caught():
    """Measured: `-k not_running` against `test_exit_application_skips_stop_scan_when_scanner_idle`.

    The step that proves the change is load-bearing would have collected no test
    at all, so "break it and watch it fail" was unfalsifiable as written.
    """
    problems = check_plan(
        _finding("Run `pytest tests/unit/test_system_service.py -k not_running` and record that it fails"),
        [{"name": "test_exit_application_skips_stop_scan_when_scanner_idle", "path": "tests/unit/test_system_service.py"}],
        code=CODE,
    )
    assert problems, "a filter that matches no test must not pass"
    assert "-k not_running" in problems[0], problems


def test_a_node_id_no_planned_test_has_is_caught():
    """Measured: step 7 named a test the planned-test list had named differently."""
    problems = check_plan(
        _finding("Run `pytest tests/unit/test_system_service.py::test_webui_sentinel_not_accessed_when_window_injected -x`"),
        [{"name": "test_exit_application_with_injected_window_never_reads_webui_module", "path": "tests/unit/test_system_service.py"}],
        code=CODE,
    )
    assert problems, "a node id that exists nowhere must not pass"
    assert "not_accessed_when_window_injected" in problems[0], problems


def test_a_filter_that_selects_the_wrong_existing_test_is_caught():
    """Measured: `-k not_running` on the very file the plan was adding
    `test_..._when_scanner_idle` to.

    It collected an older, similarly named test instead -- so the step that proves
    the change is load-bearing would have exercised the wrong test and passed,
    which is worse than collecting nothing.  The objection states the fact and
    the way out rather than accusing the plan of anything.
    """
    problems = check_plan(
        _finding("Falsification: run `pytest tests/unit/test_system_service.py -k not_running` and record the failure"),
        [{"name": "test_exit_application_skips_stop_scan_when_scanner_idle", "path": "tests/unit/test_system_service.py"}],
        code={
            "tests/unit/test_system_service.py": (
                "def test_webui_sentinel_is_load_bearing():\n    pass\n"
                "def test_exit_application_skips_stop_scan_when_scanner_not_running():\n    pass\n"
            )
        },
    )
    assert problems, "a filter that misses every test the plan adds must not pass"
    assert "selects none of the tests this plan adds" in problems[0], problems
    assert "when_scanner_not_running" in problems[0], problems


def test_a_filter_over_a_file_the_plan_does_not_touch_is_left_alone():
    """A regression sweep is written this way: -k over tests the tree already has.

    Nothing here contradicts anything else, so the check stays silent -- the rule
    is scoped to files the plan is itself adding tests to.
    """
    problems = check_plan(
        _finding("Run `pytest tests/unit/test_stats.py -k mean` and record that it still passes"),
        [{"name": "test_exit_application_skips_stop_scan_when_scanner_idle", "path": "tests/unit/test_system_service.py"}],
        code=dict(CODE, **{"tests/unit/test_stats.py": "def test_mean_of_empty_is_zero():\n    pass\n"}),
    )
    assert problems == [], problems


def test_a_planned_test_pytest_cannot_collect_is_caught():
    problems = check_plan(_finding("Write the regression test"), [{"name": "sentinel_is_load_bearing", "path": ""}], code=CODE)
    assert problems and "test_" in problems[0], problems


def test_a_second_module_with_the_same_basename_is_caught_when_no_step_mentions_it():
    """Measured: creating `tests/unit/services/test_system_service.py` while
    `tests/unit/test_system_service.py` was already in the tree, with no step
    saying whether the older module stays, moves, or is dead."""
    problems = check_plan(
        _finding("Create `tests/unit/services/__init__.py` and `tests/unit/services/test_system_service.py`"),
        [{"name": "test_a", "path": "tests/unit/services/test_system_service.py"}],
        code=CODE,
    )
    assert problems, "two test modules with one basename and no migration story must not pass"
    assert "tests/unit/test_system_service.py" in problems[0], problems


def test_migrating_a_test_module_is_not_flagged():
    """A plan that names the older file has answered the question."""
    problems = check_plan(
        _finding("Move `tests/unit/test_system_service.py` into `tests/unit/services/test_system_service.py` and fix the imports"),
        [{"name": "test_a", "path": "tests/unit/services/test_system_service.py"}],
        code=CODE,
    )
    assert problems == [], problems


def test_the_same_name_planned_twice_is_caught():
    problems = check_plan(
        _finding("Add both tests"),
        [{"name": "test_dup", "path": "tests/unit/test_a.py"}, {"name": "test_dup", "path": "tests/unit/test_a.py"}],
        code=CODE,
    )
    assert problems and "planned twice" in problems[0], problems


def test_a_consistent_plan_reports_nothing():
    problems = check_plan(
        _finding(
            "Add `test_exit_application_skips_stop_scan` to `tests/unit/test_system_service.py`, then run "
            "`pytest tests/unit/test_system_service.py::test_exit_application_skips_stop_scan -q`"
        ),
        [{"name": "test_exit_application_skips_stop_scan", "path": "tests/unit/test_system_service.py"}],
        code=CODE,
    )
    assert problems == [], problems


def test_an_empty_plan_is_not_a_problem():
    assert check_plan({}, [], code=CODE) == []
    assert check_plan(None, None, code={}) == []
