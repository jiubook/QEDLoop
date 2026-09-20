# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Verification layer tests: anchored edits, markers, static checks, sandbox."""

from __future__ import annotations

import sys

import pytest

from qedloop.core import detect_markers, render_markers
from qedloop.sandbox import (
    apply_ops,
    coverable_files,
    lint_argv,
    materialise,
    parse_pytest_summary,
    pytest_available,
    run_lint,
    run_pytest,
    seed_from_directory,
    static_analysis,
    verification_snapshot,
)


def _pytest_blocked(report) -> bool:
    """True when the host sandbox refused to start the subprocess."""
    return "could not start pytest" in (report.error or "")


# --------------------------------------------------------------------------- #
# anchored edits
# --------------------------------------------------------------------------- #


def test_apply_ops_replaces_a_unique_anchor():
    code = {"src/a.py": "def f():\n    return 1\n"}
    result = apply_ops(code, [{"path": "src/a.py", "search": "return 1", "replace": "return 2", "rationale": "fix"}])
    assert result.ok and result.code["src/a.py"] == "def f():\n    return 2\n"
    assert result.applied[0]["path"] == "src/a.py"


def test_apply_ops_rejects_a_missing_anchor():
    result = apply_ops({"src/a.py": "x = 1\n"}, [{"path": "src/a.py", "search": "y = 2", "replace": "z"}])
    assert not result.ok and "anchor not found" in result.errors[0]
    assert result.code == {"src/a.py": "x = 1\n"}, "a failed apply must not modify the tree"


def test_apply_ops_rejects_an_ambiguous_anchor():
    result = apply_ops({"src/a.py": "x = 1\nx = 1\n"}, [{"path": "src/a.py", "search": "x = 1", "replace": "x = 2"}])
    assert not result.ok and "not unique" in result.errors[0]


def test_apply_ops_rejects_unknown_paths():
    result = apply_ops({"src/a.py": "x = 1\n"}, [{"path": "src/ghost.py", "search": "x", "replace": "y"}])
    assert not result.ok and "not part of the candidate tree" in result.errors[0]


def test_apply_ops_is_all_or_nothing():
    code = {"src/a.py": "x = 1\n", "src/b.py": "y = 1\n"}
    result = apply_ops(code, [
        {"path": "src/a.py", "search": "x = 1", "replace": "x = 2"},
        {"path": "src/b.py", "search": "not here", "replace": "y = 2"},
    ])
    assert not result.ok
    assert result.code == code


def test_apply_ops_handles_multiline_anchors():
    code = {"src/a.py": "def f():\n    a = 1\n    b = 2\n    return a + b\n"}
    result = apply_ops(code, [{"path": "src/a.py", "search": "    a = 1\n    b = 2", "replace": "    a = 2\n    b = 3"}])
    assert result.ok and "a = 2" in result.code["src/a.py"]


def test_an_lf_anchor_matches_a_crlf_file():
    """Line endings are content to a substring match and nothing to a reader.

    Reproduces a real run: EER-Ai is checked out with CRLF (``core.autocrlf`` is
    true and there is no ``.gitattributes``), the candidate tree kept those
    endings, and the model returned its anchors with LF.  All three ops failed as
    "anchor not found" and the run closed nothing.  Replayed against the stored
    patch: CRLF body -> ok=False, same ops with an LF body -> ok=True.
    """
    code = {"src/a.py": "def f():\r\n    return 1\r\n"}
    result = apply_ops(code, [{"path": "src/a.py", "search": "def f():\n    return 1", "replace": "def f():\n    return 2"}])
    assert result.ok, result.errors
    assert "return 2" in result.code["src/a.py"]


def test_a_patched_crlf_file_stays_uniformly_crlf():
    """Splicing LF lines into a CRLF file must not leave it half and half.

    Python accepts mixed endings, so nothing downstream would notice -- the file
    just quietly becomes a patch no reviewer would call clean.
    """
    code = {"src/a.py": "def f():\r\n    a = 1\r\n    b = 2\r\n"}
    result = apply_ops(
        code,
        [{"path": "src/a.py", "search": "    a = 1\n    b = 2", "replace": "    a = 2\n    b = 3\n    c = 4"}],
    )
    assert result.ok, result.errors
    patched = result.code["src/a.py"]
    assert patched.count("\r\n") == 4, "every line keeps the file's own convention"
    assert patched.count("\n") - patched.count("\r\n") == 0, "no lone LF was introduced"


def test_a_crlf_anchor_is_never_doubled():
    """An anchor that already uses CRLF must not become ``\\r\\r\\n``."""
    code = {"src/a.py": "x = 1\r\n"}
    result = apply_ops(code, [{"path": "src/a.py", "search": "x = 1\r\n", "replace": "x = 2\r\n"}])
    assert result.ok, result.errors
    assert result.code["src/a.py"] == "x = 2\r\n"


def test_a_mixed_line_ending_replacement_is_normalised_too():
    """A replacement can arrive with endings of its own, mixed or not.

    Testing only for "the replacement has no CRLF anywhere" skips the LF lines
    of a half-converted block, which puts the mixed endings right back.
    """
    code = {"src/a.py": "def f():\r\n    return 1\r\n"}
    anchor = "def f():\n    return 1"
    replacement = "def f():\r\n    value = 1\n    return value"
    result = apply_ops(code, [{"path": "src/a.py", "search": anchor, "replace": replacement}])
    assert result.ok, result.errors
    patched = result.code["src/a.py"]
    assert patched.count("\n") - patched.count("\r\n") == 0, "no lone LF survives the splice"
    assert patched.count("\r\n") == 3


def test_line_ending_tolerance_does_not_weaken_ambiguity():
    """Trying two forms must not turn an ambiguous anchor into a silent pick."""
    code = {"src/a.py": "x = 1\r\nx = 1\r\n"}
    result = apply_ops(code, [{"path": "src/a.py", "search": "x = 1\n", "replace": "x = 2\n"}])
    assert not result.ok
    assert "not unique" in result.errors[0], result.errors


def test_a_missing_anchor_is_still_a_hard_failure():
    """Tolerance for endings is not tolerance for absence."""
    result = apply_ops({"src/a.py": "x = 1\r\n"}, [{"path": "src/a.py", "search": "y = 2\n", "replace": "z\n"}])
    assert not result.ok and "anchor not found" in result.errors[0]


# --------------------------------------------------------------------------- #
# adding a file
# --------------------------------------------------------------------------- #


def test_an_op_with_no_search_adds_a_new_file():
    """The ops contract had no way to say "create this file".

    Reproduces a real run: the patch added a regression test in
    ``tests/unit/test_matrix_export_route.py``, had to send an empty ``search``
    because the file had no contents to anchor to, and the whole proposal -- a
    good one -- was rejected as malformed.  The agent's own rationale said "新文件
    无既有内容，故 search 为空".
    """
    reasoning = "新增回归测试文件：断言超限请求下 b64decode 从未被调用"
    result = apply_ops(
        {"src/mod.py": "value = 1\n"},
        [{"path": "tests/unit/test_new_mod.py", "search": "", "replace": "def test_it():\n    assert True\n",
          "rationale": reasoning}],
    )
    assert result.ok, result.errors
    assert result.code["tests/unit/test_new_mod.py"] == "def test_it():\n    assert True\n"
    assert result.applied[0]["created"] is True, "the report can tell a creation from an edit"


def test_creating_a_file_that_exists_is_refused():
    """An empty search must never become "replace the whole file"."""
    result = apply_ops(
        {"src/a.py": "a = 1\n"},
        [{"path": "src/a.py", "search": "", "replace": "a = 2\n", "rationale": "新增文件"}],
    )
    assert not result.ok
    assert "already exists" in result.errors[0]


def test_creating_a_file_needs_the_intent_stated():
    """A truncated reply also arrives with an empty search; it must not pass."""
    result = apply_ops(
        {},
        [{"path": "src/new.py", "search": "", "replace": "x = 1\n", "rationale": "fixes the issue"}],
    )
    assert not result.ok
    assert "does not say the file is new" in result.errors[0]


def test_creating_a_file_needs_content_and_a_python_path():
    empty = apply_ops({}, [{"path": "src/new.py", "search": "", "replace": "  \n", "rationale": "new file"}])
    assert not empty.ok and "needs content" in empty.errors[0]

    wrong_kind = apply_ops({}, [{"path": "docs/notes.txt", "search": "", "replace": "hi\n", "rationale": "new file"}])
    assert not wrong_kind.ok and "only python files" in wrong_kind.errors[0]

    no_path = apply_ops({}, [{"search": "", "replace": "x = 1\n", "rationale": "new file"}])
    assert not no_path.ok and "path is required" in no_path.errors[0]


def test_an_edit_with_no_search_is_refused_for_a_file_that_does_not_exist():
    """Creation is not a licence to write into a path the tree never had, silently."""
    result = apply_ops({}, [{"path": "src/ghost.py", "search": "x", "replace": "y"}])
    assert not result.ok and "not part of the candidate tree" in result.errors[0]


# --------------------------------------------------------------------------- #
# declared defect markers
# --------------------------------------------------------------------------- #


def test_detect_markers_reads_name_kind_and_line():
    markers = detect_markers({"a.py": "x = 1\n# BUG: demo/thing -- it is wrong\ny = 2\n# TODO: later\n"})
    assert len(markers) == 1, "TODO is not an actionable defect marker"
    assert markers[0].name == "demo/thing"
    assert markers[0].line == 2
    assert markers[0].actionable is True
    assert "it is wrong" in markers[0].note


def test_render_markers_is_readable_when_empty():
    assert render_markers([]) == "(no declared defect markers)"


def test_markers_keys_are_stable_across_line_shifts():
    before = detect_markers({"a.py": "# BUG: demo/thing -- wrong\nx = 1\n"})
    after = detect_markers({"a.py": "\n\n# BUG: demo/thing -- wrong\nx = 1\n"})
    assert before[0].key == after[0].key == "demo/thing"


# --------------------------------------------------------------------------- #
# static analysis
# --------------------------------------------------------------------------- #


def test_static_analysis_reports_resolved_markers_and_compiles():
    before = {"a.py": "# BUG: demo/thing -- wrong\nx = 1\n"}
    after = {"a.py": "x = 2\n"}
    report = static_analysis(before, after)
    assert report["compiles"] is True
    assert [m["name"] for m in report["markers_resolved"]] == ["demo/thing"]
    assert report["markers_after"] == 0
    assert report["changed_files"] == ["a.py"]


def test_static_analysis_catches_a_syntax_error():
    report = static_analysis({"a.py": "x = 1\n"}, {"a.py": "def f(:\n"})
    assert report["compiles"] is False
    assert report["syntax_errors"][0]["path"] == "a.py"


def test_static_analysis_flags_newly_introduced_markers():
    report = static_analysis({"a.py": "x = 1\n"}, {"a.py": "# FIXME: later/later -- oops\nx = 1\n"})
    assert [m["name"] for m in report["markers_introduced"]] == ["later/later"]


# --------------------------------------------------------------------------- #
# pytest parsing + sandbox
# --------------------------------------------------------------------------- #


def test_parse_pytest_summary_reads_counts():
    text = "=== 2 failed, 5 passed, 1 skipped, 3 warnings in 0.12s ==="
    counts = parse_pytest_summary(text)
    assert counts == {"passed": 5, "failed": 2, "errors": 0, "skipped": 1}


def test_parse_pytest_summary_does_not_bleed_across_lines():
    """A count and its word must be on one line.

    With ``\\s+`` between them, an assertion message ending in a number reached
    across the newline into the following ``FAILED ...`` line: ``assert 0 == 5``
    plus ``FAILED tests/...`` parsed as "5 failed" on top of the real "4 failed".
    """
    text = (
        "E       AssertionError: assert 0 == 5\n"
        "FAILED tests/test_x.py::test_clamp\n"
        "=========================== short test summary info ===========================\n"
        "FAILED tests/test_x.py::test_clamp - assert 0 == 5\n"
        "4 failed, 4 passed in 0.13s\n"
    )
    counts = parse_pytest_summary(text)
    assert counts == {"passed": 4, "failed": 4, "errors": 0, "skipped": 0}, counts


def test_coverable_files_detects_test_modules():
    assert coverable_files({"a.py": "", "tests/test_a.py": "", "b_test.py": ""}) == ["tests/test_a.py", "b_test.py"]


def test_run_pytest_without_tests_errors_cleanly():
    report = run_pytest({"mod.py": "x = 1\n"})
    assert not report.ran and "no test files" in report.error


@pytest.mark.skipif(not pytest_available(), reason="pytest is not importable in this interpreter")
def test_sandbox_runs_a_green_suite(demo_suite):
    report = run_pytest(demo_suite, timeout=90)
    if _pytest_blocked(report):
        pytest.skip("host sandbox refused to start pytest: %s" % report.error)
    assert report.green, report.to_dict()
    assert report.passed == 1 and report.workspace == "", "the scratch tree must be cleaned up"


@pytest.mark.skipif(not pytest_available(), reason="pytest is not importable in this interpreter")
def test_sandbox_measures_a_regression_and_a_fix(demo_suite):
    broken = dict(demo_suite)
    broken["mod.py"] = "def add(a, b):\n    return a - b\n"
    snapshot = verification_snapshot(broken, demo_suite, timeout=90)
    tests = snapshot["tests"]
    if not tests.get("ran"):
        pytest.skip("host sandbox refused to start pytest: %s" % tests.get("after", {}).get("error"))
    assert tests["after"]["green"] is True
    assert tests["before"]["green"] is False
    assert tests["regression"]["fixed"] >= 1
    assert snapshot["static"]["compiles"] is True


def test_seed_from_directory_skips_caches(demo_copy):
    seeded = seed_from_directory(demo_copy)
    assert "src/tinylib/stats.py" in seeded
    assert not [p for p in seeded if "__pycache__" in p]


# --------------------------------------------------------------------------- #
# repository-copy sandbox
# --------------------------------------------------------------------------- #


def test_copy_repository_brings_non_python_assets(tmp_path):
    """A suite that reads an image or a JSON fixture must still work.

    Copying only the supplied ``.py`` files reported failures that did not exist
    in the real repository -- the sandbox must mirror the tree.
    """
    from qedloop.sandbox import copy_repository

    source = tmp_path / "repo"
    (source / "src").mkdir(parents=True)
    (source / "resources").mkdir()
    (source / "tests").mkdir()
    (source / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    (source / "src" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (source / "resources" / "fixture.png").write_bytes(b"\x89PNG\r\n\x1a\n fake")
    (source / "tests" / "test_mod.py").write_text("def test_ok():\n    pass\n", encoding="utf-8")
    (source / ".venv").mkdir()
    (source / ".venv" / "junk.py").write_text("x = 1\n", encoding="utf-8")
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "mod.cpython-313.pyc").write_bytes(b"\x00")

    destination = tmp_path / "sandbox"
    destination.mkdir()
    copied = copy_repository(source, destination)

    assert (destination / "pyproject.toml").is_file()
    assert (destination / "resources" / "fixture.png").is_file()
    assert (destination / "src" / "mod.py").is_file()
    assert not (destination / ".venv").exists(), "virtualenvs are not part of the repository"
    assert not (destination / "__pycache__").exists()
    assert copied >= 4


def test_sandbox_overlays_candidate_files_on_the_repository_copy(tmp_path):
    """The candidate edit must win over the copy of the original file."""
    from qedloop.sandbox import run_pytest

    source = tmp_path / "repo"
    (source / "tests").mkdir(parents=True)
    (source / "mod.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (source / "tests" / "test_mod.py").write_text(
        "from mod import value\n\n\ndef test_value():\n    assert value() == 2\n", encoding="utf-8"
    )

    before = run_pytest({"mod.py": "def value():\n    return 1\n"}, target_root=str(source), timeout=90)
    if _pytest_blocked(before):
        pytest.skip("host sandbox refused to start pytest: %s" % before.error)
    assert before.failed == 1, "the unpatched repository copy must fail"

    after = run_pytest({"mod.py": "def value():\n    return 2\n"}, target_root=str(source), timeout=90)
    assert after.green is True, "the overlaid candidate must be what runs"


def test_sandbox_shadows_an_installed_copy_of_a_src_layout_project(tmp_path, monkeypatch):
    """A ``src/`` layout must not fall through to the ambient install.

    Real repositories are importable from the surrounding environment all the
    time -- an editable install, or a ``.pth`` file pointing at the checkout.
    Under a ``src/`` layout the package is *not* at the repository root, so if
    the sandbox only puts its root on ``sys.path`` the import resolves to the
    original tree and every candidate measures as the unmodified repository.
    That is a number which cannot fail, so the run can "verify" anything.
    """
    original = tmp_path / "original" / "src" / "qedloop_shadow_probe"
    original.mkdir(parents=True)
    (original / "__init__.py").write_text('VALUE = "original"\n', encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(original.parents[1]))

    repository = tmp_path / "repo"
    package = repository / "src" / "qedloop_shadow_probe"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('VALUE = "original"\n', encoding="utf-8")
    (repository / "tests").mkdir()
    (repository / "tests" / "test_value.py").write_text(
        "from qedloop_shadow_probe import VALUE\n\n\ndef test_value():\n"
        '    assert VALUE == "candidate"\n',
        encoding="utf-8",
    )

    report = run_pytest(
        {"src/qedloop_shadow_probe/__init__.py": 'VALUE = "candidate"\n'},
        target_root=str(repository),
        timeout=90,
    )
    if _pytest_blocked(report):
        pytest.skip("host sandbox refused to start pytest: %s" % report.error)
    assert report.green is True, "the sandbox's src/ must win over the installed original: %s" % report.to_dict()


def test_check_measures_the_repository_copy(tmp_path, capsys):
    """``run.py check`` must report the same baseline the loop will measure.

    Materialising only the matched ``.py`` files made ``check`` report failures
    that the real repository does not have -- here, a test that reads a JSON
    fixture outside the Python sources.
    """
    from argparse import Namespace

    from qedloop.cli import cmd_check

    repository = tmp_path / "repo"
    (repository / "resources").mkdir(parents=True)
    (repository / "resources" / "fixture.json").write_text('{"value": 2}', encoding="utf-8")
    (repository / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repository / "tests").mkdir()
    (repository / "tests" / "test_fixture.py").write_text(
        "import json\nimport pathlib\n\n\ndef test_fixture():\n"
        "    data = json.loads(pathlib.Path('resources/fixture.json').read_text(encoding='utf-8'))\n"
        "    assert data['value'] == 2\n",
        encoding="utf-8",
    )

    exit_code = cmd_check(Namespace(target=str(repository), include="*.py", no_tests=False))
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "baseline suite: green" in output, "check must measure the copy, not just the .py files:\n%s" % output


def test_materialise_keeps_a_crlf_body_byte_for_byte(tmp_path):
    """The measured tree must be the patch's own text, not a re-encoded copy.

    ``write_text`` translates every "\\n" to the platform separator by default,
    and a candidate body already carries the *target's* convention, so on a CRLF
    checkout every "\\r\\n" came back as "\\r\\r\\n" -- which reads as two line
    breaks and inserts a blank line between every line of every edited file.
    Measured on a real run: the patch applied to the target was 74 insertions /
    5 deletions where the model wrote roughly 30 / 6.  Nothing caught it because
    Python does not care about blank lines, so the suite stayed green and the
    existing line-ending test only counted "\\r\\n" (which "\\r\\r\\n" contains).
    """
    body = "def f():\r\n    return 1\r\n"

    materialise({"pkg/mod.py": body}, tmp_path)

    raw = (tmp_path / "pkg" / "mod.py").read_bytes()
    assert raw == body.encode("utf-8"), "the candidate body was rewritten in transit"
    assert b"\r\r\n" not in raw, "a doubled carriage return reads as an extra blank line"


# --------------------------------------------------------------------------- #
# the target's own static check
# --------------------------------------------------------------------------- #


def _lint_target(tmp_path):
    """A stand-in repository whose "linter" needs no third-party dependency."""
    repo = tmp_path / "target"
    repo.mkdir()
    # Stands in for the target's own configuration file: the rules live in the
    # repository, which is why the check has to run against a copy of it.
    (repo / "ruff.toml").write_text("line-length = 88\n", encoding="utf-8")
    (repo / "check.py").write_text(
        "import sys\n"
        "problems = [p for p in sys.argv[1:] if 'BAD' in open(p, encoding='utf-8').read()]\n"
        "for p in problems:\n"
        "    print('%s: pretend violation' % p)\n"
        "sys.exit(1 if problems else 0)\n",
        encoding="utf-8",
    )
    return repo


def test_lint_argv_expands_placeholders_token_by_token():
    """``{files}`` must be a token of its own so one file becomes one argument.

    The template is split without a shell on purpose: a quoting layer would
    behave differently on Windows, and the same command has to mean the same
    thing in every sandbox.
    """
    argv = lint_argv("{python} -m ruff check {files}", ["a.py", "b/c.py"])
    assert argv[0] == sys.executable
    assert argv[1:] == ["-m", "ruff", "check", "a.py", "b/c.py"]
    assert lint_argv("ruff check {files}", []) == ["ruff", "check"]


def test_run_lint_grades_the_candidate_tree_and_scopes_to_the_changed_files(tmp_path):
    """Only this patch's files may fail the gate.

    A whole-repository check would also fail on violations the loop is not
    allowed to touch, which is a blocker it could never clear.
    """
    repo = _lint_target(tmp_path)
    command = "{python} check.py {files}"

    clean = run_lint(command, {"src/a.py": "value = 1\n"}, ["src/a.py"], target_root=repo)
    assert clean.ran and clean.ok, clean.to_dict()

    dirty = run_lint(command, {"src/a.py": "value = 1  # BAD\n"}, ["src/a.py"], target_root=repo)
    assert dirty.ran and not dirty.ok and dirty.exit_code == 1
    assert "src/a.py" in dirty.stdout, "the failing file has to be named"
    assert not (repo / "src").exists(), "the target repository itself is never written to"


def test_run_lint_refuses_to_guess_without_the_targets_configuration(tmp_path):
    """Linting the mapping alone would apply defaults the project never chose
    and report a verdict its CI would not recognise."""
    report = run_lint("{python} -c pass", {"src/a.py": "value = 1\n"}, ["src/a.py"], target_root=None)
    assert report.configured and not report.ran and not report.ok
    assert "no baseline root" in report.error


def test_lint_is_off_unless_it_is_configured(tmp_path):
    report = run_lint("", {"src/a.py": "value = 1\n"}, ["src/a.py"], target_root=tmp_path)
    assert not report.configured and not report.ran
    assert not report.ok, "an unconfigured gate is not a passing gate -- policy decides that"
