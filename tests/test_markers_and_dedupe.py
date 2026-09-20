# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Regression tests for the helpers whose subtleties cost real debugging time.

Each test here corresponds to a bug found while building the loop, and the
docstring says what went wrong without it.
"""

from __future__ import annotations

import pytest

from qedloop.core import (
    comment_index,
    decode_source,
    detect_markers,
    is_marker_line,
    marker_text,
    strip_marker,
)
from qedloop.phases.discover import _dedupe, _pick_target, issue_signature
from qedloop.sandbox import code_only, is_test_path, parse_failing_tests

# --------------------------------------------------------------------------- #
# declaring a defect
# --------------------------------------------------------------------------- #


def test_marker_name_survives_a_slash_and_a_dash():
    markers = detect_markers({"a.py": "# BUG: text/title-case_off_by_one -- the join corrupts it\n"})
    assert markers[0].name == "text/title-case_off_by_one"
    assert markers[0].note == "the join corrupts it"


def test_marker_may_be_parenthesised_or_bare():
    assert detect_markers({"a.py": "# FIXME(name): later\n"})[0].name == "name"
    assert detect_markers({"a.py": "# BUG: plain\n"})[0].name == "plain"
    assert detect_markers({"a.py": "# BUG -- prose only\n"})[0].name == ""


def test_todo_is_not_a_defect_marker():
    assert detect_markers({"a.py": "# TODO: tidy this up\n"}) == []


def test_marker_inside_a_string_is_not_a_declaration():
    """A marker-shaped string is documentation, not a claim about the code."""
    assert detect_markers({"a.py": 'msg = "# BUG: fake/thing"\n'}) == []
    line = 'msg = "# BUG: x"  # real'
    assert line[comment_index(line):] == "# real"


# --------------------------------------------------------------------------- #
# retracting a declaration
# --------------------------------------------------------------------------- #


def test_strip_keeps_code_when_the_marker_trails_it():
    """A patch can put the marker comment on the same line as real code.

    Removing the whole line then deletes the fix and corrupts the file; only the
    comment may go.
    """
    code = {"a.py": "if not values:\n    raise ValueError('x')  # BUG: stats/mean_zero -- must raise\n"}
    out = strip_marker(code, "stats/mean_zero")["a.py"]
    assert out == "if not values:\n    raise ValueError('x')\n"
    compile(out, "a.py", "exec")


def test_strip_removes_a_whole_line_block():
    code = {"a.py": "# BUG: demo/thing -- the guard is inverted\n# and the docstring lies about it.\nx = 1\n"}
    out = strip_marker(code, "demo/thing")["a.py"]
    assert out == "x = 1\n", "the continuation comment belongs to the declaration"


def test_strip_leaves_unrelated_comments_and_markers():
    code = {"a.py": "# BUG: demo/one -- first\n# BUG: demo/two -- second\nx = 1\n"}
    out = strip_marker(code, "demo/one")["a.py"]
    assert "# BUG: demo/two" in out
    assert "# BUG: demo/one" not in out


def test_strip_is_scoped_to_the_named_paths():
    code = {"a.py": "# BUG: demo/thing -- x\n", "b.py": "# BUG: demo/thing -- x\n"}
    out = strip_marker(code, "demo/thing", ["a.py"])
    assert "BUG" not in out["a.py"] and "BUG" in out["b.py"]


def test_strip_without_a_marker_is_a_no_op():
    code = {"a.py": "x = 1\n"}
    assert strip_marker(code, "") == code


def test_marker_text_matches_what_the_scanner_reads():
    code = {"a.py": "# BUG: demo/thing -- reason\nx = 1\n"}
    assert is_marker_line("# BUG: demo/thing -- reason\n", "demo/thing") is True
    assert marker_text("demo/thing").startswith("# BUG: demo/thing")


# --------------------------------------------------------------------------- #
# one defect, several reports
# --------------------------------------------------------------------------- #


def _report(marker, lens, severity="high", confidence=0.8, title=None):
    return {
        "id": "ISS-%s" % lens[:3],
        "title": title or ("declared defect %s" % marker),
        "bug_marker": marker,
        "severity": severity,
        "confidence": confidence,
        "lens": lens,
        "files": ["src/mod.py"],
    }


def test_signature_prefers_the_marker_over_the_wording():
    a = {"bug_marker": "demo/thing", "title": "one wording", "files": ["src/a.py"]}
    b = {"bug_marker": "demo/thing", "title": "completely different wording", "files": ["src/b.py"]}
    assert issue_signature(a) == issue_signature(b)


def test_signature_falls_back_to_title_and_file():
    a = {"bug_marker": "", "title": "Division  by zero", "files": ["src/a.py"]}
    b = {"bug_marker": "", "title": "division by zero", "files": ["src/a.py"]}
    assert issue_signature(a) == issue_signature(b)
    assert issue_signature(a) != issue_signature({"bug_marker": "", "title": "division by zero", "files": ["src/b.py"]})


def test_three_lenses_reporting_one_defect_become_one_row():
    reports = [_report("demo/thing", lens) for lens in ("archaeology", "behavior", "security")]
    reported, fresh = _dedupe(reports, [])
    assert len(reported) == 1 and len(fresh) == 1
    assert reported[0]["lenses"] == ["archaeology", "behavior", "security"]
    assert reported[0]["confidence"] > 0.8, "independent agreement is evidence"


def test_resighting_keeps_the_known_row_and_its_id():
    ledger = [{"id": "ISS-0001", "title": "known", "bug_marker": "demo/thing", "severity": "high", "confidence": 0.8, "lens": "archaeology", "files": ["src/mod.py"]}]
    reported, fresh = _dedupe([_report("demo/thing", "behavior")], ledger)
    assert fresh == [], "an existing issue is not a new row"
    assert reported[0]["id"] == "ISS-0001", "the loop must keep working on the row it owns"
    assert "behavior" in reported[0]["lenses"]


def test_strongest_severity_wins_across_lenses():
    reports = [_report("demo/thing", "a", severity="low"), _report("demo/thing", "b", severity="critical")]
    reported, _ = _dedupe(reports, [])
    assert reported[0]["severity"] == "critical"


def test_target_selection_skips_verified_and_ranks_by_severity():
    issues = [
        {"id": "ISS-0001", "severity": "low", "confidence": 0.9, "bug_marker": "a"},
        {"id": "ISS-0002", "severity": "critical", "confidence": 0.5, "bug_marker": "b"},
        {"id": "ISS-0003", "severity": "critical", "confidence": 0.9, "bug_marker": "c", "status": "verified"},
    ]
    state = {"issues": issues}
    assert _pick_target(issues, state)["id"] == "ISS-0002"


def test_target_selection_returns_none_when_everything_is_verified():
    issues = [{"id": "ISS-0001", "severity": "high", "status": "verified", "bug_marker": "a"}]
    assert _pick_target(issues, {"issues": issues}) is None


# --------------------------------------------------------------------------- #
# test files versus code files
# --------------------------------------------------------------------------- #


def test_test_paths_are_recognised():
    for path in ("tests/test_a.py", "src/tests/test_a.py", "conftest.py", "test_a.py", "a_test.py"):
        assert is_test_path(path) is True, path
    for path in ("src/tinylib/stats.py", "src/testing_helpers/not_a_test.py".replace("not_a_test.py", "util.py")):
        assert is_test_path(path) is False, path


def test_static_analysis_ignores_markers_quoted_in_tests():
    """A regression test may quote the defect it guards; that is not a defect."""
    code = {
        "src/mod.py": "x = 1\n",
        "tests/test_mod.py": "# BUG: demo/thing -- quoted in a test on purpose\n",
    }
    assert code_only(code) == {"src/mod.py": "x = 1\n"}


def test_parse_failing_tests_reads_quiet_and_verbose_output():
    quiet = "FAILED tests/test_a.py::test_one - AssertionError: nope\n1 failed, 2 passed in 0.1s"
    verbose = "____ test_two ____\n\n    assert 1 == 2\n\n____ test_three ____\n"
    assert parse_failing_tests(quiet) == ["tests/test_a.py::test_one"]
    assert parse_failing_tests(verbose) == ["test_two", "test_three"]
    assert parse_failing_tests("3 passed in 0.1s") == []


# --------------------------------------------------------------------------- #
# byte-order marks
# --------------------------------------------------------------------------- #


def test_utf8_bom_is_stripped_so_the_file_still_compiles():
    """PowerShell's ``-Encoding UTF8`` writes a BOM.

    Decoding that as plain utf-8 leaves ``\\ufeff`` at the head of the source,
    ``compile()`` refuses it, and every later check inherits the mistake -- a
    valid repository would look permanently broken.
    """
    raw = b"\xef\xbb\xbfdef f():\n    return 1\n"
    text = decode_source(raw)
    assert text is not None and not text.startswith("\ufeff")
    compile(text, "a.py", "exec")


def test_decode_source_tolerates_non_utf8_bytes():
    """A latin-1 source file must not be reported as unreadable."""
    assert decode_source(b"caf\xe9 = 1\n") == "caf\xe9 = 1\n"


def test_decode_source_never_raises_on_binary_noise():
    assert decode_source(b"\x00\x01\x02\xff\xfe") is None or isinstance(decode_source(b"\x00\x01\x02\xff\xfe"), str)


def test_a_bom_does_not_hide_a_marker():
    text = decode_source("\ufeffx = 1  # BUG: demo/thing -- x\n".encode("utf-8"))
    assert [m.name for m in detect_markers({"a.py": text})] == ["demo/thing"]
