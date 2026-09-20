# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Measured verification.

The QA phase is only meaningful if some of its evidence is *produced*, not
guessed.  This module runs the target's real test suite against the candidate
tree in a scratch directory, so "before" and "after" are measurements rather
than model claims.  The LLM agents then interpret that evidence (Phase 5) and
the gate combines both.

Everything degrades gracefully: a missing pytest, a timeout, or an unreadable
cwd yields a report with ``error`` set instead of raising into the graph.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .core import Marker, decode_source, detect_markers, render_markers, sha_bytes

#: pytest's terminal summary -- ``4 failed, 4 passed, 1 skipped in 0.13s``.
#: The gap between count and word is horizontal whitespace only, never ``\s``:
#: a newline in that gap let an assertion message ending in a number swallow the
#: following ``FAILED ...`` line, and ``assert 0 == 5`` plus ``FAILED ...`` read
#: as "5 failed".  On the demo fixture that turned a real "4 failed" into 9.
PYTEST_SUMMARY_RE = re.compile(
    r"(?P<count>\d+)[ \t]+(?P<kind>passed|failed|error|errors|skipped|xfailed|xpassed|warning|warnings)",
    re.I,
)
#: ``FAILED tests/test_mod.py::test_name - AssertionError: ...`` and the
#: ``_____ test_name _____`` section headers pytest prints above the traceback.
PYTEST_FAILURE_RE = re.compile(r"^(?:FAILED|ERROR)\s+(?P<nodeid>\S+)", re.M)
PYTEST_SECTION_RE = re.compile(r"^_{3,}\s+(?P<name>[^_\s].*?)\s+_{3,}$", re.M)
SYNTAX_ERROR_RE = re.compile(r'File "(?P<path>[^"]+)", line (?P<line>\d+)')

#: A newline that is not already part of a CRLF pair.  Used to decide whether a
#: search anchor can be retried with Windows line endings.
LONE_LF_RE = re.compile(r"(?<!\r)\n")

#: Words a model uses when it means "this file is new".  Checked because an empty
#: ``search`` is also what a truncated reply looks like; requiring the intent to
#: be stated keeps a half-written op from being read as a whole-file write.
CREATION_HINTS = ("create", "new file", "new test", "新增", "新建", "新文件", "创建")


def _creation_rejection(path: str, replace: str, rationale: str, candidate: Mapping[str, str]) -> str:
    """Why this empty-``search`` op may not be applied as a file creation."""
    if not replace.strip():
        return "path and search are both required (an empty search means 'create this file', which needs content)"
    lowered = rationale.lower()
    if not any(hint in lowered for hint in CREATION_HINTS):
        return (
            "search is empty and the rationale does not say the file is new; to replace existing text, "
            "search must be copied verbatim from the file, and to create a file say so in the rationale"
        )
    if path in candidate:
        return "%s already exists, so it cannot be created; use an anchored search instead" % path
    if not path.endswith(".py"):
        return "only python files may be created (%s)" % path
    return ""

SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "env", "node_modules", ".idea", ".vscode", "dist", "build", "runs",
    ".tox", ".eggs", ".tmp", ".claude", "logs", "exports", "release", "output",
}
DEFAULT_INCLUDE = ("*.py",)


@dataclass
class SandboxReport:
    workspace: str = ""
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration_s: float = 0.0
    ok: bool = False
    stdout: str = ""
    stderr: str = ""
    command: List[str] = field(default_factory=list)
    error: str = ""
    files_written: int = 0

    @property
    def ran(self) -> bool:
        return not self.error and bool(self.command)

    @property
    def pass_rate(self) -> float:
        total = self.passed + self.failed + self.errors
        return (self.passed / total) if total else 0.0

    @property
    def green(self) -> bool:
        return self.ran and self.failed == 0 and self.errors == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace": self.workspace,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "duration_s": round(self.duration_s, 2),
            "ok": self.ok,
            "green": self.green,
            "pass_rate": round(self.pass_rate, 3),
            "ran": self.ran,
            "error": self.error,
            "command": self.command,
            "stdout_tail": _tail(self.stdout, 1600),
            "stderr_tail": _tail(self.stderr, 800),
        }


def _tail(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else "...\n" + text[-limit:]


@dataclass
class LintReport:
    """The target repository's own static check, run against the candidate tree.

    The loop already proves that the tree compiles and that the suite passes, and
    neither can see a style or typing rule.  A patch can therefore be green here
    and red in the target's CI.  Measured on EER-Ai: the crew produced a patch
    whose only defect was ``TC003`` (a standard-library import that ruff wants in
    a ``TYPE_CHECKING`` block) -- 507 tests green, lint red.

    A gate that was configured and could not be evaluated must not read as
    satisfied, or a typo in the command would silently disable the check.
    """

    configured: bool = False
    command: List[str] = field(default_factory=list)
    exit_code: Optional[int] = None
    duration_s: float = 0.0
    stdout: str = ""
    stderr: str = ""
    error: str = ""

    @property
    def ran(self) -> bool:
        return self.configured and self.exit_code is not None

    @property
    def ok(self) -> bool:
        return self.ran and self.exit_code == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "configured": self.configured,
            "ran": self.ran,
            "ok": self.ok,
            "command": self.command,
            "exit_code": self.exit_code,
            "duration_s": round(self.duration_s, 2),
            "error": self.error,
            "stdout_tail": _tail(self.stdout, 1600),
            "stderr_tail": _tail(self.stderr, 800),
        }


def parse_pytest_summary(text: str) -> Dict[str, int]:
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for match in PYTEST_SUMMARY_RE.finditer(text or ""):
        kind = match.group("kind").lower()
        value = int(match.group("count"))
        if kind.startswith("pass"):
            counts["passed"] += value
        elif kind.startswith("fail"):
            counts["failed"] += value
        elif kind.startswith("err"):
            counts["errors"] += value
        elif kind.startswith("skip"):
            counts["skipped"] += value
    return counts


def parse_failing_tests(text: str) -> List[str]:
    """Extract the test node ids (or names) that failed or errored.

    Tolerates both pytest's ``-q`` summary lines and the ``_____ name _____``
    section headers of verbose output, because the same evidence is read
    regardless of the reporter flags used for the run.
    """
    found: List[str] = []
    for match in PYTEST_FAILURE_RE.finditer(text or ""):
        nodeid = match.group("nodeid").strip()
        if nodeid and nodeid not in found:
            found.append(nodeid)
    if not found:
        for match in PYTEST_SECTION_RE.finditer(text or ""):
            name = match.group("name").strip()
            if name and name not in found and not name.lower().startswith(("warnings", "short test summary")):
                found.append(name)
    return found


#: Canonical test-name shape for a declared defect: ``domain/name`` becomes
#: ``test_domain_name``, so a failing node id can be matched back to the issue
#: that predicted it.  Documented in docs/phases.md and asserted in the specs.
TEST_NAME_TEMPLATE = "test_{marker}"


def test_name_for_marker(marker: str) -> str:
    return TEST_NAME_TEMPLATE.format(marker=re.sub(r"[^A-Za-z0-9]+", "_", str(marker or "")).strip("_").lower())


def failing_tests_for_marker(marker: str, node_ids: Sequence[str]) -> List[str]:
    """Failing test ids that this defect marker predicted."""
    wanted = test_name_for_marker(marker)
    if not wanted or wanted == "test_":
        return []
    out = []
    for node_id in node_ids or []:
        leaf = str(node_id).split("::")[-1].strip().lower()
        if leaf.startswith(wanted) or wanted in leaf:
            out.append(str(node_id))
    return out


def pytest_available() -> bool:
    return importlib.util.find_spec("pytest") is not None


def materialise(code: Mapping[str, str], root: Path) -> int:
    """Write a candidate tree to disk.  Returns the number of files written."""
    written = 0
    for rel, body in code.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        # ``newline=""`` is load-bearing, not cosmetic.  ``write_text`` defaults
        # to translating every "\n" to the platform separator, and a candidate
        # body already carries the *target's* convention (CRLF on a CRLF
        # checkout), so every "\r\n" came back as "\r\r\n" -- which reads as two
        # line breaks and inserts a blank line between every line of every
        # edited file.  Measured on a real run: the applied patch was 74
        # insertions / 5 deletions where the model wrote roughly 30 / 6, and no
        # test noticed because Python does not care about blank lines.
        target.write_text(body, encoding="utf-8", newline="")
        written += 1
    return written


def _sandbox_env(root: Path) -> Dict[str, str]:
    """The environment every command runs in inside a candidate sandbox.

    Shared by the measured suite and the lint gate on purpose: two environments
    would mean two answers to the same question, and the target's rules have to
    be the ones its CI would apply.  ``{python}`` in a ``lint_command`` resolves
    to this same interpreter, so a gate can never quietly build a second one --
    the sandbox copy excludes ``.venv`` and ``node_modules``, so a command like
    ``uv run`` would reconstruct an environment per cycle: slow, network-bound,
    and not the environment the suite was measured in.
    """
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    # The measured suite gets its own pytest temp root inside the scratch tree.
    # Two pytest processes sharing the user-level ``pytest-of-<user>`` directory
    # take turns replacing its ``pytest-current`` symlink, and the process that
    # finishes second then fails in its atexit cleanup -- which is exactly the
    # normal case here, because a run is usually started from a pytest session
    # (the framework's own tests, or another loop run).
    pytest_tmp = root / ".pytest-tmp"
    pytest_tmp.mkdir(parents=True, exist_ok=True)
    env["PYTEST_DEBUG_TEMPROOT"] = str(pytest_tmp)
    # The tree under test may shadow a dependency name; it must win.  That
    # includes the project's *own* package under a ``src/`` layout: the package
    # is not at the repository root, so without ``<root>/src`` the import falls
    # through to whatever the ambient environment already provides -- in
    # practice an editable install or a ``.pth`` pointing at the *original*
    # checkout.  The candidate overlay would then never be imported, and every
    # patch would measure as the unmodified repository: a number that cannot
    # fail, which is worse than no measurement at all.
    path_entries = [str(root)]
    source_root = root / "src"
    if source_root.is_dir():
        path_entries.append(str(source_root))
    existing = env.get("PYTHONPATH", "")
    if existing:
        path_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(path_entries)
    return env


def run_pytest(
    code: Mapping[str, str],
    *,
    workspace: Optional[str] = None,
    target_root: Optional[str] = None,
    timeout: float = 120.0,
    keep: bool = False,
    extra_args: Sequence[str] = ("-p", "no:cacheprovider", "-q"),
) -> SandboxReport:
    """Run the target's suite against a candidate tree.

    When ``target_root`` is given, the sandbox is a **copy of the repository**
    (minus caches, VCS metadata and virtualenvs) with the candidate files
    overlaid on top.  Copying the tree rather than only the supplied files is
    what makes the measurement trustworthy: test suites routinely depend on
    non-Python assets -- images, templates, JSON fixtures, frontend bundles --
    and a sandbox that drops them reports failures that do not exist in the real
    repository.  The copy also gives pytest the project's own configuration.

    Without ``target_root`` the supplied mapping is materialised on its own,
    which is enough for self-contained suites and for the framework's tests.
    """
    report = SandboxReport()
    if not pytest_available():
        report.error = "pytest is not importable in this interpreter (%s)" % sys.executable
        return report

    temp = None
    if workspace:
        root = Path(workspace)
        if root.exists() and any(root.iterdir()):
            report.error = "workspace %s is not empty; refusing to run" % root
            return report
        root.mkdir(parents=True, exist_ok=True)
    else:
        temp = tempfile.TemporaryDirectory(prefix="qedloop-sbx-")
        root = Path(temp.name)

    report.workspace = str(root)
    if target_root:
        copy_repository(target_root, root)

    report.files_written = materialise(code, root)

    # The check is on the *sandbox*, not on the supplied mapping: a candidate
    # tree may legitimately carry only the edited modules, with the tests coming
    # from the repository copy beneath it.
    if not any(root.rglob("test_*.py")) and not any(root.rglob("*_test.py")):
        report.error = "no test files in the sandbox; nothing to run"
        if temp is not None and not keep:
            temp.cleanup()
            report.workspace = ""
        return report

    report.command = [sys.executable, "-m", "pytest", *extra_args]

    env = _sandbox_env(root)

    started = time.time()
    try:
        completed = subprocess.run(
            report.command,
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        report.stdout = completed.stdout or ""
        report.stderr = completed.stderr or ""
        report.ok = completed.returncode == 0
    except subprocess.TimeoutExpired:
        report.error = "pytest timed out after %.0fs" % timeout
    except OSError as exc:
        report.error = "could not start pytest: %s" % exc
    report.duration_s = time.time() - started

    combined = (report.stdout or "") + "\n" + (report.stderr or "")
    counts = parse_pytest_summary(combined)
    report.passed, report.failed, report.errors, report.skipped = (
        counts["passed"], counts["failed"], counts["errors"], counts["skipped"],
    )
    report.total = report.passed + report.failed + report.errors + report.skipped
    report.ok = report.ok or (report.total > 0 and report.failed == 0 and report.errors == 0)

    if temp is not None and not keep:
        temp.cleanup()
        report.workspace = ""
    return report


def lint_argv(template: str, files: Sequence[str]) -> List[str]:
    """Split a ``lint_command`` template into an argv, expanding placeholders.

    Substitution is per whitespace-separated token and ``{files}`` must be a
    token of its own, so one changed file becomes exactly one argument.  That
    keeps the command shell-free: no quoting layer, which is where Windows and
    POSIX would otherwise disagree.
    """
    argv: List[str] = []
    for token in (template or "").split():
        if token == "{python}":
            argv.append(sys.executable)
        elif token == "{files}":
            argv.extend(str(path) for path in files)
        else:
            argv.append(token.replace("{python}", sys.executable))
    return argv


def run_lint(
    command: str,
    code: Mapping[str, str],
    files: Sequence[str],
    *,
    target_root: str | Path | None = None,
    timeout: float = 120.0,
    keep: bool = False,
) -> LintReport:
    """Run the target's own static check against the candidate tree.

    The command is **read-only by contract**: it must not rewrite files.  A gate
    that edits the tree it is grading would break the promise that ``apply``
    writes the bytes Phase 5 measured, and a ``--fix``-style check reports
    failure *and* repairs, so the verdict would be about a tree that no longer
    exists.

    ``files`` is what ``{files}`` expands to -- the files this cycle changed, not
    every file in the tree.  Scoping the check to the patch is what keeps its
    failures attributable: a whole-repository check would also fail on violations
    the loop is not allowed to touch, which is a blocker it can never clear.
    """
    template = (command or "").strip()
    if not template:
        return LintReport()
    if not target_root:
        # The rules live in the repository's own configuration file, so linting
        # the supplied mapping alone would apply defaults the project never chose
        # and report a verdict its CI would not recognise.
        return LintReport(configured=True, error="no baseline root to lint against")

    report = LintReport(configured=True, command=lint_argv(template, files))
    temp = tempfile.TemporaryDirectory(prefix="qedloop-lint-")
    root = Path(temp.name)
    started = time.time()
    try:
        copy_repository(target_root, root)
        materialise(code, root)
        completed = subprocess.run(
            report.command,
            cwd=str(root),
            env=_sandbox_env(root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        report.stdout = completed.stdout or ""
        report.stderr = completed.stderr or ""
        report.exit_code = completed.returncode
    except subprocess.TimeoutExpired:
        report.error = "lint timed out after %.0fs" % timeout
    except OSError as exc:
        report.error = "could not start the lint command: %s" % exc
    report.duration_s = time.time() - started
    if not keep:
        temp.cleanup()
    return report


def copy_repository(source: str | Path, destination: Path, limit: int = 20000) -> int:
    """Copy a repository into a scratch tree, skipping caches and env dirs."""
    src = Path(source)
    copied = 0
    for path in sorted(src.rglob("*")):
        if copied >= limit:
            break
        relative = path.relative_to(src)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        destination_path = destination / relative
        try:
            if path.is_dir():
                destination_path.mkdir(parents=True, exist_ok=True)
            elif path.is_file():
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination_path)
                copied += 1
        except OSError:
            continue  # unreadable file: not worth failing the run over
    return copied


def coverable_files(code: Mapping[str, str]) -> List[str]:
    return [p for p in code if is_test_path(p)]


#: Paths that are test code.  Test files are executed by the sandbox but are
#: excluded from *static* analysis: a test file may legitimately quote a defect
#: marker (that is how a regression test documents what it guards), and counting
#: those would make every run report phantom defects.
TEST_PATH_RE = re.compile(r"(^|/)(tests?|testing)/|(^|/)(conftest|test_[^/]*|[^/]*_test)\.py$")


def is_test_path(path: str) -> bool:
    return bool(TEST_PATH_RE.search(str(path).replace("\\", "/")))


def code_only(code: Mapping[str, str]) -> Dict[str, str]:
    return {path: body for path, body in code.items() if not is_test_path(path)}


# --------------------------------------------------------------------------- #
# static analysis (cheap, deterministic, no external deps)
# --------------------------------------------------------------------------- #


def static_analysis(before: Mapping[str, str], after: Mapping[str, str]) -> Dict[str, Any]:
    """Compile-check the candidate tree and diff declared defect markers.

    Test files are excluded from both halves of this comparison (see
    :data:`TEST_PATH_RE`); they are still compiled by pytest when the suite runs.
    """
    before_code = code_only(before)
    after_code = code_only(after)

    syntax_errors: List[Dict[str, Any]] = []
    compiled = 0
    for path in sorted(after_code):
        if not path.endswith(".py"):
            continue
        try:
            compile(after_code[path], path, "exec")
            compiled += 1
        except SyntaxError as exc:
            syntax_errors.append({"path": path, "line": exc.lineno or 0, "message": exc.msg or "syntax error"})

    before_markers = detect_markers(before_code)
    after_markers = detect_markers(after_code)
    before_keys = {m.key: m for m in before_markers}
    after_keys = {m.key: m for m in after_markers}
    resolved = [before_keys[k].to_dict() for k in sorted(before_keys) if k not in after_keys]
    introduced = [after_keys[k].to_dict() for k in sorted(after_keys) if k not in before_keys]

    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    return {
        "compiled_files": compiled,
        "syntax_errors": syntax_errors,
        "compiles": not syntax_errors,
        "markers_before": len(before_markers),
        "markers_after": len(after_markers),
        "markers_resolved": resolved,
        "markers_introduced": introduced,
        "changed_files": changed,
        "tree_sha": sha_bytes("".join(sorted(after.values())).encode("utf-8")),
    }


def verification_snapshot(
    before: Mapping[str, str],
    after: Mapping[str, str],
    *,
    baseline_root: str | Path | None = None,
    run_tests: bool = True,
    timeout: float = 120.0,
    keep_workspaces: bool = False,
    lint_command: str = "",
) -> Dict[str, Any]:
    """Everything Phase 5 needs: regression delta plus static facts.

    ``baseline_root`` is the repository the candidate came from.  The baseline
    run happens *inside a copy of that repository*, so "before" means the same
    thing the project's own CI means by it; the candidate run overlays the
    edited files on the same copy.  Both runs therefore see the same assets and
    configuration, and their difference is attributable to the patch.
    """
    evidence: Dict[str, Any] = {"static": static_analysis(before, after)}
    if not run_tests:
        evidence["tests"] = {"ran": False, "error": "test execution disabled by configuration"}
        return evidence

    root = str(baseline_root) if baseline_root else None
    before_report = run_pytest(before, target_root=root, timeout=timeout, keep=keep_workspaces)
    after_report = run_pytest(after, target_root=root, timeout=timeout, keep=keep_workspaces)
    before_failed = parse_failing_tests(before_report.stdout + "\n" + before_report.stderr)
    after_failed = parse_failing_tests(after_report.stdout + "\n" + after_report.stderr)
    evidence["tests"] = {
        "ran": after_report.ran,
        "baseline_root": root,
        "before": before_report.to_dict(),
        "after": after_report.to_dict(),
        "failing_before": before_failed,
        "failing_after": after_failed,
        "repaired_tests": sorted(set(before_failed) - set(after_failed)),
        "regression": {
            "new_failures": max(0, after_report.failed + after_report.errors - (before_report.failed + before_report.errors)),
            "fixed": max(0, (before_report.failed + before_report.errors) - (after_report.failed + after_report.errors)),
            "before_green": before_report.green,
            "after_green": after_report.green,
            "delta_passed": after_report.passed - before_report.passed,
        },
    }
    # Part of the measured path, so it runs whenever the suite does: both gates
    # answer "is this candidate tree acceptable", and a run that disabled the
    # suite cannot converge anyway (``require_tests``).
    if lint_command:
        changed = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
        evidence["lint"] = run_lint(
            lint_command,
            after,
            changed,
            target_root=root,
            timeout=timeout,
            keep=keep_workspaces,
        ).to_dict()
    return evidence


# --------------------------------------------------------------------------- #
# applying a patch to a candidate tree
# --------------------------------------------------------------------------- #


@dataclass
class ApplyResult:
    ok: bool
    code: Dict[str, str] = field(default_factory=dict)
    applied: List[Dict[str, Any]] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "applied": self.applied,
            "skipped": self.skipped,
            "errors": self.errors,
            "strategy": "anchored-search-replace",
        }


def _anchor_forms(search: str) -> List[str]:
    """The anchor as written, plus its Windows-line-ending twin.

    Line endings are content to a substring match and nothing to a reader: on a
    repository with ``core.autocrlf=true`` every checked-out file is CRLF, while
    a model asked to copy a block verbatim frequently returns LF.  Measured on a
    real run: the same three anchors missed with a CRLF body and all three
    applied with an LF body, so a patch that was correct in every visible way
    failed as "anchor not found" three times over.

    The CRLF twin is only derived from an anchor that has no CR at all, so an
    anchor already using Windows endings cannot be turned into ``\\r\\r\\n``.
    """
    if "\r" in search:
        return [search]
    twin = LONE_LF_RE.sub("\r\n", search)
    return [search] if twin == search else [search, twin]


def apply_ops(code: Mapping[str, str], ops: Sequence[Mapping[str, Any]]) -> ApplyResult:
    """Apply anchored search/replace ops; all-or-nothing per proposal.

    Ambiguity (a non-unique anchor) and absence (anchor not found) are hard
    failures -- they are the two ways a model-generated patch silently corrupts
    a repository, so they never degrade into a partial apply.  Line-ending
    variants are tried only to *locate* the anchor: the replacement is applied
    to whichever form was found, so the file keeps the endings it already had,
    and two forms matching at once is ambiguity rather than a coin toss.

    An op with **no ``search`` and a ``replace``** means "create this file".  The
    contract had no way to say that, so a patch that added a regression test had
    to send an empty ``search`` and was rejected as malformed: measured on a real
    run, the agent's own rationale read "新文件无既有内容，故 search 为空" ("the new
    file has no existing content, so search is empty") and the whole proposal --
    a good one -- was thrown away.  Creation is deliberately narrow: the intent
    must be explicit in the op's rationale, and only ``.py`` files may be created,
    because a bare "replace everything" op is how a patch would overwrite the
    repository by accident.
    """
    candidate = dict(code)
    applied: List[Dict[str, Any]] = []
    errors: List[str] = []
    for index, op in enumerate(ops or []):
        path = str(op.get("path", "")).replace("\\", "/")
        search = str(op.get("search", ""))
        replace = str(op.get("replace", ""))
        rationale = str(op.get("rationale", ""))
        if not path:
            errors.append("op %d: path is required" % index)
            continue
        if not search:
            reason = _creation_rejection(path, replace, rationale, candidate)
            if reason:
                errors.append("op %d: %s" % (index, reason))
                continue
            candidate[path] = replace
            applied.append({"path": path, "rationale": rationale, "bytes": len(replace), "created": True})
            continue
        if path not in candidate:
            errors.append("op %d: %s is not part of the candidate tree" % (index, path))
            continue

        body = candidate[path]
        forms = [(form, body.count(form)) for form in _anchor_forms(search)]
        matched = [(form, count) for form, count in forms if count]
        ambiguous = [form for form, count in matched if count > 1]
        if ambiguous:
            errors.append(
                "op %d: anchor is not unique in %s (%d matches)"
                % (index, path, max(count for _, count in matched))
            )
            continue
        if not matched:
            errors.append("op %d: anchor not found in %s" % (index, path))
            continue

        anchor = matched[0][0]
        # Express the replacement in whatever convention the file already uses.
        # A model copying a CRLF block returns LF about as often as CRLF, and
        # splicing those lines in unchanged leaves the file half CRLF and half
        # LF: valid to Python, but a patch no reviewer would call clean.  This
        # normalises unconditionally rather than only when the replacement looks
        # LF-only -- a replacement can arrive with mixed endings, and testing for
        # "no CRLF anywhere" would then skip the lines that needed it.
        if "\r\n" in anchor:
            replace = LONE_LF_RE.sub("\r\n", replace.replace("\r\n", "\n"))
        candidate[path] = body.replace(anchor, replace, 1)
        applied.append({"path": path, "rationale": op.get("rationale", ""), "bytes": len(replace) - len(search)})
    ok = bool(applied) and not errors
    return ApplyResult(ok=ok, code=candidate if ok else dict(code), applied=applied, errors=errors)


def seed_from_directory(root: str | Path, patterns: Sequence[str] = DEFAULT_INCLUDE) -> Dict[str, str]:
    """Read a repo into ``{relative_path: content}`` (used for smoke runs)."""
    base = Path(root)
    if not base.is_dir():
        raise NotADirectoryError(str(root))
    out: Dict[str, str] = {}
    for pattern in patterns:
        for path in sorted(base.rglob(pattern)):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            rel = path.relative_to(base).as_posix()
            try:
                text = decode_source(path.read_bytes())
            except OSError:
                continue
            if text is not None:
                out[rel] = text
    return out
