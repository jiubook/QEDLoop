# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Typed records plus the ledger reducers used by the state bus.

Everything that crosses a state-bus channel is a plain dict, so the whole run
state stays JSON serialisable and traceable.  Dataclasses exist for typing and
normalisation at the edges (LLM output -> state), never as the on-bus format.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeVar
# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

T = TypeVar("T")

ID_PREFIXES = {
    "issue": "ISS",
    "finding": "FND",
    "plan": "PLN",
    "review": "REV",
    "patch": "PAT",
    "change": "CHG",
    "check": "CHK",
    "cycle": "CYC",
    "run": "RUN",
}

_counters: Dict[str, int] = {}


def new_id(kind: str) -> str:
    """Deterministic-per-process identifier: ``ISS-0001``, ``REV-0007`` ..."""
    prefix = ID_PREFIXES.get(kind, kind[:3].upper())
    n = _counters.get(prefix, 0) + 1
    _counters[prefix] = n
    return "%s-%04d" % (prefix, n)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def decode_source(data: bytes) -> Optional[str]:
    """Decode a source file, tolerating a BOM and mixed encodings.

    Windows editors (and PowerShell's ``-Encoding UTF8``) happily write a UTF-8
    BOM.  Decoding that as plain ``utf-8`` leaves ``\\ufeff`` at the head of the
    string, which makes ``compile()`` fail -- so a perfectly good file looks
    broken and every later check inherits the mistake.  ``utf-8-sig`` strips it.
    """
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        return text.lstrip("\ufeff")
    return None


def to_lf(text: str) -> str:
    """The LF form of a text, for newline-insensitive comparisons.

    Lives here rather than with the run-directory helpers because the report
    writer needs it too, and the report cannot import the orchestrator (the
    orchestrator imports the report).
    """
    return text.replace("\r\n", "\n")


def to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError("cannot convert %r to dict" % (type(obj),))


def to_dicts(items: Iterable[Any]) -> List[Dict[str, Any]]:
    return [to_dict(i) for i in (items or [])]


def clip(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    half = max(1, limit // 2 - 20)
    return text[:half] + "\n... [%d chars elided] ...\n" % (len(text) - 2 * half) + text[-half:]


def tally(items: Sequence[Mapping[str, Any]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in items or []:
        value = str(item.get(key, "unknown"))
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


#: Words a model reaches for when the prompt asked for a number.  Read as the
#: level they name rather than discarded: "low" is real information about the
#: agent's own confidence.
CONFIDENCE_WORDS = (
    ("very low", 0.10),
    ("very high", 0.95),
    ("critical", 0.05),
    ("minimal", 0.15),
    ("low", 0.25),
    ("medium", 0.50),
    ("moderate", 0.50),
    ("high", 0.80),
)


def as_float(value: Any, default: float = 0.0) -> float:
    """Coerce model-authored numbers; never raise.

    Agents answer with the number the contract asked for *most* of the time.
    The rest of the time they answer with the sentence that was in their head --
    ``"low - no candidate patches were present in the context"`` -- and a
    ``ValueError`` there used to abort the whole run *after* the tokens were
    spent, discarding every finding the run had produced.  A cosmetic
    disagreement must not cost a run.
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        pass
    match = re.match(r"[-+]?\d*\.?\d+", text)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            pass
    for word, number in CONFIDENCE_WORDS:
        if text.startswith(word):
            return number
    return default


def metrics_of(entries: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Any]:
    """Aggregate a numeric field across ledger rows (tokens, latency, ...)."""
    values = [as_float(e.get(key)) for e in entries or []]
    if not values:
        return {"count": 0, "sum": 0.0, "max": 0.0}
    return {"count": len(values), "sum": round(sum(values), 2), "max": round(max(values), 2)}


def recap(items: Sequence[Mapping[str, Any]], keys: Sequence[str], limit: int = 40) -> str:
    """One-line-per-item compression used when injecting ledgers into prompts."""
    lines: List[str] = []
    for item in list(items)[:limit]:
        parts = [str(item.get(k, "")) for k in keys if item.get(k) not in (None, "")]
        lines.append("- " + " | ".join(parts))
    if not items:
        return "  (empty)"
    if len(items) > limit:
        lines.append("- ... %d more elided" % (len(items) - limit))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# reducers -- the heart of the state bus
# --------------------------------------------------------------------------- #


class Replace:
    """Marker for *whole-value* channels.

    A node returning ``{"working_code": Replace(new_map)}`` overwrites the
    channel; a node returning ``{"changes": [patch]}`` appends through a reducer.
    """

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return "Replace(%r)" % (self.value,)


def _normalise(kind: str, left: List[Any], incoming: Any) -> List[Dict[str, Any]]:
    """Merge ``incoming`` into ``left`` for ledger ``kind`` (see state.CHANNELS)."""
    if incoming is None:
        return list(left)
    if isinstance(incoming, Mapping):
        incoming = [incoming]
    incoming = list(incoming)
    if not incoming:
        return list(left)

    out = [to_dict(i) for i in left]
    index = {str(i.get("id", "")): pos for pos, i in enumerate(out) if i.get("id")}

    if kind == "issues":
        # Counters are recomputed from the stored row, never taken from the
        # incoming delta: a delta that omits "seen_count" must not reset it.
        for item in incoming:
            rec = to_dict(item)
            rec.setdefault("id", new_id("issue"))
            rec.setdefault("cycle", 0)
            pos = index.get(rec["id"])
            previous = out[pos] if pos is not None else {}
            rec["first_seen_cycle"] = previous.get("first_seen_cycle", rec.get("first_seen_cycle", rec["cycle"]))
            rec["seen_count"] = int(previous.get("seen_count") or 0) + 1
            rec["last_seen_cycle"] = int(rec.get("cycle") or previous.get("last_seen_cycle") or 0)
            rec.setdefault("status", previous.get("status", "open") or "open")
            if pos is None:
                index[rec["id"]] = len(out)
                out.append(rec)
            else:
                merged = dict(out[pos])
                merged.update({k: v for k, v in rec.items() if v not in (None, "")})
                out[pos] = merged

    elif kind == "findings":
        # keep the latest refinement per issue, remember earlier attempts
        for item in incoming:
            rec = to_dict(item)
            rec.setdefault("id", new_id("finding"))
            pos = index.get(rec["id"])
            if pos is None:
                index[rec["id"]] = len(out)
                out.append(rec)
            else:
                prev = dict(out[pos])
                rec = dict(rec)
                rec["superseded"] = list(prev.get("superseded", [])) + [
                    {k: prev.get(k) for k in ("version", "refined_at", "decision", "summary")}
                ]
                rec["version"] = int(prev.get("version", 1)) + 1
                out[pos] = rec

    elif kind == "plans":
        # A refinement round is a whole plan, not a delta on the previous one.
        # Plans arrive with fresh ids every round, so the append branch below
        # would keep every generation alive at once: measured on a real run, a
        # reviewer was shown five plan rows of which three named the same test
        # function, and it rejected the round -- correctly -- for ambiguity that
        # only existed because superseded generations were still being served.
        # Retiring is by generation stamp (``refined_at``) rather than by id, and
        # the retired rows stay in the ledger carrying ``superseded_at`` so the
        # audit trail and the replay stay intact.
        for item in incoming:
            rec = to_dict(item)
            rec.setdefault("id", new_id("plan"))
            stamp = str(rec.get("refined_at") or "")
            if stamp:
                for existing in out:
                    same_issue = str(existing.get("issue_id") or "") == str(rec.get("issue_id") or "")
                    previous = str(existing.get("refined_at") or "")
                    if same_issue and previous and previous != stamp and not existing.get("superseded_at"):
                        existing["superseded_at"] = stamp
            pos = index.get(rec["id"])
            if pos is None:
                index[rec["id"]] = len(out)
                out.append(rec)
            else:
                merged = dict(out[pos])
                merged.update(rec)
                out[pos] = merged

    else:  # reviews / patches / changes / checks / cycles: append-only ledger
        for item in incoming:
            out.append(to_dict(item))

    return out


def merge_ledger(kind: str) -> Callable[[List[Any], Any], List[Dict[str, Any]]]:
    def _reducer(left: List[Any], incoming: Any) -> List[Dict[str, Any]]:
        return _normalise(kind, list(left or []), incoming)

    _reducer.__name__ = "merge_%s" % kind
    _reducer.ledger_kind = kind  # type: ignore[attr-defined]
    return _reducer


def merge_dict(left: Optional[Mapping[str, Any]], incoming: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Shallow dict union, later writes win. Used for evidence / metrics."""
    return {**(left or {}), **(incoming or {})}


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #


@dataclass
class FileRecord:
    path: str
    content: str
    language: str = "python"
    lines: int = 0
    sha: str = ""

    def __post_init__(self) -> None:
        if not self.lines:
            self.lines = self.content.count("\n") + 1
        if not self.sha:
            self.sha = sha_bytes(self.content.encode("utf-8"))

    @property
    def is_test(self) -> bool:
        p = self.path.replace("\\", "/")
        name = p.rsplit("/", 1)[-1]
        return name.startswith("test_") or name.endswith("_test.py") or "/tests/" in p or p.startswith("tests/")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "content": self.content,
            "language": self.language,
            "lines": self.lines,
            "sha": self.sha,
            "is_test": self.is_test,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FileRecord":
        return cls(
            path=str(data["path"]),
            content=str(data.get("content", "")),
            language=str(data.get("language", "python")),
            lines=int(data.get("lines") or 0),
            sha=str(data.get("sha") or ""),
        )


@dataclass
class Codebase:
    root: str
    files: List[FileRecord] = field(default_factory=list)
    revision: str = ""
    manifest: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.revision:
            joined = "".join(sorted(f"{f.path}:{f.sha}\n" for f in self.files))
            self.revision = sha_bytes(joined.encode("utf-8"))
        if not self.manifest:
            self.manifest = {f.path: f.sha for f in sorted(self.files, key=lambda x: x.path)}

    @property
    def code_files(self) -> List[FileRecord]:
        return [f for f in self.files if not f.is_test]

    @property
    def test_files(self) -> List[FileRecord]:
        return [f for f in self.files if f.is_test]

    def as_map(self) -> Dict[str, str]:
        return {f.path: f.content for f in self.files}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": self.root,
            "revision": self.revision,
            "manifest": dict(self.manifest),
            "files": [f.to_dict() for f in self.files],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Codebase":
        return cls(
            root=str(data.get("root", "")),
            files=[FileRecord.from_dict(f) for f in data.get("files", [])],
            revision=str(data.get("revision", "")),
            manifest=dict(data.get("manifest", {})),
        )

    def summary(self, limit: int = 24) -> str:
        lines = [
            "root=%s revision=%s files=%d (code=%d, test=%d)"
            % (self.root, self.revision, len(self.files), len(self.code_files), len(self.test_files))
        ]
        for f in sorted(self.files, key=lambda x: x.path)[:limit]:
            lines.append("  %-42s %5d lines  %s" % (f.path, f.lines, "test" if f.is_test else "src"))
        if len(self.files) > limit:
            lines.append("  ... %d more" % (len(self.files) - limit))
        return "\n".join(lines)


@dataclass
class Issue:
    id: str = ""
    title: str = ""
    phase_hint: str = "unknown"
    lens: str = "unknown"
    severity: str = "medium"
    confidence: float = 0.5
    files: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    evidence: str = ""
    why_it_matters: str = ""
    proposed_direction: str = ""
    cycle: int = 0
    status: str = "open"
    seen_count: int = 0
    bug_marker: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Issue":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload = {k: v for k, v in dict(data).items() if k in known}
        payload.setdefault("files", [])
        payload.setdefault("symbols", [])
        issue = cls(**payload)
        if not issue.id:
            issue.id = new_id("issue")
        if not isinstance(issue.files, list):
            issue.files = [str(issue.files)]
        if not isinstance(issue.symbols, list):
            issue.symbols = [str(issue.symbols)]
        issue.confidence = as_float(issue.confidence, 0.5)
        issue.severity = str(issue.severity).lower()
        return issue


@dataclass
class Finding:
    id: str = ""
    issue_id: str = ""
    decision: str = "act"          # act | defer | reject
    summary: str = ""
    acceptance: List[str] = field(default_factory=list)
    non_goals: List[str] = field(default_factory=list)
    #: How the fix could be *shown* wrong: which line to break, what the test
    #: must then report, and that the break must be reverted.  Acceptance
    #: criteria say what must be true afterwards; this says how to tell that the
    #: suite is actually testing the change rather than passing beside it.  A
    #: reviewer rejected three rounds running for its absence, and no later phase
    #: can invent it -- the patch phase only sees the requirement.
    falsification: str = ""
    blast_radius: str = "module"
    risk: str = "medium"
    version: int = 1
    refined_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TestPlan:
    id: str = ""
    issue_id: str = ""
    covers: List[str] = field(default_factory=list)
    kind: str = "unit"
    name: str = ""
    #: File the test belongs in.  Without it the patch phase is told *what* to
    #: test but not *where*, and an anchored edit needs the file's text: a fix
    #: that is "add the missing assertion" can never be expressed.
    path: str = ""
    given: str = ""
    when: str = ""
    then: str = ""
    regression_of: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Review:
    id: str = ""
    issue_id: str = ""
    lens: str = ""
    verdict: str = "abstain"       # approve | reject | abstain
    confidence: float = 0.5
    blocking: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    cycle: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FileOp:
    """A single anchored edit inside one file."""

    path: str
    search: str
    replace: str
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChangeProposal:
    id: str = ""
    issue_ids: List[str] = field(default_factory=list)
    strategy: str = "minimal"
    ops: List[FileOp] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    rationale: str = ""
    risk: str = "medium"
    expected_effect: str = ""
    bug_marker: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["files"] = sorted(set(self.files or [op.path for op in self.ops]))
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChangeProposal":
        ops = [
            FileOp(
                path=str(op.get("path", "")),
                search=str(op.get("search", "")),
                replace=str(op.get("replace", "")),
                rationale=str(op.get("rationale", "")),
            )
            for op in data.get("ops", [])
        ]
        return cls(
            id=str(data.get("id") or ""),
            issue_ids=list(data.get("issue_ids", [])),
            strategy=str(data.get("strategy", "minimal")),
            ops=ops,
            files=list(data.get("files", [])),
            rationale=str(data.get("rationale", "")),
            risk=str(data.get("risk", "medium")),
            expected_effect=str(data.get("expected_effect", "")),
            bug_marker=str(data.get("bug_marker", "")),
        )


def find_open(issues: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(i) for i in issues if str(i.get("status", "open")) == "open"]


# --------------------------------------------------------------------------- #
# declared defects
# --------------------------------------------------------------------------- #

#: ``# BUG: mean_zero -- returns 0 silently`` / ``# BUG(text/title_case): note``
#: The name may contain slashes, dots, underscores and inner dashes; the name
#: ends before `` -- `` or a trailing ``:`` so prose-only markers still match.
MARKER_RE = re.compile(
    r"#\s*(?P<kind>BUG|FIXME|HACK|XXX)\b"
    r"\s*(?:[(:\-]\s*)?"
    r"(?P<name>[A-Za-z0-9_][A-Za-z0-9_./]*(?:-[A-Za-z0-9_./]+)*)?"
    r"\s*[):]?\s*(?:[-:]{1,2}\s*)?(?P<note>.*)?",
    re.I,
)

#: Markers are evidence of an *asserted* defect, not automatically of a real
#: one.  Only these kinds are treated as actionable by reporting/scoring.
ACTIONABLE_MARKERS = ("bug", "fixme")


def comment_index(line: str) -> int:
    """Index of the ``#`` that starts a comment, or ``-1``.

    Quote-aware so a marker inside a string literal (``"# BUG: not a comment"``)
    is not mistaken for a declaration.
    """
    quote = ""
    escaped = False
    for index, char in enumerate(line):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in "\"'":
            quote = char
        elif char == "#":
            return index
    return -1


@dataclass
class Marker:
    path: str
    line: int
    kind: str
    name: str
    note: str
    source: str

    @property
    def key(self) -> str:
        """Stable identity across line shifts: name when declared, else path+line."""
        return self.name if self.name else "%s:%d" % (self.path, self.line)

    @property
    def actionable(self) -> bool:
        return self.kind.lower() in ACTIONABLE_MARKERS

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["key"] = self.key
        data["actionable"] = self.actionable
        return data


def detect_markers(code: Mapping[str, str]) -> List[Marker]:
    out: List[Marker] = []
    for path in sorted(code or {}):
        for line_no, line in enumerate((code[path] or "").splitlines(), start=1):
            index = comment_index(line)
            if index < 0:
                continue
            match = MARKER_RE.search(line, index)
            if not match:
                continue
            out.append(
                Marker(
                    path=path,
                    line=line_no,
                    kind=match.group("kind").upper(),
                    name=(match.group("name") or "").strip(),
                    note=(match.group("note") or "").strip(),
                    source=line.strip(),
                )
            )
    return out


def render_markers(markers: Sequence[Marker], limit: int = 20) -> str:
    if not markers:
        return "(no declared defect markers)"
    lines = []
    for marker in list(markers)[:limit]:
        kind = "actionable" if marker.actionable else "informational"
        lines.append("- %s:%d [%s] %s (%s)" % (marker.path, marker.line, marker.name or "-", marker.source, kind))
    if len(markers) > limit:
        lines.append("- ... %d more" % (len(markers) - limit))
    return "\n".join(lines)


def is_marker_line(line: str, marker: str) -> bool:
    """True when ``line`` carries the declaration comment for ``marker``."""
    index = comment_index(line)
    if index < 0:
        return False
    matched = detect_markers({"<line>": line})
    if not matched:
        return False
    return matched[0].key == marker or matched[0].name == marker


def marker_text(name: str) -> str:
    """The declaration comment a fixed defect should no longer carry."""
    return "# BUG: %s --"% name if name else ""


def strip_marker(code: Mapping[str, str], marker: str, paths: Sequence[str] | None = None) -> Dict[str, str]:
    """Retract the declaration comment for a fixed defect.

    A ``# BUG:`` marker is a *claim* that the code is wrong.  Once a patch has
    fixed the behavior, leaving the claim in place would make the next
    verification pass read the defect as still present -- a marker that outlives
    its defect is worse than no marker.

    Two shapes must survive this intact, because getting either wrong corrupts
    the file:

    * a marker on its **own line** takes that line, plus any immediately
      following same-indent comment lines with it (declarations are often a
      short block);
    * a marker **trailing real code** strips only the comment, never the code.

    Only comments are ever removed here: the code change always comes from the
    patch, never from this function.
    """
    out = dict(code)
    if not marker:
        return out

    for path in [p for p in (paths or out.keys()) if p in out]:
        lines = out[path].splitlines(keepends=True)
        if not any(is_marker_line(line, marker) for line in lines):
            continue
        kept: List[str] = []
        drop_comment_indent: Optional[int] = None
        for line in lines:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip(" \t"))
            if is_marker_line(line, marker):
                index = comment_index(line)
                if line[:index].strip():
                    kept.append(line[:index].rstrip() + "\n")  # trailing comment only
                    drop_comment_indent = None
                else:
                    drop_comment_indent = indent  # whole line goes
                continue
            if drop_comment_indent is not None and stripped.startswith("#") and indent >= drop_comment_indent:
                # A continuation comment belongs to the declaration -- unless it
                # is another defect's declaration, which must survive untouched.
                if detect_markers({"<line>": line}):
                    drop_comment_indent = None
                    kept.append(line)
                    continue
                continue
            drop_comment_indent = None
            kept.append(line)
        out[path] = "".join(kept)
    return out


def compact_issue(issue: Mapping[str, Any]) -> Dict[str, Any]:
    """Trim an issue to the fields that belong in an LLM prompt."""
    keep = (
        "id", "title", "severity", "confidence", "phase_hint", "lens",
        "files", "symbols", "evidence", "why_it_matters", "proposed_direction",
        "status", "seen_count", "bug_marker",
    )
    return {k: issue.get(k) for k in keep if issue.get(k) not in (None, "", [], {})}


# --------------------------------------------------------------------------- #
# diffs
# --------------------------------------------------------------------------- #


def diff_summary(before: Mapping[str, str], after: Mapping[str, str], context: int = 2) -> str:
    """Unified diff of a candidate tree against its baseline."""
    import difflib

    chunks: List[str] = []
    for path in sorted(set(before) | set(after)):
        old = (before.get(path) or "").splitlines(keepends=True)
        new = (after.get(path) or "").splitlines(keepends=True)
        if old == new:
            continue
        chunks.extend(difflib.unified_diff(old, new, fromfile=path, tofile=path, n=context))
    return "".join(chunks) if chunks else "(no textual change)"
