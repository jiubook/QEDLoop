# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Agent specs (who is in each phase) and prompt construction (what they see).

Three agents per phase, five phases, fifteen roles.  Every role declares
* its ``lens`` -- the single perspective it is allowed to reason from,
* its ``mode`` -- what its answer contributes to the phase decision,
* its JSON contract -- which :func:`qedloop.llm.extract_json` keys are read.

Keeping the contract in one place is what makes the phases reviewer-friendly:
you can change an agent's behaviour without touching graph wiring.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .core import clip, find_open, recap, to_dicts
from .llm import LLMProvider, Message

# --------------------------------------------------------------------------- #
# spec
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AgentSpec:
    role: str
    phase: str
    lens: str
    mode: str = "note"        # note | vote | produce | synthesize
    requires_tests: bool = False
    want_code: bool = True
    temperature: float = 0.2
    max_tokens: int = 1400
    description: str = ""
    keys: Sequence[str] = ()

    def __str__(self) -> str:  # pragma: no cover - display helper
        return "%s/%s" % (self.phase, self.role)


AGENTS: Dict[str, List[AgentSpec]] = {
    "discover": [
        AgentSpec(
            "discover_archaeology", "discover", "code-archaeology", temperature=0.1,
            description="reads the code as a contract: dead paths, unreachable branches, contradicted docstrings",
            keys=("issues",),
        ),
        AgentSpec(
            "discover_behavior", "discover", "behavior-and-tests", temperature=0.4, requires_tests=True,
            description="reasons from existing tests and their gaps: uncovered contract, asserted-but-false behavior",
            keys=("issues",),
        ),
        AgentSpec(
            "discover_security", "discover", "security-and-robustness", temperature=0.3,
            description="looks for input trust, resource bounds, silent failure and crash paths",
            keys=("issues",),
        ),
    ],
    "refine": [
        AgentSpec(
            "refine_synthesize", "refine", "requirement-synthesis", mode="produce", temperature=0.2,
            description="turns an issue into an actionable, falsifiable requirement",
            keys=("decision", "summary", "acceptance", "non_goals", "blast_radius", "risk"),
        ),
        AgentSpec(
            "plan_split", "refine", "implementation-planning", mode="produce", temperature=0.2,
            description="decomposes the requirement into ordered steps with a rollback story",
            keys=("steps", "estimate", "rollback"),
        ),
        AgentSpec(
            "test_design", "refine", "test-design", mode="produce", temperature=0.3, requires_tests=True,
            description="designs the test that will falsify or confirm the fix",
            keys=("tests",),
        ),
    ],
    "review": [
        AgentSpec(
            "architecture_review", "review", "architecture", mode="vote", temperature=0.2,
            description="does the change belong here, does it duplicate or warp an existing abstraction",
            keys=("verdict", "confidence", "blocking", "notes"),
        ),
        AgentSpec(
            "correctness_review", "review", "correctness", mode="vote", temperature=0.1,
            description="does the patch actually make the stated acceptance criteria true",
            keys=("verdict", "confidence", "blocking", "notes"),
        ),
        AgentSpec(
            "risk_review", "review", "risk-and-regression", mode="note", temperature=0.2, requires_tests=True,
            description="blast radius, regression surface, reversibility of the change",
            keys=("verdict", "blast_radius", "risk", "notes"),
        ),
    ],
    "patch": [
        AgentSpec(
            "patch_generate", "patch", "minimal-fix", mode="produce", temperature=0.1,
            description="smallest anchored edit that satisfies the acceptance criteria",
            keys=("ops", "rationale", "risk", "expected_effect", "bug_marker"),
        ),
        AgentSpec(
            "patch_refactor", "patch", "refactor-first", mode="produce", temperature=0.5,
            description="alternative patch that removes the class of defect rather than the instance",
            keys=("ops", "rationale", "risk", "expected_effect", "bug_marker"),
        ),
        AgentSpec(
            "patch_reconcile", "patch", "reconciliation", mode="synthesize", temperature=0.1,
            description="judges the two candidate patches and explains the pick",
            keys=("strategy", "pick", "reason"),
        ),
    ],
    "qa": [
        AgentSpec(
            "static_qa", "qa", "static-analysis", mode="note", temperature=0.1,
            description="reads the measured static report: compile status, syntax, new markers",
            keys=("verdict", "observations", "blocking"),
        ),
        AgentSpec(
            "test_sandbox", "qa", "sandboxed-test-run", mode="vote", temperature=0.1, requires_tests=True,
            description="reads before/after test evidence and votes on whether the defect is gone",
            keys=("verdict", "observations", "confidence"),
        ),
        AgentSpec(
            "edge_case_qa", "qa", "adversarial-edge-cases", mode="vote", temperature=0.5,
            description="tries to break the patch with inputs the author did not consider",
            keys=("verdict", "confidence", "observations", "blocking"),
        ),
    ],
}

#: phase -> (node function name, human label) used by reporting.
PHASE_LABELS: Dict[str, str] = {
    "discover": "Phase 1 - Issue discovery",
    "refine": "Phase 2 - Requirement refinement",
    "review": "Phase 3 - Code review",
    "patch": "Phase 4 - Code change",
    "qa": "Phase 5 - QA verification",
}

#: Roles whose ``vote`` counts toward a phase gate.
VOTING_ROLES = tuple(spec.role for specs in AGENTS.values() for spec in specs if spec.mode == "vote")

#: Roles that must reach a verdict for the QA gate to advance.
QA_VOTING_ROLES = tuple(spec.role for spec in AGENTS["qa"] if spec.mode == "vote")


def specs_for(phase: str) -> List[AgentSpec]:
    if phase not in AGENTS:
        raise KeyError("unknown phase %r (known: %s)" % (phase, ", ".join(AGENTS)))
    return list(AGENTS[phase])


def all_specs() -> List[AgentSpec]:
    return [spec for specs in AGENTS.values() for spec in specs]


def find_spec(role: str) -> AgentSpec:
    for spec in all_specs():
        if spec.role == role:
            return spec
    raise KeyError("unknown agent role %r" % role)


# --------------------------------------------------------------------------- #
# prompt construction
# --------------------------------------------------------------------------- #

BASE_SYSTEM = """\
You are one specialised member of an autonomous software-maintenance crew that
iterates over a real repository. The crew runs a fixed cycle:
discover issues -> refine the requirement -> review the plan -> change the code
-> verify with QA -> repeat while quality is below the target.

Rules that override everything else:
1. Answer with ONE JSON object and nothing else. No prose outside the JSON.
2. Never invent test output, file paths, or symbols. If you did not see it in
   the supplied context, say so in a "confidence" field and lower it.
3. Cite concrete anchors: file path plus a copied code line or symbol name.
4. Prefer a defect that can be demonstrated over a matter of taste. Never
   report formatting or naming preferences as issues.
5. If you find nothing worth acting on, return an empty list. An empty batch is
   a valid and useful answer.
6. Your <agent_role> is fixed. Reason only from your lens; other lenses are
   other agents' jobs.
"""

CODE_BLOCK = """\
The repository under iteration (the MANIFEST lists every file; bodies may be
clipped or omitted when the repository is large -- say so in "confidence" if a
defect depends on code you could not read):
{code}
"""

JSON_ONLY = "Return only the JSON object described above."

#: Largest slice of a single file shown to an agent; bigger files are clipped
#: head-and-tail so the beginning (imports, class shape) and the end survive.
PER_FILE_BUDGET = 6000

#: A manifest entry costs ~70 characters and buys an agent the knowledge that a
#: file exists.  Past this cap that trade stops paying, so the list is trimmed
#: with an explicit note rather than crowding out file bodies.
MANIFEST_BUDGET = 7000

#: Character budgets for file *bodies*, per phase.  Discovery gets the most
#: because it has to find something in a tree it has never seen; the later
#: phases are scoped to one issue's files, so they can afford to be tighter.
#: The manifest is counted separately and capped by MAX_MANIFEST_ENTRIES.
DISCOVERY_BUDGET = 24000
REVIEW_BUDGET = 12000
PATCH_BUDGET = 18000
QA_BUDGET = 10000

#: Cap on the per-target project constraints (``run.brief``).  They ride in the
#: system message of *every* agent, so a long file is paid for 15 times per
#: cycle; past this point the cost stops buying compliance.
BRIEF_BUDGET = 4000

#: Wrapper for the project constraints.  Two things matter here: they are
#: standing rules rather than findings, and they are *not* evidence -- an
#: agent may not cite the brief as proof that a defect exists or is gone
#: (invariant 3: only measurement produces evidence).
PROJECT_BRIEF = """\
<project_constraints>
Rules for THIS repository, from its maintainers.  Follow them for every
decision; where they conflict with your own preference, they win.  They are not
evidence: a claim about the code still needs code you were shown, and a claim
about behaviour still needs a measured test result.

{brief}
</project_constraints>"""


def brief_block(state: Mapping[str, Any]) -> str:
    """Render ``run.brief`` for the system message, or "" when unset."""
    brief = str((state or {}).get("brief") or "").strip()
    if not brief:
        return ""
    if len(brief) > BRIEF_BUDGET:
        brief = brief[:BRIEF_BUDGET].rstrip() + "\n\n... [brief truncated at %d characters]" % BRIEF_BUDGET
    return PROJECT_BRIEF.format(brief=brief)


def system_prompt(spec: AgentSpec) -> str:
    return "%s\nPhase: %s\nLens: %s\n<agent_role>%s</agent_role>\n%s" % (
        BASE_SYSTEM,
        PHASE_LABELS.get(spec.phase, spec.phase),
        spec.lens,
        spec.role,
        spec.description,
    )


def render_codebase(
    code_payload: Mapping[str, str],
    budget: int = 14000,
    only: Optional[Sequence[str]] = None,
    lead: Optional[Sequence[str]] = None,
) -> str:
    """Stable, diff-friendly rendering of the tree handed to the agents.

    On a real repository the tree does not fit in a prompt.  Three rules keep
    the payload both affordable and honest:

    * the **manifest** always lists every file with its size, so an agent knows
      what exists even when the bodies are elided -- silently showing only the
      first N files is how a crew ends up reporting issues that are not there;
    * **large files are clipped** head-and-tail rather than dropped, because the
      interesting part of a big module is rarely its middle;
    * ``lead`` puts chosen files first so the budget is spent on them before the
      rest (discovery leads with the test files: they state the contract).
    """
    available = [p for p in sorted(code_payload) if not only or p in only]
    order: List[str] = []
    for path in list(lead or []) + available:
        if path in code_payload and (not only or path in only) and path not in order:
            order.append(path)

    parts: List[str] = [_render_manifest(code_payload, available, order)]
    used = len(parts[0])
    omitted: List[str] = []
    for path in order:
        body = code_payload[path] or ""
        noted = ""
        if len(body) > PER_FILE_BUDGET:
            half = max(1, PER_FILE_BUDGET // 2 - 40)
            elided = body.count("\n") - 2 * half
            body = body[:half] + "\n# ... [%d lines elided] ...\n" % max(0, elided) + body[-half:]
            noted = " [clipped]"
        chunk = "===== FILE: %s =====%s\n%s\n" % (path, noted, body.rstrip("\n"))
        if used + len(chunk) > budget:
            omitted.append(path)
            continue
        used += len(chunk)
        parts.append(chunk)
    if omitted:
        # One summary line instead of one marker per file: the manifest above
        # already names every path, and per-file markers would cost more than
        # the bodies they stand in for.
        listed = ", ".join(omitted[:12]) + (", ..." if len(omitted) > 12 else "")
        parts.append(
            "\n===== OMITTED FOR BUDGET: %d file(s) -- %s\n(their paths are in the manifest; "
            "ask for one by naming it in your evidence or lower your confidence)\n" % (len(omitted), listed)
        )
    if len(parts) == 1:
        parts.append("(no file bodies fitted the prompt budget)\n")
    return "".join(parts)


def _render_manifest(code_payload: Mapping[str, str], available: Sequence[str], order: Sequence[str]) -> str:
    """List the tree -- files in play first, then the rest -- within a byte cap.

    The manifest is the cheapest way to stop an agent reporting issues against
    files it never saw, but it must not crowd out the bodies.  So the budget is
    split: files the agent is *shown* plus the test files (the contract) get
    first claim, the remaining files fill what is left, and anything dropped is
    named in a short closing note rather than silently truncated.
    """
    tests = [p for p in available if _looks_like_test(p)]
    priority: List[str] = []
    for path in list(order) + tests + list(available):
        if path in available and path not in priority:
            priority.append(path)

    lines = ["===== MANIFEST (%d files) =====" % len(available)]
    used = len(lines[0])
    reserved = MANIFEST_BUDGET * 0.5
    listed = 0
    for path in priority:
        body = code_payload[path] or ""
        entry = "  %-58s %6d lines%s" % (
            path if len(path) <= 58 else "..." + path[-55:],
            body.count("\n") + 1,
            "  <-- clipped below" if len(body) > PER_FILE_BUDGET else "",
        )
        cap = MANIFEST_BUDGET if listed >= len(order) + len(tests) else reserved + len(lines[0])
        if used + len(entry) > cap:
            break
        used += len(entry) + 1
        listed += 1
        lines.append(entry)

    dropped = [p for p in available if p not in priority[:listed]]
    if dropped:
        note = "  ... %d more not listed (narrow with --include/--exclude); e.g. %s" % (
            len(dropped), ", ".join(dropped[:3]))
        lines.append(note[:400])
    return "\n".join(lines) + "\n"


def _module_stems(test_path: str) -> List[str]:
    """Likely module names for the implementation a test file exercises."""
    name = test_path.replace("\\", "/").rsplit("/", 1)[-1]
    if not name.endswith(".py"):
        return []
    stem = name[:-3]
    if stem.startswith("test_"):
        stem = stem[len("test_"):]
    stems = [stem]
    # ``test_screenshot_api.py`` covers ``routes/screenshot.py``: the route module
    # is named after the endpoint, the test after the endpoint plus its kind.
    for suffix in ("_api", "_service", "_module", "_endpoint"):
        if stem.endswith(suffix):
            stems.append(stem[: -len(suffix)])
    return [s for s in stems if s]


def source_pairs(test_path: str, code_payload: Mapping[str, str], limit: int = 1) -> List[str]:
    """Non-test modules a test file is most likely written against.

    A test states the contract; the module it imports is where the contract is
    kept.  Showing one without the other is how a crew reports "the test never
    asserts the parameters reach the service" while the route that decides those
    parameter names stays out of the prompt -- measured on EER-Ai, where the
    agent answered that the file it needed "is listed in the manifest but its
    body is under OMITTED FOR BUDGET".

    Matching is by name (``tests/test_x_api.py`` -> ``x.py``, ``x_api.py``,
    ``routes/x.py``, anywhere on the path).  Names collide -- EER-Ai has both
    ``schemas/screenshot.py`` and ``api/routes/screenshot.py`` -- so candidates
    are ranked rather than taken smallest-first: a directory hint taken from the
    test's own filename (``_api``) and the ``routes``/``api`` convention decide,
    because picking the smaller file would have paired the test with a 421-byte
    schema instead of the 3 kB route that actually forwards the parameters.
    """
    name = test_path.replace("\\", "/").rsplit("/", 1)[-1]
    if not name.endswith(".py"):
        return []
    test_directory = test_path.replace("\\", "/").rsplit("/", 1)[0] if "/" in test_path else ""
    hints = [s.lstrip("_") for s in name[:-3].split("_") if s not in ("test", "tests", "py")]

    candidates: List[str] = []
    for stem in _module_stems(test_path):
        for candidate in ("%s.py" % stem, "routes/%s.py" % stem):
            for path in code_payload:
                if _looks_like_test(path):
                    continue
                normalized = path.replace("\\", "/")
                if (normalized == candidate or normalized.endswith("/" + candidate)) and path not in candidates:
                    candidates.append(path)

    def rank(path: str) -> tuple:
        normalized = path.replace("\\", "/")
        words = {part.lower() for part in re.split(r"[/_.]", normalized)}
        same_directory = bool(test_directory) and normalized.rsplit("/", 1)[0] == test_directory
        return (
            not same_directory,                              # written beside the test
            not any(hint in words for hint in hints),        # matches a name hint
            not any(part in normalized for part in ("routes/", "/api/")),  # the convention
            -len(code_payload.get(path) or ""),              # the real module is not the stub
            normalized,
        )

    return sorted(candidates, key=rank)[:limit]


CONTRACT_DIR_HINTS = ("integration", "e2e", "api", "contract", "http", "route", "endpoint")


def _directory_priority(directory: str) -> int:
    """Lower sorts first when the lead walks the test tree.

    Integration tests exercise a whole path -- an endpoint forwards its query
    parameters, a handler propagates an error -- and a defect there is reachable
    from the test text alone.  Unit tests of a pure helper are the opposite: the
    contract lives entirely inside the module, so a test file without its module
    says very little.  Measured on EER-Ai, where the reported issues all sat in
    ``tests/integration`` while a dozen 800-byte ``tests/unit/...`` files
    consumed the rotation ahead of them.
    """
    lowered = directory.lower()
    if any(hint in lowered for hint in CONTRACT_DIR_HINTS):
        return 0
    return 1


def lead_files(
    code_payload: Mapping[str, str],
    *,
    budget: int = DISCOVERY_BUDGET,
    test_share: float = 0.6,
) -> List[str]:
    """The best value for the first slice of the body budget.

    Four rules, each of which exists because of a measured failure:

    * **one test per directory first, contract-dense directories first.**  The
      lead is sampled across packages rather than filled with whatever sorts
      smallest: taking the smallest tests outright put a dozen
      ``tests/unit/recognition/...`` files in front of
      ``tests/integration/test_screenshot_api.py``.
    * **a test pairs with the module it exercises.**  Tests are the cheapest
      statement of intended behaviour, but ``assert_called_once()`` only looks
      weak next to the route that should have forwarded four parameters.  Leading
      with tests alone put 6 test files and 3 source files in the prompt, none of
      them the file the crew said it needed -- "listed in the manifest but its
      body is under OMITTED FOR BUDGET".
    * **nothing empty, nothing that does not fit.**  An empty ``__init__.py`` is
      not "the smallest useful module"; it costs a header and teaches nothing.
      And a candidate that does not fit is skipped, not fatal: stopping the walk
      there spent 24,791 characters on 9 files while leaving room for 28.
    * **``test_share`` bounds the test half** so a test-only reading cannot crowd
      out implementation code; whatever is left goes to the smallest real modules.

    What this does *not* do is make a 145-file tree fit in 24 kB.  Measured on
    EER-Ai the whole first rotation of test directories (one representative each)
    already costs ~27 kB with the manifest, so any given large test file can
    still lose the race -- at the default budget this selection shows 29 files
    (6 tests / 23 source) where the previous one showed 9 (6 tests / 3 source).
    """
    def usable(path: str) -> bool:
        return bool((code_payload.get(path) or "").strip())

    tests = [p for p in sorted(code_payload) if _looks_like_test(p) and usable(p)]

    # One test per directory first, smallest first inside each directory, and
    # contract-dense directories ahead of pure-unit ones.  A test list ordered
    # purely by size never reaches the larger files: EER-Ai has a dozen 800-byte
    # files under tests/unit/recognition/tasks, so
    # tests/integration/test_screenshot_api.py (4.3 kB) stayed out of the prompt
    # at every budget tried up to 32 kB.  Rotating across directories samples the
    # tree instead of the smallest corner of it.
    by_directory: Dict[str, List[str]] = {}
    for path in tests:
        by_directory.setdefault(path.replace("\\", "/").rsplit("/", 1)[0], []).append(path)
    for bucket in by_directory.values():
        bucket.sort(key=lambda p: (len(code_payload[p]), p))
    directories = sorted(by_directory, key=lambda d: (_directory_priority(d), d))
    ordered_tests: List[str] = []
    round_index = 0
    while True:
        added = False
        for directory in directories:
            bucket = by_directory[directory]
            if round_index < len(bucket):
                ordered_tests.append(bucket[round_index])
                added = True
        if not added:
            break
        round_index += 1

    # Two buckets, because they answer different questions and one would
    # otherwise starve the other:
    #
    # * tests state the contract, and each one is admitted together with the
    #   module it exercises -- a test that asserts ``assert_called_once()`` only
    #   looks weak next to the route that should have forwarded four parameters.
    #   Measured on EER-Ai: leading with tests alone put 6 test files and 3 source
    #   files in the prompt, and the file the crew said it needed was not among
    #   them.
    # * implementation code is where most defects live, so the rest of the budget
    #   goes to the smallest source modules, which fit whole.
    #
    # Nothing that does not fit is admitted, and the walk continues rather than
    # stopping: the previous selection spent 24,791 characters on 9 files while
    # leaving room for a dozen more, because one oversized candidate ended the
    # list.
    test_budget = max(1, int(budget * test_share))
    chosen: List[str] = []
    seen: set = set()
    used = 0
    for path in ordered_tests:
        group = [path]
        module_pair = source_pairs(path, code_payload)
        if module_pair and len(code_payload[path]) + len(code_payload[module_pair[0]]) <= test_budget:
            group.append(module_pair[0])
        if any(member in seen for member in group):
            continue
        cost = sum(len(code_payload[member]) for member in group)
        if used + cost > test_budget:
            continue
        for member in group:
            chosen.append(member)
            seen.add(member)
        used += cost

    for path in sorted(
        (p for p in code_payload if p not in seen and not _looks_like_test(p) and usable(p)),
        key=lambda p: (len(code_payload[p]), p),
    ):
        size = len(code_payload[path])
        if used + size > budget:
            continue
        chosen.append(path)
        seen.add(path)
        used += size
    return chosen


def focus_files(
    focus: Mapping[str, Any],
    plans: Sequence[Mapping[str, Any]] = (),
    existing: Sequence[str] = (),
) -> List[str]:
    """Paths that must win the body-budget race, in claim order.

    Every prompt that asks an agent to judge or edit a specific change has to
    show it the files that change touches.  Without this the renderer spends the
    budget alphabetically and those files lose to whatever sorts first: measured
    on a real repository, 136 of 145 files were elided and the two the issue
    named were both among them -- the reviewers were then asked to certify
    parameter names they were structurally forbidden from reading, and every
    round they refused for exactly that reason.

    The issue's own ``files`` come first (they are where the defect was seen),
    then any file a plan says it will be written to.  Callers that keep a
    separately ordered ``lead`` open-code that themselves; this helper exists so
    refine, review and patch cannot drift apart on which files matter.
    """
    ordered: List[str] = []
    for candidate in list(focus.get("files") or []) + [p.get("path") for p in plans] + list(existing):
        path = str(candidate or "")
        if path and path not in ordered:
            ordered.append(path)
    return ordered


def _looks_like_test(path: str) -> bool:
    normalized = path.replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or "/tests/" in normalized
        or normalized.startswith("tests/")
    )


def _json_block(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _issue_batch(issues: Sequence[Mapping[str, Any]], limit: int = 6) -> str:
    return _json_block(to_dicts(issues)[:limit])


# --------------------------------------------------------------------------- #
# per-phase prompt builders
# --------------------------------------------------------------------------- #


def discover_prompt(spec: AgentSpec, state: Mapping[str, Any]) -> List[Message]:
    issues = state.get("issues") or []
    cycle = int(state.get("cycle", 0))
    code = state.get("working_code") or {}
    prior = recap(issues, ("id", "title", "severity", "status", "seen_count"), limit=40)
    verified = [i for i in issues if str(i.get("status")) == "verified"]
    tests = sorted(p for p in code if _looks_like_test(p))
    baseline = state.get("baseline_tests") or {}
    measured = (
        "the suite is GREEN (%d passed, %d skipped) -- look for defects the tests do not cover"
        % (baseline.get("passed", 0), baseline.get("skipped", 0))
        if baseline.get("green")
        else "the suite is RED (%d passed, %d failed, %d errors) -- start from these failures:\n%s"
        % (
            baseline.get("passed", 0),
            baseline.get("failed", 0),
            baseline.get("errors", 0),
            "\n".join("  " + str(t) for t in (baseline.get("failing") or [])[:15]) or "  (failure names unavailable)",
        )
        if baseline
        else "the baseline suite was not measured"
    )
    task = """\
Cycle: {cycle}
Repository: {files} files under iteration ({tests} of them are tests).

Measured baseline: {measured}

Ledger of issues already known (do NOT re-report these; a new cycle starts only
after the previous batch was fixed and verified):
{prior}

Issues already fixed and verified (measured green):
{verified}

Your job: find NEW issues in the repository below, from your lens only.
Return at most 3, ordered by how strongly the evidence supports them.
On a large repository the file bodies below may be clipped or omitted; the
MANIFEST lists everything that exists. Prefer issues you can point at with a
copied line. Never report style, naming or formatting preferences.

Schema:
{{
  "issues": [
    {{
      "title": "short, specific, <= 100 chars",
      "severity": "critical|high|medium|low",
      "confidence": 0.0,
      "files": ["path/relative/to/repo.py"],
      "symbols": ["function_or_class_name"],
      "evidence": "the exact copied line(s) that prove it",
      "why_it_matters": "what breaks for a caller or user",
      "proposed_direction": "the direction of the fix, not the patch",
      "bug_marker": "marker name if the code declares its own defect, else \\"\\""
    }}
  ]
}}
If the code declares a defect on its own line (a marker comment naming the
defect), report it and copy that marker verbatim into "bug_marker".
""" .format(
        cycle=cycle,
        files=len(code),
        tests=len(tests),
        measured=measured,
        prior=prior,
        verified=", ".join(str(i.get("id")) for i in verified) or "(none)",
    )
    return [
        Message("system", system_prompt(spec)),
        Message(
            "user",
            task
            + "\n"
            + CODE_BLOCK.format(
                code=render_codebase(
                    code,
                    budget=DISCOVERY_BUDGET,
                    lead=lead_files(code, budget=DISCOVERY_BUDGET),
                )
            )
            + "\n"
            + JSON_ONLY,
        ),
    ]
    return [
        Message("system", system_prompt(spec)),
        Message("user", task + "\n" + CODE_BLOCK.format(code=render_codebase(state.get("working_code") or {})) + "\n" + JSON_ONLY),
    ]


def refine_prompt(
    spec: AgentSpec,
    state: Mapping[str, Any],
    focus: Mapping[str, Any],
    synthesis: Mapping[str, Any] | None = None,
) -> List[Message]:
    reviews = [r for r in (state.get("reviews") or []) if r.get("verdict") == "reject"][-3:]
    blockers = []
    for review in reviews:
        blockers.extend(to_dicts(review.get("blocking") or []))
    schemas = {
        "refine_synthesize": """\
{
  "decision": "act|defer|reject",
  "summary": "the requirement in one paragraph, stated so it can be proven false",
  "acceptance": ["observable check 1", "observable check 2"],
  "falsification": "the line to break and the run that must then fail, so the change is proven to be what makes it pass",
  "non_goals": ["what this change must NOT touch"],
  "blast_radius": "function|module|package|system",
  "risk": "low|medium|high"
}

``decision`` is a judgement about this round, not about the issue's importance.
``act`` means the requirement can be stated so that a test can prove it false.
``reject`` means the finding is not a real defect.  ``defer`` means it is a real
defect that cannot be turned into an executable plan yet -- it needs a design
decision, a dependency, or an edit outside the surface this run may write.  A
deferral is neither free nor silent: the issue leaves the batch, and with no
other issue in flight the run hands itself back to a human instead of spending
its remaining refine budget on steps nobody can run.  Say so when that is the
honest answer; do not use it to avoid a hard requirement.
""",
        "plan_split": """\
{
  "steps": [{"order": 1, "action": "concrete edit or command", "files": ["path.py"]}],
  "estimate": "XS|S|M|L",
  "rollback": "how to undo this safely"
}

A plan is not complete until it says how the change could be shown to be the
thing that makes the test pass.  The ``falsification`` the requirement states
becomes steps of its own: break exactly that line, run the named test, record
that it fails, revert the break, run it again, record that it passes.  A patch
whose test would pass without the patch has not been verified, and the reviewers
reject a plan that cannot tell the difference.
""",
        "test_design": """\
{
  "tests": [{
    "kind": "unit|regression|property|integration",
    "name": "test_...",
    "path": "tests/test_mod.py -- the file this test is added to, or \"\" for a new file",
    "given": "input/state",
    "when": "call or command",
    "then": "expected observable result",
    "regression_of": "issue id or empty"
  }]
}""",
    }
    # The synthesis runs first and its non-goals and acceptance criteria are
    # handed to the two agents that plan and design.  They used to run in
    # parallel with no sight of each other, so the synthesis wrote "do not add a
    # new test function" while the test designer -- which never saw that line --
    # proposed one, and the reviewers rejected the round for contradicting a
    # requirement written four seconds earlier in the same phase.
    constraint = (
        _json_block(synthesis)
        if synthesis
        else "(not stated for this round: stay inside the issue's own scope)"
    )
    task = """\
Issue under refinement:
{issue}

Previously rejected review findings to resolve (may be empty):
{blockers}

Constraints this round's refinement already committed to -- your output must
not contradict them, and a violation is grounds for the review to reject the
whole plan:
{constraint}

Return JSON:
{schema}
""" .format(
        issue=_json_block(focus),
        blockers=_json_block(blockers),
        constraint=constraint,
        schema=schemas[spec.role],
    )
    messages = [Message("system", system_prompt(spec)), Message("user", task)]
    if spec.want_code:
        messages.append(
            Message(
                "user",
                CODE_BLOCK.format(
                    code=render_codebase(
                        state.get("working_code") or {},
                        budget=REVIEW_BUDGET,
                        lead=focus_files(focus),
                    )
                )
                + "\n"
                + JSON_ONLY,
            )
        )
    else:
        messages.append(Message("user", JSON_ONLY))
    return messages


def review_prompt(spec: AgentSpec, state: Mapping[str, Any]) -> List[Message]:
    focus = focus_issue(state)
    finding = latest_finding(state, str(focus.get("id", "")))
    plans = plans_for(state, str(focus.get("id", "")))
    task = """\
Issue: {issue}

Refined requirement:
{finding}

Planned tests:
{plans}

Evidence currently measured (may be empty at this point in the cycle):
{evidence}

Judge the PLAN, not an implementation that does not exist yet.
Return JSON:
{{
  "verdict": "approve|reject|abstain",
  "confidence": 0.0,
  "blocking": [{{"issue_id": "{issue_id}", "reason": "what must change before this may proceed"}}],
  "notes": "one short paragraph from your lens",
  "blast_radius": "function|module|package|system",
  "risk": "low|medium|high"
}}
Reject only for a concrete, nameable defect in the plan. If your lens has no
objection, approve; use abstain only when the plan lacks the information you
would need to judge.
""" .format(
        issue=_json_block(focus),
        finding=_json_block(finding) if finding else "(none yet)",
        plans=_json_block(plans) if plans else "(none yet)",
        evidence=_json_block(state.get("evidence") or {}),
        issue_id=focus.get("id", ""),
    )
    return [
        Message("system", system_prompt(spec)),
        Message("user", task),
        # The reviewers judge a change to code, so the code has to be in front of
        # them.  This block used to get the smallest budget in the file and no
        # lead at all, which meant the two files the issue named were always
        # among the elided: every rejection cited a parameter name it could not
        # read.  Same lead rule as refine and patch.
        Message(
            "user",
            CODE_BLOCK.format(
                code=render_codebase(
                    state.get("working_code") or {},
                    budget=REVIEW_BUDGET,
                    lead=focus_files(focus, plans),
                )
            )
            + "\n"
            + JSON_ONLY,
        ),
    ]


def patch_prompt(spec: AgentSpec, state: Mapping[str, Any]) -> List[Message]:
    focus = focus_issue(state)
    finding = latest_finding(state, str(focus.get("id", "")))
    plans = plans_for(state, str(focus.get("id", "")))
    # The planned files come first as well as the issue's own: when the fix *is*
    # a test, the file it must be anchored in is named only by the plan, and a
    # body that loses the budget race against the rest of the tree is a patch
    # the agents cannot write ("no anchored edit is possible with the
    # information given").
    lead = focus_files(focus, plans)
    strategy = "smallest possible anchored edit" if spec.role == "patch_generate" else "remove the class of defect, may touch more lines"
    task = """\
Issue: {issue}

Requirement and acceptance criteria:
{finding}

Planned tests that must pass afterwards:
{plans}

Your strategy for this attempt: {strategy}

Emit edits as exact anchored replacements. "search" MUST appear verbatim in the
named file, character for character, including indentation, and MUST be unique
inside that file. Keep "search" as short as is still unique.

To ADD a file, emit one op with an empty "search", the whole file in "replace",
and a rationale that says the file is new (for example "新增回归测试文件"). Only
python files can be added, and only when they do not already exist -- a file that
exists must be changed with an anchor instead. Do not leave "search" empty for
any other reason: an empty search is read as "create this file", never as "replace
the whole file".

Return JSON:
{{
  "ops": [{{"path": "src/mod.py", "search": "exact existing text", "replace": "new text", "rationale": "why"}}],
  "rationale": "one paragraph",
  "risk": "low|medium|high",
  "expected_effect": "the observable difference after this patch",
  "bug_marker": "marker name if you are fixing a declared defect, else \\"\\""
}}
If no anchored edit is possible with the information given, return
{{"ops": [], "rationale": "why not", "risk": "unknown", "expected_effect": "none"}}.
""" .format(
        issue=_json_block(focus),
        finding=_json_block(finding) if finding else "(none)",
        plans=_json_block(plans) if plans else "(none)",
        strategy=strategy,
    )
    return [
        Message("system", system_prompt(spec)),
        Message("user", task),
        Message(
            "user",
            CODE_BLOCK.format(
                code=render_codebase(
                    state.get("working_code") or {},
                    budget=PATCH_BUDGET,
                    lead=lead,
                )
            )
            + "\n"
            + JSON_ONLY,
        ),
    ]


def reconcile_prompt(spec: AgentSpec, state: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> List[Message]:
    task = """\
Two candidate patches for the same issue. Candidate A is the minimal edit,
candidate B is the refactor-first alternative.

{candidates}

Choose the one that is most likely to satisfy the acceptance criteria while
keeping the change reviewable and reversible. Prefer A unless B removes the
defect class without widening the blast radius.

Return JSON:
{{
  "strategy": "minimal|refactor",
  "pick": "A|B",
  "reason": "one paragraph"
}}
""" .format(candidates=_json_block(candidates))
    return [Message("system", system_prompt(spec)), Message("user", task)]


def qa_prompt(spec: AgentSpec, state: Mapping[str, Any]) -> List[Message]:
    focus = focus_issue(state)
    finding = latest_finding(state, str(focus.get("id", "")))
    plan = state.get("selected_patch") or (state.get("patches") or [{}])[-1]
    evidence = state.get("evidence") or {}
    task = """\
Cycle {cycle}. Issue under verification:
{issue}

Acceptance criteria:
{finding}

Patch that was applied to the candidate tree:
{patch}

Measured evidence (this is the only source of truth about test outcomes):
{evidence}

Your lens: {lens}. Do not repeat the test numbers as if you had run them
yourself; interpret them. If the evidence contradicts the acceptance criteria,
reject. If you found a concrete counterexample, put it in "blocking".

Return JSON:
{{
  "verdict": "approve|reject|abstain",
  "confidence": 0.0,
  "observations": "one short paragraph",
  "blocking": [{{"issue_id": "{issue_id}", "reason": "counterexample or unmet criterion"}}]
}}
""" .format(
        cycle=state.get("cycle", 0),
        issue=_json_block(focus),
        finding=_json_block(finding) if finding else "(none)",
        patch=_json_block({k: plan.get(k) for k in ("id", "strategy", "files", "rationale", "expected_effect")}),
        evidence=_json_block(evidence),
        lens=spec.lens,
        issue_id=focus.get("id", ""),
    )
    return [
        Message("system", system_prompt(spec)),
        Message("user", task),
        Message(
            "user",
            CODE_BLOCK.format(
                code=render_codebase(
                    state.get("working_code") or {},
                    budget=QA_BUDGET,
                    only=list(plan.get("files") or []) or None,
                )
            )
            + "\n"
            + JSON_ONLY,
        ),
    ]


# --------------------------------------------------------------------------- #
# focus helpers -- which issue the cycle is currently working on
# --------------------------------------------------------------------------- #


def focus_issue(state: Mapping[str, Any]) -> Dict[str, Any]:
    """The single issue a refine/review/patch/qa pass is scoped to.

    Batched phases still keep the state bus honest: every downstream ledger row
    records ``focus_id`` so a report can be regrouped per issue later.
    """
    planned = state.get("focus_ids") or []
    open_issues = find_open(state.get("issues") or [])
    if planned:
        for issue in open_issues:
            if issue["id"] == planned[0]:
                return issue
    if open_issues:
        return open_issues[0]
    return {"id": "NONE", "title": "(no open issue)", "status": "closed"}


def latest_finding(state: Mapping[str, Any], issue_id: str) -> Dict[str, Any]:
    matches = [f for f in (state.get("findings") or []) if f.get("issue_id") == issue_id]
    return matches[-1] if matches else {}


def plans_for(state: Mapping[str, Any], issue_id: str) -> List[Dict[str, Any]]:
    """The test plans in force for one issue: this round's, not history.

    A refine round replaces the whole plan, and the ledger keeps earlier
    generations for audit with ``superseded_at`` set.  Showing them to a
    reviewer is what produced findings like "PLN-0003, PLN-0007 and PLN-0009 are
    three separate entries for the same existing test" -- true of the ledger,
    false of the plan the round actually agreed on.
    """
    return [
        plan
        for plan in (state.get("test_plans") or [])
        if issue_id in (plan.get("covers") or []) and not plan.get("superseded_at")
    ]


# --------------------------------------------------------------------------- #
# prompt registry used by the crew runner
# --------------------------------------------------------------------------- #

PROMPT_BUILDERS = {
    "discover": lambda spec, state, ctx: discover_prompt(spec, state),
    "refine": lambda spec, state, ctx: refine_prompt(
        spec,
        state,
        ctx.get("focus") or focus_issue(state),
        ctx.get("synthesis") or None,
    ),
    "review": lambda spec, state, ctx: review_prompt(spec, state),
    "patch": lambda spec, state, ctx: patch_prompt(spec, state),
    "qa": lambda spec, state, ctx: qa_prompt(spec, state),
    "reconcile": lambda spec, state, ctx: reconcile_prompt(spec, state, ctx.get("candidates") or []),
}


def build_messages(spec: AgentSpec, state: Mapping[str, Any], **ctx: Any) -> List[Message]:
    """Single entry point for every agent call.

    The project constraints are appended to the system message *here* rather
    than in each of the seven prompt builders: every agent, in every phase and
    in the reconcile pass, has to work under the same rules, and one injection
    point is the only version of that which cannot drift.
    """
    builder = PROMPT_BUILDERS["reconcile"] if spec.mode == "synthesize" else PROMPT_BUILDERS[spec.phase]
    messages = list(builder(spec, state, ctx))
    brief = brief_block(state)
    if brief and messages and messages[0].role == "system":
        messages[0] = Message("system", messages[0].content.rstrip() + "\n\n" + brief)
    return messages
