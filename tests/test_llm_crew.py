# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 jiubook

"""Agent layer tests: JSON contracts, provider behaviour, crew dispatch."""

from __future__ import annotations

import json

import pytest

from qedloop.crew import AgentOutputError, AgentTask, Crew, extract_json
from qedloop.core import as_float
from qedloop.llm import (
    CachingProvider,
    LLMProvider,
    LLMReply,
    Message,
    MockProvider,
    RecordingProvider,
    make_provider,
)
from qedloop.prompts import AGENTS, all_specs, build_messages, find_spec, specs_for


# --------------------------------------------------------------------------- #
# tolerant JSON extraction
# --------------------------------------------------------------------------- #


def test_extract_plain_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_fenced_object():
    assert extract_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}


def test_extract_object_with_prose_around_it():
    text = 'Here is my answer:\n{"verdict": "approve"}\nHope that helps!'
    assert extract_json(text) == {"verdict": "approve"}


def test_extract_object_containing_braces_in_strings():
    payload = {"note": "use {braces} carefully", "n": 2}
    assert extract_json("prefix " + json.dumps(payload) + " suffix") == payload


def test_extract_prefers_the_object_over_surrounding_text():
    text = 'noise {"first": 1} more noise {"second": 2}'
    assert extract_json(text) == {"first": 1}


def test_extract_raises_on_non_json():
    with pytest.raises(AgentOutputError):
        extract_json("I could not complete the task.")


# --------------------------------------------------------------------------- #
# agents
# --------------------------------------------------------------------------- #


def test_five_phases_with_three_agents_each():
    assert sorted(AGENTS) == ["discover", "patch", "qa", "refine", "review"]
    for phase, specs in AGENTS.items():
        assert len(specs) == 3, "phase %s must have exactly three lenses" % phase
    assert len(all_specs()) == 15


def test_every_voting_gate_has_at_least_one_voter():
    assert len([s for s in specs_for("review") if s.mode == "vote"]) >= 2
    assert len([s for s in specs_for("qa") if s.mode == "vote"]) >= 2


def test_prompt_carries_role_and_phase_contract():
    state = {"cycle": 2, "working_code": {"src/a.py": "x = 1\n"}, "issues": []}
    messages = build_messages(find_spec("discover_security"), state)
    assert messages[0].role == "system"
    assert "<agent_role>discover_security</agent_role>" in messages[0].content
    blob = "\n".join(m.content for m in messages)
    assert "===== FILE: src/a.py =====" in blob
    assert "one JSON object" in messages[0].content.lower() or "ONE JSON object" in messages[0].content


def test_prompt_lists_known_issues_so_they_are_not_refiled():
    state = {
        "cycle": 1,
        "working_code": {},
        "issues": [{"id": "ISS-0001", "title": "known", "severity": "high", "status": "verified", "seen_count": 2}],
    }
    blob = "\n".join(m.content for m in build_messages(find_spec("discover_archaeology"), state))
    assert "ISS-0001" in blob and "do NOT re-report" in blob


def test_every_spec_has_a_prompt_builder():
    state = {"cycle": 0, "working_code": {}, "issues": [], "reviews": [], "evidence": {}, "findings": []}
    for spec in all_specs():
        messages = build_messages(spec, state, candidates=[], focus={})
        assert messages and messages[0].role == "system"


# --------------------------------------------------------------------------- #
# per-target project constraints (run.brief)
# --------------------------------------------------------------------------- #


def _bare_state(**extra) -> dict:
    state = {"cycle": 0, "working_code": {}, "issues": [], "reviews": [], "evidence": {}, "findings": []}
    state.update(extra)
    return state


def test_project_brief_reaches_every_agent():
    """Constraints are standing orders for all fifteen agents, in every phase.

    One injection point in ``build_messages`` is what keeps that true: a new
    lens -- or the reconcile pass, which has its own builder -- cannot silently
    opt out of the rules the maintainer wrote down.
    """
    state = _bare_state(brief="Never edit generated files.")
    for spec in all_specs():
        content = build_messages(spec, state, candidates=[], focus={})[0].content
        assert "<project_constraints>" in content, spec.role
        assert "Never edit generated files." in content, spec.role


def test_project_brief_is_absent_when_unset():
    content = build_messages(find_spec("discover_security"), _bare_state())[0].content
    assert "<project_constraints>" not in content, "no empty section should be sent"


def test_project_brief_is_capped_and_says_so():
    from qedloop.prompts import BRIEF_BUDGET

    content = build_messages(find_spec("discover_security"), _bare_state(brief="x" * (BRIEF_BUDGET * 3)))[0].content
    assert "brief truncated" in content, "a clipped brief must announce itself"
    assert len(content) < BRIEF_BUDGET + 3000


def test_project_brief_states_that_it_is_not_evidence():
    """A rule is not a finding: the brief must never license an unmeasured claim."""
    content = " ".join(build_messages(find_spec("discover_security"), _bare_state(brief="Prefer small diffs."))[0].content.split())
    assert "not evidence" in content


# --------------------------------------------------------------------------- #
# per-agent output budget
# --------------------------------------------------------------------------- #


def test_a_channel_output_budget_overrides_the_agent_default():
    """A thinking model needs room for its chain of thought.

    Reasoning tokens are billed as output and counted against ``max_tokens``,
    so the framework's 1400-token JSON answer budget truncates the reply before
    the JSON starts.  A channel that declares its own ceiling wins.
    """
    spec = find_spec("discover_security")
    plain = Crew(make_provider("mock"))
    assert plain.output_budget(spec) == spec.max_tokens, "no declaration: the contract decides"

    declared = Crew(make_provider("mock", max_tokens=16000))
    assert declared.output_budget(spec) == 16000


def test_the_declared_budget_is_what_actually_reaches_the_provider():
    """The ceiling must arrive as the request's ``max_tokens``, not merely be computed."""
    seen = {}

    class _Recorder(LLMProvider):
        max_tokens = 16000

        def complete(self, messages, *, temperature=0.2, max_tokens=1200):
            seen["max_tokens"] = max_tokens
            return LLMReply(text='{"issues": []}', model="recorder")

    Crew(_Recorder()).run_one(AgentTask(spec=find_spec("discover_security"), state=_bare_state()))
    assert seen["max_tokens"] == 16000


def test_a_truncated_reply_names_the_budget_instead_of_only_the_symptom():
    """"not a single JSON object" hides a limit the caller can actually raise."""

    class _Truncating(LLMProvider):
        max_tokens = 1400

        def complete(self, messages, *, temperature=0.2, max_tokens=1200):
            return LLMReply(text='{"issues": [{"title": "half a js', model="cut",
                             meta={"finish_reason": "length"})

    result = Crew(_Truncating(), max_retries=0).run_one(
        AgentTask(spec=find_spec("discover_security"), state=_bare_state())
    )
    assert result.ok is False
    assert "max_tokens=1400" in result.error and "cut off" in result.error


# --------------------------------------------------------------------------- #
# model-authored numbers
# --------------------------------------------------------------------------- #


def test_a_word_where_a_number_was_asked_for_is_read_not_fatal():
    """Agents answer "low - ..." for a 0..1 field; that must not abort a run.

    A real run died on ``float("low - no candidate patches ...")`` after 144k
    tokens had been spent, and -- because the graph discarded the state -- the
    report of that run said no issue had been found.  Reading the word keeps the
    information and costs nothing.
    """
    assert as_float(0.85) == 0.85
    assert as_float("0.85") == 0.85
    assert as_float("low - no candidate patches or issue text were present") == 0.25
    assert as_float("High") == 0.80
    assert as_float("85%") == 85.0
    assert as_float("", 0.5) == 0.5
    assert as_float(None, 0.5) == 0.5
    assert as_float([1, 2], 0.5) == 0.5, "a wrong shape falls back instead of raising"


def test_agent_rows_survives_a_prose_confidence():
    """The crash site itself: the row builder reads every agent's answer."""
    from qedloop.crew import AgentResult
    from qedloop.phases.base import agent_rows

    result = AgentResult(role="patch_reconcile", phase="patch", lens="reconcile", mode="synthesize")
    result.data = {"confidence": "low - no candidate patches were present in the context"}
    rows = agent_rows([result], phase="patch", cycle=0)
    assert rows[0]["confidence"] == 0.25


def test_run_phase_can_narrow_the_fan_out():
    """A lens that is a *tool* (reconcile) must not run as a peer of the generators."""
    crew = Crew(MockProvider())
    everything = [r.role for r in crew.run_phase("patch", _bare_state(), focus={})]
    assert "patch_reconcile" in everything, "precondition: the phase declares three lenses"

    narrowed = [r.role for r in crew.run_phase("patch", _bare_state(), roles=["patch_generate"], focus={})]
    assert narrowed == ["patch_generate"]


def test_usage_reports_the_prompt_completion_split():
    """Input and output tokens are priced apart; a run's cost needs both."""

    class _Priced(LLMProvider):
        def complete(self, messages, *, temperature=0.2, max_tokens=1200):
            return LLMReply(text='{"issues": []}', model="priced", prompt_tokens=2000, completion_tokens=500)

    from qedloop.orchestrator import usage_of

    crew = Crew(_Priced())
    crew.run_one(AgentTask(spec=find_spec("discover_security"), state=_bare_state()))
    usage = usage_of(crew)
    assert usage["prompt_tokens"] == 2000 and usage["completion_tokens"] == 500
    assert usage["tokens"] == 2500


# --------------------------------------------------------------------------- #
# rendering a real-sized repository
# --------------------------------------------------------------------------- #


def _big_tree(files: int = 120, lines: int = 60) -> dict:
    tree = {"tests/test_mod.py": "def test_ok():\n    assert True\n"}
    for index in range(files):
        tree["src/module_%03d.py" % index] = "".join(
            "def f_%d_%d():\n    return %d\n" % (index, line, line) for line in range(lines)
        )
    return tree


def test_manifest_lists_every_file_even_when_bodies_are_elided():
    """On a real repository the bodies cannot fit; hiding the rest of the tree
    would make agents report issues against files they never saw."""
    from qedloop.prompts import render_codebase

    tree = _big_tree()
    rendered = render_codebase(tree, budget=6000)
    assert "MANIFEST (%d files)" % len(tree) in rendered
    assert "tests/test_mod.py" in rendered, "the files whose bodies are shown must be listed"
    assert "OMITTED FOR BUDGET" in rendered
    assert "more not listed" in rendered, "dropped entries are named, not hidden"


def test_manifest_is_capped_but_never_misleading():
    """Every listed entry must be a real file, and the count must be honest."""
    from qedloop.prompts import MANIFEST_BUDGET, render_codebase

    tree = _big_tree(files=1500)
    rendered = render_codebase(tree, budget=1000)
    manifest = rendered.split("===== FILE: ")[0]
    assert len(manifest) < MANIFEST_BUDGET + 600, "the manifest respects its own cap"
    assert "MANIFEST (1501 files)" in manifest
    listed = [line for line in manifest.splitlines() if line.startswith("  ") and "more not listed" not in line]
    for line in listed[:20]:
        path = line[2:60].strip()
        assert path in tree or path.startswith("..."), "manifest names only real files: %r" % path


def test_render_respects_the_budget():
    """Manifest (capped bytes) plus bodies (the phase budget)."""
    from qedloop.prompts import MANIFEST_BUDGET, render_codebase

    tree = _big_tree()
    rendered = render_codebase(tree, budget=8000)
    manifest_end = rendered.find("\n===== FILE: ")
    assert manifest_end == -1 or manifest_end < MANIFEST_BUDGET + 1000
    assert "OMITTED FOR BUDGET" in rendered, "the rest of the tree must be named, not hidden"
    assert len(rendered) < MANIFEST_BUDGET + 8000 + 2500, "manifest + bodies + summary"


def test_large_files_are_clipped_head_and_tail_not_dropped():
    from qedloop.prompts import PER_FILE_BUDGET, render_codebase

    body = "".join("line_%04d = %d\n" % (i, i) for i in range(3000))
    tree = {"src/huge.py": body}
    rendered = render_codebase(tree, budget=PER_FILE_BUDGET * 2)
    assert "[clipped]" in rendered
    assert "line_0000" in rendered and "line_2999" in rendered
    assert "lines elided" in rendered


def test_lead_files_are_rendered_first():
    from qedloop.prompts import render_codebase

    tree = {"aaa/zzz.py": "x = 1\n", "tests/test_a.py": "def test_a():\n    pass\n"}
    rendered = render_codebase(tree, budget=100000, lead=["tests/test_a.py"])
    assert rendered.index("FILE: tests/test_a.py") < rendered.index("FILE: aaa/zzz.py")


def test_the_patch_prompt_shows_the_file_the_plan_names():
    """A fix that lives in a test file has to be able to see that test file.

    Measured on a real repository: the issue pointed at a source route, the plan
    said "add the missing assertion", and the test file lost the body-budget
    race against the rest of the tree (tests/ sorts after src/).  Both patch
    agents answered "no anchored edit is possible with the information given"
    and the cycle closed nothing.  The plan's ``path`` now leads the render.
    """
    tree = {"src/part_%02d.py" % n: "value = %d\n" % n * 300 for n in range(20)}
    tree["tests/test_app.py"] = "def test_thing():\n    assert True\n"
    state = _bare_state(
        working_code=tree,
        issues=[{"id": "ISS-0001", "title": "no test covers this", "severity": "low",
                 "status": "open", "files": ["src/part_00.py"]}],
        focus_ids=["ISS-0001"],
        test_plans=[{"issue_id": "ISS-0001", "covers": ["ISS-0001"], "kind": "integration",
                     "name": "test_thing", "path": "tests/test_app.py"}],
    )

    blob = "\n".join(m.content for m in build_messages(find_spec("patch_generate"), state))
    assert "===== FILE: tests/test_app.py =====" in blob, "the planned file must be rendered"
    assert "===== FILE: src/part_00.py =====" in blob, "and the issue's own file must stay visible"


def test_refine_and_review_are_shown_the_files_the_issue_names():
    """Every phase that judges a change must be able to read the change's files.

    Measured on a real repository: refine and review rendered the tree with no
    lead, so the budget went alphabetically and the issue's own two files were
    both elided (136 of 145 files were).  The agents said so in their own notes
    -- "Both files the assertion depends on are missing from what I was shown"
    -- and then every review round rejected the plan for guessing parameter
    names it was forbidden from reading.  Only ``patch_prompt`` had a lead.
    """
    tree = {"src/part_%02d.py" % n: "value = %d\n" % n * 300 for n in range(20)}
    tree["tests/test_app.py"] = "def test_thing():\n    assert True\n"
    state = _bare_state(
        working_code=tree,
        issues=[{"id": "ISS-0001", "title": "no test covers this", "severity": "low",
                 "status": "open", "files": ["src/part_00.py"]}],
        focus_ids=["ISS-0001"],
        reviews=[],
        test_plans=[],
    )
    wanted = "===== FILE: src/part_00.py ====="

    for role in ("refine_synthesize", "plan_split", "test_design",
                 "architecture_review", "correctness_review", "risk_review"):
        blob = "\n".join(m.content for m in build_messages(find_spec(role), state))
        assert wanted in blob, "%s must be shown the file the issue names" % role


def test_review_is_shown_the_file_the_plan_will_be_written_to():
    """Same rule for the plan's target: the reviewer judges that exact file."""
    tree = {"src/part_%02d.py" % n: "value = %d\n" % n * 300 for n in range(20)}
    tree["tests/test_app.py"] = "def test_thing():\n    assert True\n"
    state = _bare_state(
        working_code=tree,
        issues=[{"id": "ISS-0001", "title": "no test covers this", "severity": "low",
                 "status": "open", "files": ["src/part_00.py"]}],
        focus_ids=["ISS-0001"],
        reviews=[],
        test_plans=[{"issue_id": "ISS-0001", "covers": ["ISS-0001"], "kind": "integration",
                     "name": "test_thing", "path": "tests/test_app.py"}],
    )

    blob = "\n".join(m.content for m in build_messages(find_spec("correctness_review"), state))
    assert "===== FILE: tests/test_app.py =====" in blob, "the planned file must be rendered"


def test_refine_runs_the_synthesis_before_the_planning_lenses():
    """The two planning lenses must be handed the criteria they are bound by.

    They used to run in parallel with the synthesis and see none of it, so a
    round produced a plan that contradicted a non-goal written in the same
    phase -- and the reviewers rejected it for exactly that, three rounds in a
    row.  This pins both halves: emission order, and the constraint reaching the
    later prompts.
    """
    from qedloop.phases.refine import make_refine_node

    order: list = []
    first_prompt: list = []

    class _Scripted(LLMProvider):
        """Answers per role and remembers the order the roles were asked in."""

        def complete(self, messages, **kwargs):
            blob = "\n".join(m.content for m in messages)
            # Identify the caller by the role tag in its system prompt.  Take the
            # *last* occurrence: BASE_SYSTEM's own rules mention a literal
            # <agent_role> before the real tag is emitted.
            tag = "<agent_role>"
            role_name = blob.rsplit(tag, 1)[1].split("</agent_role>", 1)[0] if tag in blob else ""
            if role_name == "refine_synthesize":
                role, payload = "synthesis", {
                    "decision": "act",
                    "summary": "strengthen the existing test",
                    "acceptance": ["one assertion names the arguments"],
                    "non_goals": ["SENTINEL-NONGOAL: do not add a new test function"],
                    "blast_radius": "function",
                    "risk": "low",
                }
            elif role_name == "test_design":
                role, payload = "test_design", {"tests": []}
            else:
                role, payload = "plan_split", {"steps": [], "estimate": "XS", "rollback": "revert"}
            order.append(role)
            if not first_prompt:
                first_prompt.append(blob)
            return LLMReply(text=json.dumps(payload), model="scripted")

    state = _bare_state(
        issues=[{"id": "ISS-0001", "title": "weak assertion", "severity": "low",
                 "status": "open", "files": ["src/part_00.py"]}],
        working_code={"src/part_00.py": "value = 1\n"},
    )
    refine = make_refine_node(Crew(_Scripted()))
    refine(state)

    assert order[:3] == ["synthesis", "plan_split", "test_design"], (
        "the synthesis must land before the planning lenses"
    )
    # The synthesizer must not be constrained by its own output.  The block is
    # still present in its prompt (one template, one code path) but empty; the
    # sentinel it later emits is what the planning lenses must be able to see.
    assert "SENTINEL-NONGOAL" not in first_prompt[0], (
        "the synthesizer cannot be constrained by its own output"
    )


def test_the_planning_lenses_receive_the_synthesised_constraints():
    """Acceptance criteria and non-goals are binding on plan_split/test_design."""
    from qedloop.phases.refine import make_refine_node

    prompts: dict = {}

    class _Scripted(LLMProvider):
        def complete(self, messages, **kwargs):
            blob = "\n".join(m.content for m in messages)
            tag = "<agent_role>"
            role_name = blob.rsplit(tag, 1)[1].split("</agent_role>", 1)[0] if tag in blob else ""
            if role_name == "refine_synthesize":
                role, payload = "synthesis", {
                    "decision": "act",
                    "summary": "strengthen the existing test",
                    "acceptance": ["SENTINEL-ACCEPTANCE"],
                    "non_goals": ["SENTINEL-NONGOAL"],
                    "blast_radius": "function",
                    "risk": "low",
                }
            elif role_name == "test_design":
                role, payload = "test_design", {"tests": []}
            else:
                role, payload = "plan_split", {"steps": [], "estimate": "XS", "rollback": "revert"}
            prompts.setdefault(role, []).append(blob)
            return LLMReply(text=json.dumps(payload), model="scripted")

    state = _bare_state(
        issues=[{"id": "ISS-0001", "title": "weak assertion", "severity": "low",
                 "status": "open", "files": ["src/part_00.py"]}],
        working_code={"src/part_00.py": "value = 1\n"},
    )
    make_refine_node(Crew(_Scripted()))(state)

    for role in ("plan_split", "test_design"):
        blob = prompts[role][-1]
        assert "SENTINEL-NONGOAL" in blob, "%s must see the non-goals it is bound by" % role
        assert "SENTINEL-ACCEPTANCE" in blob, "%s must see the acceptance criteria" % role
    assert "SENTINEL-NONGOAL" not in prompts["synthesis"][0], (
        "the synthesizer cannot be constrained by its own output"
    )


def test_retired_plans_are_not_shown_to_the_reviewer():
    """Reviewers judge this round's plan, not the history of plans.

    The rejections named rows from earlier rounds ("PLN-0003, PLN-0007 and
    PLN-0009 are three separate entries for the same existing test"), so a
    retired generation leaking back into the prompt is what made a coherent plan
    look self-contradictory.
    """
    from qedloop.prompts import plans_for

    state = _bare_state(
        test_plans=[
            {"id": "PLN-0001", "issue_id": "ISS-0001", "covers": ["ISS-0001"],
             "name": "test_old", "refined_at": "T1", "superseded_at": "T2"},
            {"id": "PLN-0002", "issue_id": "ISS-0001", "covers": ["ISS-0001"],
             "name": "test_current", "refined_at": "T2"},
        ],
    )
    live = plans_for(state, "ISS-0001")
    assert [p["id"] for p in live] == ["PLN-0002"]

    state["issues"] = [{"id": "ISS-0001", "title": "t", "severity": "low", "status": "open", "files": []}]
    state["focus_ids"] = ["ISS-0001"]
    blob = "\n".join(m.content for m in build_messages(find_spec("correctness_review"), state))
    assert "test_current" in blob
    assert "test_old" not in blob, "a retired plan must not be served as if it were in force"


def test_test_plans_carry_the_path_they_will_be_written_to():
    from qedloop.crew import AgentResult

    result = AgentResult(role="test_design", phase="refine", lens="test-design", mode="produce")
    result.data = {"tests": [{"kind": "regression", "name": "test_x", "path": "tests/test_x.py",
                              "given": "g", "when": "w", "then": "t"}]}
    plan = Crew(MockProvider()).test_plans([result], issue_id="ISS-0001")[0]
    assert plan["path"] == "tests/test_x.py"


def test_the_falsification_step_reaches_the_review_and_the_patch():
    """A plan must say how the change could be shown to be what makes it pass.

    ``correctness_review`` rejected three rounds on this and no later phase can
    invent it: the patch phase only sees the requirement.  Acceptance criteria
    state what must be true afterwards; the falsification states the run that
    would fail if the change were absent, which is the only thing that tells a
    real fix from a test that passes beside it.
    """
    from qedloop.crew import AgentResult, Crew
    from qedloop.prompts import build_messages, find_spec
    from qedloop.state import initial_state

    synthesis = AgentResult(role="refine_synthesize", phase="refine", lens="requirement-synthesis", mode="produce")
    synthesis.data = {
        "decision": "act",
        "summary": "strengthen the assertion",
        "acceptance": ["SENTINEL-ACCEPTANCE"],
        "falsification": "SENTINEL-FALSIFICATION: break the forwarded width and the test must fail",
        "non_goals": [],
        "blast_radius": "function",
        "risk": "low",
    }
    finding = Crew(MockProvider()).findings([synthesis], issue_id="ISS-0001", cycle=0)[0]
    assert "SENTINEL-FALSIFICATION" in finding["falsification"], "the finding carries it"

    state = initial_state(
        {"files": [{"path": "src/part.py", "content": "value = 1\n"}]},
        target_root="/repo",
    )
    state["issues"] = [{"id": "ISS-0001", "title": "weak assertion", "severity": "low",
                        "status": "open", "files": ["src/part.py"]}]
    state["findings"] = [finding]
    state["focus_ids"] = ["ISS-0001"]
    for role in ("architecture_review", "correctness_review", "patch_generate"):
        blob = "\n".join(m.content for m in build_messages(find_spec(role), state))
        assert "SENTINEL-FALSIFICATION" in blob, "%s has to see the falsification step" % role


def test_the_plan_split_prompt_demands_falsification_steps():
    """The planner is told the falsification becomes steps, not a remark."""
    from qedloop.prompts import build_messages, find_spec

    state = _bare_state(
        issues=[{"id": "ISS-0001", "title": "t", "severity": "low", "status": "open", "files": []}],
        focus_ids=["ISS-0001"],
    )
    blob = "\n".join(m.content for m in build_messages(find_spec("plan_split"), state))
    assert "break exactly that line" in blob
    assert "revert the break" in blob


def test_the_synthesis_contract_declares_the_falsification_field():
    """The schema and the prompt live together, so they cannot drift apart."""
    from qedloop.prompts import build_messages, find_spec

    state = _bare_state(
        issues=[{"id": "ISS-0001", "title": "t", "severity": "low", "status": "open", "files": []}],
        focus_ids=["ISS-0001"],
    )
    blob = "\n".join(m.content for m in build_messages(find_spec("refine_synthesize"), state))
    assert '"falsification"' in blob


def test_lead_files_put_tests_and_small_modules_first():
    from qedloop.prompts import lead_files

    tree = {
        "tests/test_b.py": "x" * 500,
        "src/big.py": "y" * 5000,
        "src/small.py": "z" * 10,
    }
    ordered = lead_files(tree, budget=100000)
    assert "tests/test_b.py" in ordered, "the contract a test states leads"
    assert "src/small.py" in ordered, "and small modules fit whole"


def test_a_lead_test_is_paired_with_the_module_it_exercises():
    """`assert_called_once()` only looks weak next to the route it should check.

    Measured on EER-Ai: discovery's prompt held 6 test files and 3 source files,
    none of them the route the reported issue was about, and the crew answered
    that the file it needed was "listed in the manifest but its body is under
    OMITTED FOR BUDGET".
    """
    from qedloop.prompts import lead_files

    tree = {
        "tests/integration/test_thing_api.py": "def test_it():\n    assert True\n",
        "src/app/api/routes/thing.py": "def route():\n    return 1\n",
    }
    ordered = lead_files(tree, budget=100000)
    assert ordered == ["tests/integration/test_thing_api.py", "src/app/api/routes/thing.py"]


def test_pairing_prefers_the_route_over_a_same_named_schema():
    """Names collide; the smaller file is not the one under test.

    EER-Ai has both ``schemas/screenshot.py`` (421 bytes) and
    ``api/routes/screenshot.py`` (3 kB).  Taking the smallest candidate paired
    the test with the schema, which is exactly the file that does *not* forward
    the query parameters the issue was about.
    """
    from qedloop.prompts import source_pairs

    tree = {
        "tests/integration/test_screenshot_api.py": "def test_it():\n    pass\n",
        "src/app/schemas/screenshot.py": "class Request:\n    pass\n",
        "src/app/api/routes/screenshot.py": "def route():\n    return 1\n",
    }
    assert source_pairs("tests/integration/test_screenshot_api.py", tree) == [
        "src/app/api/routes/screenshot.py"
    ]


def test_empty_modules_are_never_led():
    """An empty ``__init__.py`` costs a header and teaches nothing."""
    from qedloop.prompts import lead_files

    tree = {
        "tests/test_thing.py": "def test_it():\n    pass\n",
        "src/pkg/__init__.py": "",
        "src/pkg/real.py": "value = 1\n",
    }
    ordered = lead_files(tree, budget=100000)
    assert "src/pkg/__init__.py" not in ordered
    assert "src/pkg/real.py" in ordered


# --------------------------------------------------------------------------- #
# providers
# --------------------------------------------------------------------------- #


class CountingProvider(LLMProvider):
    name = "counting"
    model = "count-1"
    concurrency = 1

    def __init__(self):
        self.calls = 0

    def complete(self, messages, **kw):
        self.calls += 1
        return LLMReply(text='{"ok": true}', model=self.model, prompt_tokens=10, completion_tokens=5)


def test_caching_provider_memoises_identical_prompts():
    inner = CountingProvider()
    cached = CachingProvider(inner)
    messages = [Message("user", "same")]
    first = cached.complete(messages)
    second = cached.complete(messages)
    assert inner.calls == 1
    assert first.text == second.text and second.cached is True
    assert cached.name == "counting+cache" and cached.model == "count-1"
    cached.complete([Message("user", "different")])
    assert inner.calls == 2


def test_recording_provider_replays_by_match():
    provider = RecordingProvider([{"match": "review", "reply": '{"verdict":"approve"}'}], default='{"verdict":"abstain"}')
    assert provider.complete([Message("user", "please review this")]).text == '{"verdict":"approve"}'
    assert provider.complete([Message("user", "something else")]).text == '{"verdict":"abstain"}'


def test_auto_provider_falls_back_to_mock_without_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert isinstance(make_provider("auto"), MockProvider)


def test_unknown_provider_is_rejected():
    with pytest.raises(Exception):
        make_provider("nope")


def test_mock_provider_reports_declared_markers_only():
    tree = {
        "src/mod.py": "def f():\n    return 1  # BUG: demo/marker -- returns the wrong thing\n",
        "src/clean.py": "def g():\n    return 2\n",
    }
    state = {"cycle": 0, "working_code": tree, "issues": []}
    payload = extract_json(MockProvider().complete(build_messages(find_spec("discover_archaeology"), state)).text)
    assert len(payload["issues"]) == 1
    assert payload["issues"][0]["bug_marker"] == "demo/marker"

    clean_state = {"cycle": 0, "working_code": {"src/clean.py": tree["src/clean.py"]}, "issues": []}
    clean = extract_json(MockProvider().complete(build_messages(find_spec("discover_archaeology"), clean_state)).text)
    assert clean["issues"] == [], "the mock must not invent defects it cannot see"


def test_mock_provider_refuses_to_patch_without_a_marker():
    state = {"cycle": 0, "working_code": {"src/clean.py": "x = 1\n"}, "issues": [{"id": "ISS-0001"}]}
    payload = extract_json(MockProvider().complete(build_messages(find_spec("patch_generate"), state)).text)
    assert payload["ops"] == []


# --------------------------------------------------------------------------- #
# crew
# --------------------------------------------------------------------------- #


def test_crew_dispatches_three_agents_per_phase_and_keeps_order():
    crew = Crew(MockProvider())
    state = {"cycle": 0, "working_code": {"a.py": "x = 1\n"}, "issues": []}
    results = crew.run_phase("discover", state)
    assert [r.role for r in results] == [s.role for s in specs_for("discover")]
    assert all(r.ok for r in results)


def test_crew_repairs_a_non_json_reply_once():
    class BadThenGood(LLMProvider):
        name = "bad-then-good"
        model = "x"
        concurrency = 1

        def __init__(self):
            self.calls = 0

        def complete(self, messages, **kw):
            self.calls += 1
            if self.calls == 1:
                return LLMReply(text="sorry, I cannot do that")
            return LLMReply(text='{"verdict": "approve"}')

    provider = BadThenGood()
    crew = Crew(provider, max_retries=2)
    result = crew.run_one(AgentTask(find_spec("architecture_review"), {"working_code": {}}))
    assert result.ok and result.data["verdict"] == "approve"
    assert result.attempts == 2
    assert provider.calls == 2


def test_crew_survives_a_provider_error_without_raising():
    class Broken(LLMProvider):
        name = "broken"
        model = "x"
        concurrency = 1

        def complete(self, messages, **kw):
            from qedloop.llm import LLMError

            raise LLMError("network down")

    crew = Crew(Broken(), max_retries=1)
    result = crew.run_one(AgentTask(find_spec("static_qa"), {"working_code": {}}))
    assert not result.ok and "network down" in result.error


def test_crew_normalises_review_rows_and_votes():
    crew = Crew(MockProvider())
    state = {"cycle": 1, "working_code": {"a.py": "x = 1\n"}, "issues": [], "reviews": [], "evidence": {}}
    results = crew.run_phase("review", state, focus={"id": "ISS-0001", "title": "t"})
    rows = crew.reviews(results, cycle=1, issue_id="ISS-0001")
    assert len(rows) == 3
    assert {r["lens"] for r in rows} == {"architecture", "correctness", "risk-and-regression"}
    assert all(r["issue_id"] == "ISS-0001" for r in rows)
    assert crew.votes(results)["architecture_review"] == "approve"
