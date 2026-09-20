# Architecture — an AI-driven, self-iterating development loop

This document explains how the five phases are wired, what each module owns, and
where the design deliberately differs from a plain agent chain. Read it together
with `phases.md` (the per-phase contracts) and the code, which is organised to
match this text one-to-one.

---

## 1. The shape of the system

```
┌─────────────────────────────────────────────────────────────┐
│                  状态管理总线 (StateGraph)                    │
│   codebase / issues / findings / reviews / changes / quality │
└──────────┬────────────────────────────────────┬─────────────┘
           │                                    │
    ┌──────▼──────┐                      ┌──────▼──────┐
    │  Phase 1    │                      │  Phase 5    │
    │  discover   │◄─────────────────────│  qa         │
    │  (3 agents) │   next cycle         │  (3 agents) │
    └──────┬──────┘                      └──────▲──────┘
           │                                    │
    ┌──────▼──────┐                      ┌──────┴──────┐
    │  Phase 2    │                      │  Phase 4    │
    │  refine     │                      │  patch      │
    │  (3 agents) │                      │  (3 agents) │
    └──────┬──────┘                      └──────▲──────┘
           │                                    │
    ┌──────▼──────┐   approve            ┌──────┴──────┐
    │  Phase 3    ├─────────────────────►│  converge?  │
    │  review     │◄─────┐  reject       │  quality    │
    │  (3 agents) │      └───────────────┤  gate       │
    └─────────────┘   back to refine     └─────────────┘
```

Three loops exist, and each one has a different guard:

| loop | edge | guard |
| --- | --- | --- |
| plan revision | `review → refine` | `gate.max_self_loops` (default 3), then the run stops for a human |
| defect iteration | `qa → discover` | `run.max_iterations` (default 3) |
| runaway protection | any | `CompiledGraph(max_steps)` + per-node `max_visits` |

There is no hidden state outside the bus: if a fact is not on a channel, it does
not exist for the next phase.

---

## 2. Why a state bus rather than message passing

An agent chain passes prose from one model to the next and loses structure at
every hop. The bus keeps every fact as a typed ledger row, and each phase reads
only the channels it needs:

| channel | written by | read by |
| --- | --- | --- |
| `codebase` / `codebase_files` | orchestrator (frozen at run start) | every phase |
| `working_code` | phase 4 | phases 1, 3, 4, 5 |
| `issues` | phase 1 (accumulates) | phases 2–5, prompts |
| `findings` / `test_plans` | phase 2 | phases 3, 4, 5（计划只取「在服役」的那一代） |
| `reviews` | phase 3 | phase 4 gate, scoring |
| `patches` / `changes` | phase 4 | phase 5, report, `run.py apply` |
| `evidence` | phase 5 (measured) | phase 5 agents, scoring |
| `quality_history` / `cycles` | phase 5 | convergence, report |
| `status` / `status_reason` | phase 1, 3, 4, 5 | graph routing, CLI exit code |

Two consequences worth stating explicitly:

1. **A report can be produced from the bus alone.** `state.json` +
   `candidate/` + `trace.jsonl` is the whole run.
2. **A phase can be unit tested without a model.** Feed it a state dict, assert
   on the returned delta; `tests/test_policy.py` and `tests/test_loop_e2e.py`
   do exactly this.

### Reducers

Channels are not a bag of fields; each one declares how a write combines with
what is already there (see `qedloop/state.py:CHANNELS`):

| reducer | behaviour | used by |
| --- | --- | --- |
| replace | the delta becomes the value | `codebase`, `working_code`, `status`, counters |
| `merge_ledger("issues")` | merge by `id`; recompute `seen_count`; never duplicate | `issues`, `issue_candidates` |
| `merge_ledger("findings")` | one row per issue; a re-refinement supersedes and versions the old one | `findings` |
| `merge_ledger(<other>)` | append-only audit trail | `reviews`, `patches`, `changes`, `checks`, `cycles` |
| `merge_dict` | shallow union, later wins | `evidence` |

`seen_count` is recomputed from the *stored* row, never taken from the delta —
a node that forgets a field must not be able to reset a counter.

---

## 3. Module map

| module | responsibility | notable design choice |
| --- | --- | --- |
| `graph.py` | `StateGraph` / `CompiledGraph`: nodes, static and conditional edges, step and visit guards, event stream, Mermaid export | mirrors LangGraph's vocabulary so the mental model transfers; ~250 lines, no dependencies |
| `state.py` | channel registry, reducers, `initial_state` | every write goes through a reducer; unknown channels are rejected |
| `core.py` | records (`Issue`, `Finding`, `Review`, `ChangeProposal`, …), id allocation, marker scanning and retraction, diffs | records are dataclasses at the edges and plain dicts on the bus, so the state stays JSON-serialisable |
| `llm.py` | provider **channels**: OpenAI-compatible HTTP (stdlib), DeepSeek, deterministic mock, replay, prompt cache | a channel is a named endpoint (base_url / model / key env / timeout); `--provider <name>` selects one, so a gateway, a local server and a hosted API differ only in configuration — no vendor SDK, and `auto` falls back to the mock so the framework always runs |
| `prompts.py` | 15 agent specs (lens, mode, JSON contract) and the prompt builders | the contract and the prompt live together, so changing an agent never touches graph wiring |
| `crew.py` | fan-out, tolerant JSON extraction, normalisation into ledger rows | one repair round-trip on malformed JSON, then a failed `AgentResult` instead of an exception |
| `sandbox.py` | real pytest runs on a candidate tree, static analysis, anchored-edit application, diffing | measurement is separated from interpretation |
| `policy.py` | the two gates and the quality score | pure functions: replayable from a trace, testable without an LLM |
| `phases/*.py` | the five nodes plus shared helpers | a node is `state -> delta`; it may call models and commands, and must not mutate the state |
| `orchestrator.py` | run directory layout, tracing, artifact writing, applying a verified run | the only module that knows about all the others |
| `report.py`, `cli.py` | report rendering, config resolution, commands | the report leads with the decision and the evidence |

---

## 4. The five phases

Each phase runs three agents with non-overlapping lenses. `mode` decides what an
answer is worth:

| mode | meaning | roles |
| --- | --- | --- |
| `note` | contributes information, never gates | discovery lenses, `risk_review`, `static_qa` |
| `vote` | counts towards a gate (`approve` / `reject` / `abstain`) | `architecture_review`, `correctness_review`, `test_sandbox`, `edge_case_qa` |
| `produce` | returns structured content (a requirement, plans, a patch) | refine and patch roles |
| `synthesize` | adjudicates between produced candidates | `patch_reconcile` |

A full description of every phase, its inputs, outputs and failure modes is in
`phases.md`.

---

## 5. The three decisions that matter

### 5.1 What work is a cycle about? (Phase 1)

Discovery returns up to nine reports (three lenses × three issues). Two filters
turn that into one unit of work:

1. **Identity de-duplication** (`discover.issue_signature`). A declared marker
   is the defect's identity; without one, the normalised title plus first file is
   the proxy. Reports of the same defect merge into the canonical ledger row,
   which records every lens that saw it. Independent agreement raises confidence
   and the strongest severity wins.
2. **Target selection** (`discover._pick_target`). Severity, then confidence,
   then id — one issue per cycle.

One issue per cycle is what makes the rest of the pipeline honest: a review
verdict, a patch and a test result all belong to exactly one defect, so they can
be attributed and replayed. It also keeps a failing patch from blocking an
unrelated fix. Other findings stay in the ledger as `open` and become later
cycles, so a discovery burst is never wasted.

If nothing new was found *and* nothing is open, the run reports a fixed point
instead of looping over the same tree.

### 5.2 May this plan be implemented? (Phase 3)

`policy.review_route` is deterministic:

- any `reject` → back to Phase 2 (a voiced objection always beats a bare
  approval — silence is not consent);
- `approve`s ≥ `min_approvals` → Phase 4;
- otherwise (`abstain`s only) → back to Phase 2, because the plan lacked the
  information reviewers needed;
- rejections past `max_self_loops` → stop and hand to a human (`human_review`).

### 5.3 Is the result good enough? (Phase 5)

Measurement happens **before** interpretation:

1. `sandbox.verification_snapshot` runs the target's own suite twice — against
   the frozen baseline and against the candidate tree — and compiles the
   candidate tree. This is the only source of test truth in the run.
2. The three QA agents interpret that evidence; the mock provider and a real
   model both read the same numbers. An agent can object; none of them can claim
   a green suite that was not measured.
3. `policy.score_cycle` blends evidence into 0–1, and `policy.convergence`
   decides:

| status | condition | exit code |
| --- | --- | --- |
| `converged` | suite green **and** ≥ `min_approvals` **and** quality ≥ `quality_target` **and** ≥ `min_iterations` cycles completed | 0 |
| `continue` | budget remains and progress is plausible | 0 |
| `human_review` | no progress for `max_no_progress` cycles, or single-cycle mode | 1 |
| `escalated` | iteration budget exhausted | 3 |
| `error` | an exception escaped a node (partial run is still reported) | 4 |

Quality is a weighted blend of the two *measured* components — tests 0.5, static 0.2 —
and reviewer verdicts are deliberately not one of them. Three hard requirements sit
*outside* the score, so a high score can never paper over them: the candidate tree
must **compile**, the suite must be **green**, and the plan must have
**≥ `min_approvals`** approvals. (The compile requirement was added
after a real find: a probe repository written with a UTF-8 BOM failed `compile()`,
Phase 5 correctly refused to credit the fix, and the gate still reported
`converged`. Both halves of that are fixed — see the BOM note below.)

Reviewer votes used to be the third weighted component (0.3) and were removed:
whether the plan cleared the gate is *already* a hard requirement, so weighting it
as well counted the same fact twice — and the weight kept moving after the gate had
approved. Measured on one run: 12 review rows across four rounds (6 approve /
6 reject), the Phase 3 gate approved 3/3 on the last round, the patch applied
cleanly and the suite went 503 → **507 passed**, yet the report read `review 0.00`
with quality 0.70, because the score divided the whole append-only ledger
(`(6-6)/12`) instead of the round that mattered; a single-cycle run handed a
success over as a failure. Scoring only the final round would not have fixed it:
`review_route` returns `refine` whenever a rejection is voiced, so any cycle that
reaches Phase 4 has zero rejections in that round and the component is 1.0 by
construction — a constant that only dilutes the measured evidence.

**Progress is "an issue closed", not "the score moved".** A repository with
three defects can move the blended score very little while each cycle genuinely
retires one, so the plateau rule counts cycles that closed nothing *and* barely
moved quality. Without that distinction the loop hands off to a human in the
middle of real progress (this bug was caught by the end-to-end test).

`min_iterations` belongs to the blocker list for the same reason as everything
else, and it was missing from it — because the cycle **index** was being compared
against a cycle **count**. `state["cycle"]` is the 0-based index of the cycle that
just finished (`qa.py` increments it *after* the gate) while `min_iterations`
counts cycles, exactly as the report's "cycles completed" does. So the shipped
default of 1 silently meant "at least two cycles", contradicting the config
comment *"never declare victory before this many cycles"*. Measured on a real run:
one clean cycle with quality **1.00**, a green suite, five reviewer approvals and
all three QA lenses approving, handed over with the reason
`single-cycle mode stopped the loop; ` and nothing after the semicolon. Comparing
counts fixes it, and putting the shortfall in `block_reasons` makes "no blockers"
and "converged" the same statement — so a non-converged branch always has at
least one reason to print.

---

## 6. How a fix is verified

This is the part worth reading twice, because it is where an autonomous loop
usually lies to itself.

Phase 5 credits a fix only when two independent measurements agree:

1. **The declared defect is gone.** The issue reported a `# BUG: name` marker;
   after the patch that marker is no longer in the candidate tree. A marker
   *asserts* a defect, so `patch` retracts the declaration of the defect it fixed
   (`core.strip_marker`) — otherwise the next pass would read the old claim and
   conclude the defect is still there. Retraction removes comments only, and
   handles both shapes: a marker on its own line takes its continuation comment
   block, a marker trailing real code strips only the comment.
2. **The suite made real progress.** At least one previously failing test now
   passes, and the patch introduced no new failure.

Plus, in both cases: the candidate tree still compiles.

What is deliberately **not** required is a fully green suite before *any* fix is
credited. With several independent defects in one repository that demand is
self-defeating: the loop would keep re-patching the already-fixed file until the
budget ran out. Final convergence still requires a green suite.

Issues with no declared marker need a green suite instead; `report.py` records
which case applied, so the weaker evidence is visible in the report rather than
implied.

### Why the demonstration fixture is built this way

`examples/buggy_service` ships with three injected defects, each declared by a
marker, each covered by a test that fails on the shipped code. That gives the
loop something a mock provider can honestly work with:

- **evidence**: the suite really goes red → green, measured by pytest;
- **attribution**: each marker is one defect, so "which issue did this patch
  fix?" has an answer;
- **non-fabrication**: the mock provider only reacts to markers it can see. Ask it
  to patch a file with no marker and it returns `ops: []` plus a reason — the
  framework then reports "the crew could not express a fix" instead of inventing
  one.

---

## 7. Providers and offline operation

`make_provider("auto")` uses the first *auto-selectable* ready channel and falls
back to the deterministic mock, so:

```powershell
python run.py run --target examples/buggy_service --provider mock   # always works
python run.py channels --config my.yml --probe                      # verify endpoints first
python run.py run --target ./myrepo --config my.yml --provider company-gateway
```

### Channels

A **channel** is a named endpoint: `kind` (openai-compat / openai / deepseek /
mock), `base_url`, `model`, `api_key_env`, `timeout`, `temperature`, `cache`.
They are declared in a config file's `providers:` section and selected by name,
with per-field precedence **explicit flag > channel > environment > built-in
default**. Anything that speaks `POST {base_url}/chat/completions` works with no
code change, which covers corporate gateways, aggregators and local servers.

Three deliberate properties:

* **keys are never printed** — `ProviderChannel.to_dict()` exposes only
  `key_present`, and `channels` shows the env var name, not the value;
* **`auto` will not dial a local preset** — built-in `ollama`/`lmstudio` are
  marked `auto_select: false`, because on a fresh machine that means hanging on a
  dead port instead of falling back to the mock; a channel *you* declare is fair
  game;
* **`timeout=None` means "unset", not "90 s"** — otherwise a channel that asks
  for 300 s for a slow local model would be silently given the default.

The mock is not a stub that returns canned JSON for every prompt: it reads the
files it is given, finds marker comments, and consults an explicit playbook
(`llm.MOCK_PLAYBOOK`) of anchored edits, matching a defect by full marker name or
by its last segment. It proves the plumbing — fan-out, reducers, gates,
measurement, reporting, applying — and it is honest about its own limits: for a
defect shape it has no recipe for it returns `ops: []` with a stated reason.
Real reasoning comes from a real model; nothing about the framework changes when
you switch.

`CachingProvider` memoises by prompt fingerprint, which makes reruns cheap and
makes a fixpoint cycle (same tree, same prompt) cost nothing.

---

## 8. Failure handling

| failure | behaviour |
| --- | --- |
| malformed JSON from an agent | one repair round-trip quoting the parser error, then the agent is marked failed and the phase continues |
| provider error / no key | retry with backoff for transport errors; 4xx (except 408/429) fails fast; the node reports the failure instead of raising |
| patch anchor missing or ambiguous | `apply_ops` is all-or-nothing: no partial apply, the change is recorded as failed with the reason, the cycle ends in `human_review` |
| patch makes the tree un-compilable | static analysis fails the cycle, the score's static component is 0, QA rejects |
| test suite cannot run | evidence records `ran: false` with the reason; with `gate.require_tests` (default) the run cannot converge |
| source file carries a UTF-8 BOM | `core.decode_source` reads with `utf-8-sig` and strips a leading `\ufeff`, so a Windows-written file is not mistaken for a broken one |
| runaway loop | `max_steps` and per-node `max_visits` raise `StepLimitExceeded`, which the orchestrator turns into `escalated` with the partial run preserved |
| target repository changed since the run | `run.py apply` compares the recorded baseline hash per file and refuses to overwrite |

One more rule is worth naming: **the loop never writes to the target while it is
thinking.** Phase 4 edits a candidate tree held in the state bus; Phase 5
verifies that candidate; only `run.py apply` (or an explicit call) writes to
disk, and only for a `converged` run, with `.qedloop.bak` backups.

---

## 9. Working on a real-sized repository

A repository with a hundred-plus modules does not fit in a prompt, and every
mechanism below exists because of a failure mode observed on one (`EER-Ai`: 170
files, 502 tests, currently green).

**Prompts carry a manifest, not an illusion of completeness.** Each prompt
starts with a capped MANIFEST of the tree — files in play and test files get
first claim on the budget — then spends its body budget on whole files, clipping
large ones head-and-tail. Everything not shown is named, and the system prompt
tells the agent to lower its `confidence` when a claim rests on code it could not
read. Silently showing the first N files is how a crew reports defects in files
it never opened.

**The files a change touches are led into the prompt, in every phase that judges
it.** A budget spent alphabetically goes to whatever sorts first, not to what the
task is about: measured here, 136 of 145 files were elided and the two the issue
named were both among them. Refine and review then had to certify parameter names
they were structurally forbidden from reading, and every review round refused for
exactly that reason while `patch_prompt` — which did lead with those files — could
have answered it. `prompts.focus_files` is the single place that decides which
paths get first claim, so the phases cannot drift apart on it.

**Discovery leads with a *paired* sample of the tree, not with its smallest
files.** `lead_files` walks the test directories round-robin so every package is
represented, and admits each test together with the module it exercises
(`tests/integration/test_x_api.py` → `api/routes/x.py`); the rest of the budget
goes to the smallest real modules. Both rules come from measurements on EER-Ai:

| selection | files shown | test dirs | the route the crew asked for |
| --- | --- | --- | --- |
| alphabetical tests, no pairing | 9 (6 tests / 3 source) | 2 | not shown |
| round-robin tests, paired | 37 (8 tests / 29 source) | 4 | shown when its test is |

Two traps are recorded here because both cost a run. Pairing by "smallest
candidate wins" matched `tests/integration/test_screenshot_api.py` with
`schemas/screenshot.py` (421 bytes) instead of `api/routes/screenshot.py` (3 kB) —
the schema is the file that does *not* forward the parameters under test — so
candidates are ranked by a name hint from the test's own filename and by the
`routes/`/`api/` convention before size. And a walk that *stops* at the first
candidate that does not fit spends its whole budget on nine files while leaving
room for twenty-eight: skipping and continuing is the difference between 9 and 37.
No mechanism here changes how much of a 145-file tree fits in 24 kB, and the
budget line notes what that leaves out.

**The measurement happens in a copy of the repository, not in a bare temp dir.**
`run_pytest(..., target_root=...)` copies the tree (minus VCS metadata, caches
and virtualenvs) and overlays the candidate files. Test suites read images, JSON
fixtures, templates and frontend bundles; a sandbox that receives only `.py`
files reports failures that do not exist. The first version of this sandbox
produced 13 phantom failures on a repository whose own `pytest` run was green.

**A copy is only honest if the copy is what gets imported.** The same repository
is also importable from the ambient environment — an editable install, or a
`.pth` file pointing at the original `src/` checkout. The package of a `src/`
layout is not at the repository root, so a sandbox that puts only its own root on
`sys.path` imports the *original* tree: the candidate overlay is never executed
and every patch measures as the unmodified repository, which is a number that
cannot fail. `run_pytest` therefore puts `<sandbox>` **and** `<sandbox>/src`
ahead of everything the environment already provides, and gives the measured
suite its own `PYTEST_DEBUG_TEMPROOT` so it cannot fight the caller's pytest over
`pytest-of-<user>/pytest-current`.

**The baseline is measured once, before any agent runs**, and recorded on the
bus (`baseline_tests`) and in the report. Discovery then reasons from a fact —
"the suite is green, find what it does not cover" or "these 13 tests fail, start
there" — instead of guessing, and Phase 5 has a like-for-like reference.

**Scope is a first-class input.** `--include` / `--exclude` narrow what counts as
the candidate tree, which is the difference between a prompt that buys breadth
and one that buys depth.

---

## 10. Extension points

| to change | do this |
| --- | --- |
| add a lens to a phase | append an `AgentSpec` to `prompts.AGENTS`; the phase and the graph pick it up automatically |
| change what an agent must return | edit its `keys` and its prompt builder together, in `prompts.py` |
| add a state channel | add it to `state.CHANNELS` with a reducer, and to `initial_state`; unknown channels are rejected at write time |
| change a gate | edit `policy.py` — it is pure functions, and `tests/test_policy.py` will tell you what you broke |
| state rules a repository must follow | point `run.brief` at a file: `prompts.build_messages` injects it into *every* agent's system message, labelled as rules rather than evidence |
| use another LLM vendor | add a class in `llm.py` and register it in `PROVIDERS`; anything OpenAI-compatible needs no code at all |
| verify something other than pytest | call the target's runner from `sandbox.run_pytest`'s contract (`SandboxReport`) and keep `verification_snapshot`'s shape |
| persist runs across processes | add a checkpointer around `CompiledGraph.stream`: the state is already JSON-serialisable, so this is additive |

---

## 11. Complexity and cost

For one cycle with the mock provider: 15 agent calls (3 discovery + 3 refine + 3
review + 3 patch + 3 QA) plus the internal reconciliation call in Phase 4, and
two pytest invocations. Measured on `examples/buggy_service`: three cycles to
clear three defects, 106 tests in the framework's own suite, a few seconds
end-to-end.

Real-model cost is dominated by the code payload in every prompt. Three knobs
keep it bounded: `render_codebase(..., budget=…)` clips the tree per phase, the
prompt cache makes repeat prompts free, and `run.token_budget` stops the crew
from spending past a ceiling.

---

## 12. Known limits

- **Single process, single target.** `run.py apply` is a file copy; there is no
  git integration, no branching, no PR creation.
- **The candidate tree is produced by anchored search/replace, not by a real
  patch tool.** It is deterministic and validated, but it cannot express a change
  that has no unique anchor, and it cannot create a new file.
- **Success depends on the repository's tests.** A defect no test can observe
  will be reported, patched and then rejected by QA — correctly, but unhelpfully.
  Feeding discovery a target with a meaningful suite is the user's job.
- **Marker-based attribution.** A reviewer looking at a report can see which
  signal credited a fix; they cannot see a proof that the defect is gone beyond
  "the marker was retracted and the suite improved". A stronger signal would
  execute a per-issue acceptance test rather than a shared suite.
- **No cross-run memory.** Each run starts from the tree on disk; the ledger is
  not persisted between runs, so two runs against the same repository re-derive
  the same issues.
- **JSON-only agent contracts.** Structured output is parsed from prose with a
  tolerant extractor rather than constrained decoding.
