# QEDLoop

**A self-iterating software development loop.** *quod erat demonstrandum* -- "which was to be demonstrated". The name is the rule: the loop may not claim convergence until the demonstration exists.

> 简体中文版：[README.zh-CN.md](README.zh-CN.md)

An AI-driven closed loop over a real repository: five phases, three agents each,
one shared state bus, and a convergence gate that is decided by *measured*
evidence rather than by a model's opinion.

```
discover ──► refine ──► review ──► patch ──► qa ──┐
    ▲           ▲          │                      │
    │           └─ reject ─┘                      │  quality < target -> next cycle
    └─────────────────────────────────────────────┘
       converged / escalated / human_review
```

Both back edges are bounded: `reject` returns to Phase 2, a quality miss starts a new cycle at Phase 1 (guards in the loops table of `docs/architecture.md`).

* **Phase 1 discover** — three lenses find new defects; reports of one defect
  collapse into a single ledger row; the cycle takes exactly one target.
* **Phase 2 refine** — the defect becomes a falsifiable requirement, an ordered
  plan and a test design.
* **Phase 3 review** — architecture and correctness reviewers vote, risk
  assesses blast radius; a rejection loops back to refinement, bounded.
* **Phase 4 patch** — two competing patches are reconciled, the winner is applied
  to a *candidate tree* by validated anchored edits.
* **Phase 5 qa** — the target's own test suite runs against baseline and
  candidate; three agents interpret the numbers; the gate converges, iterates or
  hands off to a human.

Everything is Python 3.9+ with **zero required dependencies** (pytest only to
verify a candidate tree). The whole loop runs offline against a deterministic
mock provider, so you can watch it work before spending a token.

---

## 60-second tour

```powershell
cd QEDLoop

# 1. what is in the demo target?
python run.py check --target examples/buggy_service

# 2. run the loop offline (deterministic, no API key)
python run.py run --target examples/buggy_service --provider mock

# 3. read the result
start runs\<run_id>\REPORT.md

# 4. write the verified fix onto the target, with backups
python run.py apply --run runs\<run_id> --dry-run
python run.py apply --run runs\<run_id>
```

The demo target ships with three injected defects and a failing suite. The mock
run finds all three, fixes one per cycle, and converges only when the suite is
green — `quality 1.00 >= target 0.80 with a green suite and 9 approvals`.

Use a real model by pointing the framework at any OpenAI-compatible endpoint.
Channels are declared once and selected by name, so a corporate gateway, a local
server and a hosted API are all just a `--provider <name>` away:

```powershell
python run.py channels                                  # what is available, and where each one points
python run.py channels --config my.yml --probe          # verify connectivity with one round-trip
python run.py run --target ./myrepo --provider deepseek --model deepseek-chat
python run.py run --target ./myrepo --config my.yml --provider company-gateway
```

```yaml
# my.yml
providers:
  company-gateway:
    kind: openai-compat                 # any POST {base_url}/chat/completions service
    base_url: https://llm.corp.example.com/v1
    model: gpt-4o-mini
    api_key_env: CORP_LLM_KEY           # the key comes from the environment, never this file
  local-ollama:
    kind: openai-compat
    base_url: http://127.0.0.1:11434/v1
    model: qwen2.5-coder:14b
    api_key_env: ""                     # an empty value means "needs no key"
    timeout: 300                        # local models are slow; do not time out early
```

Per-field precedence is **explicit flag > channel config > environment >
built-in default**, so `--model`, `--base-url`, `--api-key` and `--llm-timeout`
can override any channel at the command line. Environment variables:
`LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`, `OPENAI_API_KEY`, `LLM_API_KEY`,
`DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`. `--provider auto` (the default) picks
the first *auto-selectable* ready channel and falls back to the mock, so the
command never fails for lack of configuration; built-in presets for local
servers are excluded from `auto` so a fresh machine does not hang on a dead port.
Keys are never printed — only whether one was found and where it came from.

A Chinese walkthrough of the same material is in [README.zh-CN.md](README.zh-CN.md).

---

## Commands

| command | purpose |
| --- | --- |
| `run.py run --target <dir>` | run the loop; artifacts land in `runs/<run_id>/` |
| `run.py apply --run <dir>` | write a **converged** run's verified tree onto the target (`.qedloop.bak` per file, `--dry-run` to preview) |
| `run.py channels [--config <file>] [--probe]` | list model channels and their endpoints; `--probe` verifies each with one round-trip |
| `run.py agents` | list all 15 agent roles, their lens, mode and purpose |
| `run.py check --target <dir>` | baseline diagnostics: files, declared defect markers, whether the suite is red |

Useful flags: `--config file.yml`, `--provider`, `--model`, `--base-url`,
`--api-key`, `--llm-timeout`, `--max-iterations`, `--quality-target`,
`--min-approvals`, `--max-self-loops`, `--single-cycle`, `--no-tests`,
`--token-budget`, `--include/--exclude`, `--quiet`. See
`examples/loop.buggy_service.yml` for the file format, including a set of
ready-made channels.

### Exit codes

| status | code | meaning |
| --- | --- | --- |
| `converged` | 0 | green suite, approved plan, quality ≥ target |
| `continue` | 0 | budget remains; the report explains what is still open |
| `human_review` | 1 | no progress, or single-cycle mode — a person should look |
| `escalated` | 3 | iteration budget exhausted |
| `error` | 4 | an exception escaped a node; the partial run is still reported |

### Watching a run, and stopping one

Without `--quiet`, every phase node and every agent reports itself **line by line,
as it happens** (real output of
`python run.py run --target examples/buggy_service --provider mock`):

```
[run  ] start    20260920-004544-0001
[run  ] baseline measuring: copying the target repository and running its suite
[run  ] baseline done: 4 passed, 4 failed, 0 errors (ran=True)
[node ] discover  start  step=1  cycle=0
[agent] discover  agent.done             discover_archaeology abstain       0 tok     0.0ms
[agent] discover  phase1.found           new_rows=3  issues=3  deduped=3
[node ] discover  done   step=1  cycle=0  ok       4.8ms -> refine    issue_candidates=3 items ...
[agent] review    phase3.verdict         decision=approve  reason=every reviewed issue cleared the gate  votes={"architecture": "approve", ...
```

Those two baseline lines are not decoration: the first thing a run does is copy the
whole target repository and execute its suite, which is **minutes of silence** on a
real repository -- and silence is what makes an operator press `Ctrl+C` on a run
that is working fine.

`quiet` (`run.quiet: true` in a config) suppresses exactly that live view, not the
run: the result summary is still printed at the end. `--quiet` on the command line
is the same switch.

`trace.jsonl` holds the same events with every field (the full text of an
objection, all three review votes, each agent's tokens and latency). A second
window can follow it live:

```powershell
# `runs\<run_id>\trace.jsonl` cannot be pasted as-is: `<` and `>` are illegal in a
# Windows path (PowerShell reads them as redirection), and it fails with
# "路径中具有非法字符" / "the path has invalid characters". This finds the trace file
# being written right now instead -- the live run always has the newest mtime:
$run = Get-ChildItem runs -Recurse -Depth 2 -Filter trace.jsonl | Sort-Object LastWriteTime | Select-Object -Last 1
Get-Content $run.FullName -Wait -Tail 20
```

(`-Depth 2` covers both layouts, `runs\<run_id>\` and `runs\<target id>\<run_id>\`;
`-Tail 20` replays the last 20 events before following. `Ctrl+C` leaves the reader
without touching the run.)

**Stopping:** press `Ctrl+C`. The current node stops but the run is not lost -- it is
recorded as `human_review` with the reason `interrupted by the operator after N
node(s)`, `state.json` and `REPORT.md` are written as usual, and the exit code is 1.
An interrupted run deliberately gets **no candidate tree**, so `apply` refuses it
(including with `--allow-unverified`): being verified is what makes a tree writable,
and an interrupt is exactly the case where nothing was verified.

Two measured details: an interrupt that lands between **model calls** is delivered
within 2.0s (the trace keeps a `refine/start` with no matching `done`); one that
lands while the suite is being measured waits for that pytest wait to return on
Windows (measured: the child ran its full 20s, the parent returned at 20.6s) -- the
outcome above is unchanged, it is just slow. A second `Ctrl+C` kills the process
outright, at which point only `trace.jsonl` remains readable.

---

## Using it on your own repository

**You do not need to move the repository into `QEDLoop`.** `--target`
accepts any path:

```powershell
# absolute path: works from any directory
python run.py check --target D:\work\myrepo

# relative path: resolved against your *current* directory, not QEDLoop
cd D:\work
python D:\work\QEDLoop\run.py check --target myrepo

# or run from inside the repository itself
cd D:\work\myrepo
python D:\work\QEDLoop\run.py check --target .
```

Run artifacts are written to `runs/` under the **current working directory**
(override with `--out`), so the target repository is never polluted.

**Per-target settings live here, not in the target.** Anything specific to one
repository — its config, and the constraints its agents work under — belongs in
`targets/<target-id>/` inside this project, so a run leaves the target
repository's `git status` untouched:

```
targets/
  README.md       the convention
  eer-ai/
    loop.yml      python run.py run --config targets\eer-ai\loop.yml
    brief.md      injected into every agent (loop.yml's brief: points at it)
```

`brief.md` carries the rules that code alone does not reveal: generated files
that must not be edited, dependencies that must not be added, the migration
steps a schema change requires. It is sent as a `<project_constraints>` section
in **every** agent's system message (capped at 4000 characters) and is labelled
as rules, *not* evidence. A relative path resolves against the config file that
names it; an unreadable brief is a hard error rather than a silent "no
constraints". `--brief <path>` sets it for one run.

> `brief.md` feeds **the loop's 15 agents**. Conventions for **your own coding
> agent** (Claude Code and friends) are a separate file: the repository ships a
> template, `CLAUDE.md.sample`. Copy it to `CLAUDE.md` and fill it in -- it
> explains how to write itself, and tells the agent to prompt you when the
> placeholders are still unfilled. `CLAUDE.md` is in `.gitignore`: it is a local
> file.

```powershell
python run.py check --target D:\work\myrepo                      # is there anything measurable?
python run.py run --target D:\work\myrepo --provider mock --single-cycle   # discovery only, writes nothing
python run.py run --target D:\work\myrepo --provider deepseek --model deepseek-chat
python run.py apply --run runs\<run_id> --dry-run                # preview the accepted fix
python run.py apply --run runs\<run_id> --target D:\work\myrepo  # write it (one .qedloop.bak per file)
```

`apply` defaults to the target recorded in the run's `state.json` (an absolute
path), so it does not matter which directory you invoke it from.

### What the offline mock can fix

`--provider mock` does not reason: it looks up a recipe by **defect shape**. A
marker name matches by full name first, then by its last segment
(`probe/mean_zero` → `mean_zero` → the `stats/mean_zero` recipe), and the file
comes from wherever the marker was actually reported — so it works in a
different layout. For a shape it has no recipe for, it returns `ops: []` with a
stated reason and the run ends in `human_review`: **it will not invent a patch.**
To teach it more, add a `{search, replace}` entry to `MOCK_PLAYBOOK` in
`qedloop/llm.py`. For real reasoning, use a real provider.

### What the target repository needs

1. **Tests pytest can discover** (`tests/test_*.py` or `test_*.py`). A repository
   without tests still runs, but it cannot converge — Phase 5 needs real
   measurements.
2. **Ideally a currently failing suite.** An all-green repository correctly ends
   immediately with `converged` (a fixed point) and no changes.
3. Optionally, defects declared as `# BUG: name -- explanation`. A declaration is
   what makes a defect attributable and verifiable; without one, a fix needs the
   whole suite to go green before it is credited.

### Large repositories

The whole tree does not fit in a prompt, and pretending otherwise is how an
agent ends up reporting defects in files it never read. So every prompt carries
a **MANIFEST** of the tree (capped, prioritising the files being shown and the
test files) and then spends its body budget on the files that matter, clipping
large files head-and-tail rather than dropping them. Anything the agent could
not read is named in the prompt, and the agent is told to lower its `confidence`
when a judgement depends on code it never saw.

For a very large repository, narrow the scope so the budget buys depth:

```powershell
python run.py run --target . --include "src/core/*.py,src/api/*.py" --exclude "tests/fixtures/*"
```

Two measurements are taken for you, and both are recorded in the report:

* a **baseline run** of the target's suite before any agent starts — this is what
  discovery reasons from ("the suite is green, so look for what it does not
  cover"), not a guess;
* **before/after runs inside a copy of the repository**, so the sandbox sees the
  same assets and configuration the project's own CI sees. A suite that reads an
  image, a JSON fixture or a built frontend bundle behaves the same in the
  sandbox as it does in place — a sandbox that only received the `.py` files
  would report failures that do not exist.

---

## What a run produces

```
runs/<run_id>/
  REPORT.md                  decision, evidence, per-cycle story, per-agent table
  state.json                 the reportable slice of the state bus (no file contents)
  trace.jsonl                every node and agent event, append-only, replayable
  candidate/                 the exact tree Phase 5 verified
  candidate.manifest.json    baseline hashes + verified flag (apply checks these)
  candidate.diff             unified diff of the candidate against the baseline
  run.meta.json              pointers and the resolved policy
```

### About `candidate.corrupted/`

If a run directory contains **`candidate.corrupted/` and `candidate.diff.corrupted`**, those are
**quarantined bad artifacts: do not use them, and do not `apply` them.**

The cause was `write_text`'s default platform newline translation applied to a candidate body
that **already carried the target's CRLF convention**, so every `\r\n` was written as `\r\r\n` --
which reads back as *two* line breaks, inserting a blank line after **every** line. It hit
`sandbox.materialise` (the measured copy), `orchestrator.keep_candidate` (the candidate tree),
`candidate.diff`, and the diff embedded in section 4 of the report.

It was hard to notice by design: **Python does not care about blank lines, so the suite stayed
green**, and the line-ending assertion of the day counted `\r\n` -- which `\r\r\n` *contains*. The
only thing that gave it away was `git diff --numstat`: the patch landed as **74 insertions /
5 deletions** where the model wrote roughly 30 / 6.

- **Fixed**, with regression tests that check for `\r\r\n` directly and compare newline totals
  (see the Phase 5 section of `docs/phases.md`).
- Those directories are **kept, not deleted**: they are the evidence. `state.json`,
  `trace.jsonl`, `candidate.manifest.json` and `REPORT.md` are left as they were (note that
  section 4 of those reports carries the same doubled diff).
- `apply` refuses them cleanly: `this run produced no candidate tree`.
- **Runs produced after the fix are unaffected**: `candidate/`, `candidate.diff` and `REPORT.md`
  are pure LF (`CR=0`). Check `git diff --numstat` before applying.

---

## Repository layout

```
QEDLoop/
  run.py                     entry point
  qedloop/
    graph.py                 StateGraph / CompiledGraph (nodes, conditional edges, guards)
    state.py                 the state bus: channels + reducers
    core.py                  records, ids, marker scanning/retraction, diffs
    llm.py                   providers: OpenAI-compatible HTTP, DeepSeek, mock, cache, replay
    prompts.py               15 agent specs and their prompt builders
    crew.py                  fan-out, tolerant JSON extraction, normalisation
    sandbox.py               real pytest runs, static analysis, anchored edits
    policy.py                the two gates and the quality score (pure functions)
    phases/                  discover, refine, review, patch, qa
    orchestrator.py          run directories, tracing, artifacts, applying a run
    report.py, cli.py, config.py
  docs/architecture.md       why it is built this way
  docs/phases.md             per-phase contracts and agent responsibilities
  examples/buggy_service/    demo target: 3 injected defects, failing suite
  examples/loop.buggy_service.yml
  tests/                     242 tests: reducers, graph, policy, sandbox, agents, end-to-end
```

---

## Try it on your own repository

```powershell
python run.py check --target D:\path\to\your\repo          # is there anything to find?
python run.py run   --target D:\path\to\your\repo --provider mock --single-cycle
```

`--single-cycle` is the cheap way to see what discovery finds without letting the
loop change anything. The loop never writes to the target while it works; the
candidate tree lives in the state bus and only `run.py apply` touches disk.

---

## Tests

```powershell
python -m pytest tests -q
```

The suite is the specification: reducer semantics, graph routing and loop guards,
both gates, anchored-edit safety, marker scanning/retraction, agent JSON
contracts, provider behaviour, artifact writing, `apply`, and a full offline
end-to-end run that must converge on the demo fixture.

---

## Design notes worth knowing before you extend it

- **One issue per cycle.** A review verdict, a patch and a test result all belong
  to exactly one defect, so they can be attributed and replayed. Findings that do
  not fit this cycle stay in the ledger as `open`.
- **A model may interpret evidence; it may never produce it.** Test numbers come
  from pytest. The gate reads those numbers; agent verdicts only adjust the
  score.
- **`# BUG:` markers are claims, not measurements.** A fixed defect has its
  declaration retracted, and the fix is credited only when the marker is gone
  *and* the suite measurably improved. See `docs/architecture.md` §6.
- **Patches are anchored and all-or-nothing.** A missing or ambiguous anchor
  fails the proposal instead of corrupting a file.
- **The loop never writes to your repository while it is thinking.**

Full rationale, failure-mode table, extension points and known limits:
`docs/architecture.md`.

---

## License

**AGPL-3.0-only** -- `LICENSE` holds the full AGPL-3.0 text, `pyproject.toml`
declares `AGPL-3.0-only`, and every source file carries an
`SPDX-License-Identifier: AGPL-3.0-only` header. Use it on your own
repositories, internally, or redistribute it freely; **but if you offer it to
others as a network service**, the AGPL requires you to offer those users the
complete source of your modified version. That is the one substantive difference
from the GPL, and the reason for choosing it.

`only` means no permission is granted to use it under a *later* AGPL (say an
AGPL-4.0); to open that up, change `-only` to `-or-later` in `pyproject.toml`
and in the file headers.
