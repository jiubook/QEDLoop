# The five phases, agent by agent

Every phase is a graph node with the signature `state -> delta`. A node may call
models and run commands; it must not mutate the state it was given. All effects
travel back through the reducers declared in `qedloop/state.py`.

Common inputs available to every phase: `codebase` (frozen baseline),
`working_code` (candidate tree), `issues` (the accumulating ledger),
`target_issue_id` (the single issue this cycle owns), `cycle`,
`quality_history`.

---

## Phase 1 — Issue discovery

> Node `discover` · 3 agents · no gate, but it decides *whether there is work*

### Agents

| agent | lens | what it is allowed to reason from |
| --- | --- | --- |
| `discover_archaeology` | code-archaeology | the code as a contract: unreachable branches, contradicted docstrings, dead paths |
| `discover_behavior` | behavior-and-tests | existing tests and their gaps: asserted-but-false behaviour, uncovered contract |
| `discover_security` | security-and-robustness | input trust, resource bounds, silent failure, crash paths |

Each returns up to three issues with `title`, `severity`, `confidence`, `files`,
`symbols`, `evidence` (a copied line), `why_it_matters`,
`proposed_direction`, and `bug_marker` when the code declares its own defect.
Reports below `confidence < 0.35` are dropped.

### Reads / writes

Reads `working_code`, `issues` (to avoid re-reporting), `cycle`.
Writes `issue_candidates`, `issues`, `batch_issue_ids`, `target_issue_id`,
`focus_ids`, `batch_verified`, `needs_discovery`, `checks`.

### Processing

1. **Skip guard** — if `needs_discovery` is false, the current target has not
   been verified yet; re-scanning would only re-report it.
2. **De-duplicate** by defect identity (marker, else title + first file). A
   re-sighting reuses the existing ledger row, adds the lens, raises confidence,
   keeps the strongest severity.
3. **Rank and cap** at nine rows: severity, then confidence, then id.
4. **Pick one target** — highest-priority issue that is not already `verified`.

### Exits

| condition | next | `status` |
| --- | --- | --- |
| a target was selected | Phase 2 | `running` |
| nothing new, nothing open | `END` | `converged` (fixed point) |
| `needs_discovery` false | Phase 2 with the existing target | unchanged |

A fixed point is a *result*, not an error: it is the honest answer to "is there
anything left to do here".

---

## Phase 2 — Requirement refinement

> Node `refine` · 3 agents · no gate

### Agents

| agent | mode | output |
| --- | --- | --- |
| `refine_synthesize` | produce | `decision` (`act`/`defer`/`reject`), falsifiable `summary`, `acceptance` criteria, `non_goals`, `blast_radius`, `risk` |
| `plan_split` | produce | ordered `steps` with files, `estimate`, `rollback` |
| `test_design` | produce | test cases: `kind`, `name`, `path`, `given`, `when`, `then`, `regression_of` |

A requirement that cannot be proven false is not a requirement. `acceptance`
entries are the observable checks Phase 3 reviews against and Phase 5 verifies
with.

### 两趟执行：先综合，后规划

`refine_synthesize` **先单独跑**，它产出的 `acceptance` 与 `non_goals` 会作为
**约束块**注入 `plan_split` 与 `test_design` 的提示词；这两个 agent 再并行执行。

三个 agent 原先是一起发出的，互相看不见。实测后果是同一轮里
`refine_synthesize` 写下「不得新增测试函数」，而从未看到这句话的 `test_design`
提出了一个新测试函数，评审据此连续三轮否决——**否决是对的，矛盾是框架造的**。
代价是 Phase 2 从 1 个往返变成 2 个往返，墙钟时间约翻倍（token 总量不变）。

### Reads / writes

Reads the target issue, earlier `reviews` blocking findings (so a re-refinement
addresses the objection rather than repeating itself), `working_code`.
Writes `findings`, `test_plans`, `refine_votes`, `checks`.

### 计划的世代与退役

一轮细化产出的是一份**完整计划**，不是上一份计划的增量。每条 `test_plans` 行带
一个 `refined_at` 世代戳（取自本轮 finding）：新世代到达时，同一 issue 的旧世代会
被打上 `superseded_at` 并**退出服役**，但**保留在账本里**供审计与重放。

退役按世代戳而不是按 id，因为每轮的计划 id 都是新的。不退役的后果实测过：同一个
issue 细化四轮后账本留下 5 条计划行，其中 3 条指向同一个测试函数，评审每一轮都以
「哪个才是权威版本」为由否决——**账本在说谎，而不是模型在读错**。

Phase 3 评审与 Phase 4 改码只看到**在服役**的那一代（`prompts.plans_for`）。

### Failure modes

- A `defer`/`reject` decision removes the issue from this cycle's work
  (`fresh_batch` filters it) so the loop does not spend a patch on it.
- If nothing is actionable, the node sets `status = human_review` with a reason
  instead of proceeding to review an empty plan.
- 综合 agent 失败时不对后两个 agent 施加约束块（提示词里写明「本轮未声明约束」），
  而不是编造一份约束出来。

---

## Phase 3 — Code review

> Node `review` · 3 agents · **gate: may this plan be implemented?**

### Agents

| agent | lens | mode |
| --- | --- | --- |
| `architecture_review` | architecture | **vote** — does the change belong here; does it duplicate or warp an abstraction |
| `correctness_review` | correctness | **vote** — will this actually satisfy the acceptance criteria |
| `risk_review` | risk-and-regression | note — blast radius, regression surface, reversibility |

Verdicts are `approve` / `reject` / `abstain`, with `blocking[]` entries naming
what must change. Reviewers judge the *plan*; no implementation exists yet.

### Gate (`policy.review_route`)

```
plan check fails                 -> refine   (mechanical: three agent calls are skipped)
any reject                       -> refine   (blocking findings carried into Phase 2)
approvals >= min_approvals       -> patch
abstains only                    -> refine   (insufficient information)
rejects past max_self_loops      -> END      status = human_review
```

A voiced objection always wins over a bare approval. `self_loop_count` is
incremented on each `review → refine` edge and **counted across the whole run**：
没有任何地方重置它（不按轮、也不按 issue），所以一个难 issue 可以独自把
`max_self_loops` 的花光，`max_iterations` 的轮数预算还没开始用就已经交回给人。
这句原先是「reset per cycle」，与代码不符。

### Plan check (`plans.check_plan`) — 先花机器的时间，再花模型的钱

计划里自己的验证命令收集不到自己列出的测试，是**文档内部的矛盾**，不需要判断力。
实测一轮运行的四次评审里有三次都是这类问题，每次三个 agent、21–61 秒：

| 实测缺陷 | 机器怎么看出来 |
| --- | --- |
| `-k not_running` 面对的是 `test_..._when_scanner_idle`，而且它真的选中了另一个名字相近的旧测试（`..._when_scanner_not_running`）——那一步会 exercise 错测试却照样「通过」 | `-k` 的词元必须落在这个计划为**同一个文件**新增的测试名上 |
| step 7 硬编码 `::test_webui_sentinel_not_accessed_when_window_injected`，而计划里同一测试叫 `test_exit_application_with_injected_window_never_reads_webui_module` | `::` 后面的名字必须存在于计划或仓库 |
| 计划新建 `tests/unit/services/test_system_service.py`，而仓库已有 `tests/unit/test_system_service.py`，且没有任何一步提到旧文件 | 新建测试模块的 basename 撞车 + 全步骤文本里没提到旧路径 |

命中时**不调用任何评审 agent**，直接以 `lens="plan-consistency"` 的 reject 行回流：
`review_route` 照常把它计入回边预算（不给它单独的预算，否则一份坏计划可以免费来回），
下一轮 `refine_prompt` 也会在 blockers 里看到它。

它看不见判断力：没人能从字符串看出「记录所有属性访问的 sentinel 必然先记到
`__path__`」——那要靠实测，仍然归 lens。回放验证（`runs/_replay_plan_check.py`）：对那轮
5 代计划命中 3 代，正好是 3 条机械性的；对**被通过的那一代保持沉默**。

### Reads / writes

Reads the target issue, its latest finding, its **在服役的** test plans, current
`evidence`，以及 issue 与计划点名的那些文件的正文（`prompts.focus_files` 把它们
排到预算最前面）。
Writes `reviews`, `review_decision`, `review_reason`, `review_blocking`,
`self_loop_count`, `checks`. 计划检查命中时也写 `reviews`（多一条
`lens="plan-consistency"` 的行），因为下一轮的 `refine_prompt` 正是从
`reviews` 里取 blockers 的。

---

## Phase 4 — Code change

> Node `patch` · 3 agents · no gate, but it must produce an applicable patch

### Agents

| agent | mode | strategy |
| --- | --- | --- |
| `patch_generate` | produce | the smallest anchored edit satisfying the acceptance criteria |
| `patch_refactor` | produce | the alternative that removes the class of defect, may touch more lines |
| `patch_reconcile` | synthesize | picks A or B with a reason; falls back to the minimal patch if unavailable |

Edits are anchored: `search` must appear **verbatim and uniquely** in the named
file. `sandbox.apply_ops` enforces this and is all-or-nothing — a missing or
ambiguous anchor fails the whole proposal, because that is how a model-generated
patch silently corrupts a repository.

### 新增文件（`search` 为空）

一个 op **有 `path`、`search` 为空、`replace` 非空**，表示「新建这个文件」。
契约原先没有这种写法，于是「补一个回归测试」这类补丁只能交回空 `search`，再被
当成畸形输入整份拒掉——实测一次真实运行，agent 自己在 rationale 里写明了
*「新文件无既有内容，故 search 为空」*，而那份补丁本身是好的。

意图必须**明确写在 rationale 里**（`create` / `new file` / `新增` / `新建` / `创建`），
因为「被截断的回复」同样表现为空 `search`；只允许新建 `.py`，且文件**必须不存在**
（已存在的文件要走锚定修改）。这几道门的作用是：空 `search` 永远不会被读成
「整文件替换」，否则一次意外就能覆盖整个仓库。

### Reads / writes

Reads the target issue, its finding and test plans, `working_code`,
`review_decision`. Writes `patches`, `changes`, `selected_patch`,
`working_code`, `patch_error`, `checks`.

### Marker retraction

When a patch is applied, the declaration of the defect it fixed is removed
(`core.strip_marker`) — otherwise the next verification pass would read the old
`# BUG:` claim and conclude the defect is still present. Only comments are
removed, and both shapes are handled: a marker on its own line (with its
continuation comment block) and a marker trailing real code (comment only).

Nothing is written to the target here: `working_code` is a candidate tree.

---

## Phase 5 — QA verification and convergence

> Node `qa` · 3 agents · **gate: is the result good enough?**

### Measurement first

`sandbox.verification_snapshot(baseline, candidate)`:

- runs the target's own suite against the frozen baseline and the candidate, in
  scratch directories, and records passed/failed/errors, the failing test ids,
  which tests were repaired, and the regression delta;
- compiles every non-test file in the candidate tree and diffs declared defect
  markers (resolved / introduced).

This runs *before* any model is asked anything, so a model can interpret
evidence but never produce it.

测量副本与候选树都由 `sandbox.materialise` / `orchestrator.keep_candidate` 写盘，两者
**必须显式传 `newline=""`**。候选正文带着目标的 CRLF 约定，而 `write_text` 的默认行为会
把每个 `\n` 翻成平台分隔符，于是每个 `\r\n` 变成 `\r\r\n` —— 读回来是**两个换行**，等于
在每一行后面插一个空行。实测一次真实运行：落盘到目标的补丁是 74 insertions / 5 deletions，
而模型写的是约 30 / 6，**套件全程全绿**（Python 不在乎空行）。把补丁应用到目标仓库之前，
`git diff --numstat` 是唯一会把它露出来的地方。

可选地，`run.lint_command` 会在同一个候选树副本里再跑一次**目标仓库自己的静态检查**。
`{python}` 展开为沙箱解释器，`{files}` 展开为本轮改动的文件——只检查补丁碰过的文件，失败
才归因得到这一轮。结果作为一条硬性条件进 `block_reasons`（与 compiles / tests_green 同级，
不进加权分），因为闭环原本的两个证据都看不见 lint 规则：实测某轮补丁的**唯一**缺陷是
`TC003`，507 个测试全绿而 EER-Ai 自己的 ruff 拒绝它。命令**必须只读**（不加 `--fix`），
且**配了但跑不起来算没通过**，否则一个 typo 就等于把这道门悄悄关掉。

### Agents

| agent | lens | mode |
| --- | --- | --- |
| `static_qa` | static-analysis | note — reads the compile report and the marker diff |
| `test_sandbox` | sandboxed-test-run | **vote** — reads before/after numbers and says whether the defect is gone |
| `edge_case_qa` | adversarial-edge-cases | **vote** — tries to break the patch with an input the author did not consider |

### Credit rule

An issue is marked `verified` when the candidate compiles, the patch introduced
no new failure, **and**:

- its declared marker is gone **and** at least one previously failing test now
  passes (and no failing test matches the defect's predicted test name), or
- it declared no marker and the suite is green.

Not required: a fully green suite. See `architecture.md` §6 for why.

### Gate (`policy.score_cycle` + `policy.convergence`)

Quality = `0.5·tests + 0.2·static`（按权重和归一化，实际占比 0.714 / 0.286），recomputed
from evidence。**四个**硬性要求与分数分开判定，分数永远无法覆盖它们：候选树**可编译**、
套件**全绿**、计划拿到 **≥ `min_approvals`** 张赞成票、以及 **Phase 5 的三个 QA lens
没有一张否决票**。

第四条是后补的，代价已经付过。实测一次运行：质量 **1.00**（tests 1.0 / review 1.0 /
static 1.0，套件全绿、树可编译、3 张赞成票），而 `static_qa`、`test_sandbox`、
`edge_case_qa` **全部投了否决**——因为补丁只改了测试断言、没改任何行为，而分数是由
「套件是否绿」算出来的，`edge_case_qa` 的原话是 *"it does not meet the stated
acceptance contract"*。旧的门控只读分数，于是报告说「满分」，用户看不到任何反对意见，
issue 还被判成 `verified`。

评审票**曾经**是第三个加权分量（0.3），已经移除：它是硬性要求，而硬性要求不能再当分数，
否则同一个事实被数两次。实测代价：一次运行里 12 条评审记录横跨四轮（6 赞成 / 6 反对），
Phase 3 闸门在最后一轮以 **3/3** 放行，补丁干净落地、套件从 503 涨到 **507 passed**，
而分数报出 `review 0.00`、`quality 0.70` —— 因为算分除的是**整条追加式账本**
（`(6-6)/12`），不是真正起作用的那一轮。单周期运行于是把一次成功当成失败交给人。

改成「只算最后一轮」并不能修好它：`review_route` 只要有反对票就回 `refine`，所以任何走到
Phase 4 的轮次在那一轮里必然是 0 反对，这个分量恒等于 1.0，只是一个稀释实测证据的常数。
配置里残留的 `review:` 键会在加载时被丢弃（`policy.RETIRED_WEIGHTS`），报告页脚打印的是
**生效**权重，而不是配置里那个不生效的。

| status | condition |
| --- | --- |
| `converged` | compiles **and** green suite **and** approvals **and** quality ≥ `quality_target` **and** **没有任何 QA lens 否决** **and** 已完成轮数 ≥ `min_iterations` |
| `continue` | budget remains, progress plausible → `cycle += 1` |
| `human_review` | no issue closed and quality flat for `max_no_progress` cycles, or single-cycle mode |
| `escalated` | 已完成轮数 ≥ `max_iterations` |

**阻断原因只有一个来源**（`policy.block_reasons`）：每一个返回分支都必须原样带上这份
列表。以前不是——单周期分支硬编码了「quality X below target Y」，在 quality 1.00 的那次
运行里打出了自相矛盾的 *"quality 1.00 below target 0.80"*；`escalated` 分支则只说
「预算耗尽」，完全不说是什么挡住了收敛。「预算耗尽」解释的是**为什么停**，不是
**为什么失败**。

QA 的否决票还有第二个效果：**当轮不允许给任何 issue 记 `verified`**
（`qa._resolved_issue_ids(..., qa_rejected=True)` 直接返回空）。只有 QA lens 是在候选树
**实测之后**才发言的，所以「套件仍然全绿」在它反对时不再是证据——那个补丁一个字节的
行为都没改。

`min_iterations` 也是这份列表里的一条，而且它曾经**漏在外面**。原因是索引与轮数被混用了：
`state["cycle"]` 是**刚跑完那一轮的 0-based 索引**（`qa.py` 在门控之后才 `cycle + 1`），
而 `min_iterations` 数的是**轮数**（报告里的 "cycles completed" 就是 `len(history)`）。
`cycle >= min_iterations` 于是让默认值 `1` 变成了「至少要跑两轮」——配置注释写的是
*"never declare victory before this many cycles"*。实测代价：一次运行单轮跑出 quality
**1.00**、套件全绿、5 张赞成票、三个 QA lens 全部 approve，而交回给人的理由印成了
`single-cycle mode stopped the loop; `——**分号后面什么都没有**，因为 blocker 列表是空的。
现在按轮数比较（`cycle + 1`），并且「未达轮数」也在 `block_reasons` 里，于是
**「没有 blocker」与「该收敛」是同一件事**：任何非收敛分支都必然至少有一条原因可印。

**同一个索引/轮数陷阱在预算那一头也有一份**，方向相反：`escalated` 原先判的是
`cycle >= max_iterations`。索引从 0 开始，所以 `max_iterations: 3` 实际放行了
**四轮**（索引 0、1、2 各返回 `continue`，索引 3 才停），报告会印出自相矛盾的
`cycles completed | 4 of 3`。同样改成按轮数（`cycles_done >= max_iterations`），
现在 `max_iterations: 3` 就是三轮。实测：历史上 29 次运行没有一次跑到过第四轮，
所以这次修正没有改变任何既有结论；但想保留原先的**实际**预算，把配置值 +1 即可。

### Reads / writes

Reads `working_code`, `codebase_files`, `reviews`, `quality_history`,
`max_iterations`. Writes `evidence`, `checks`, `qa_votes`, `verifications`,
`quality_history`, `cycles`, `no_progress_cycles`, `issues` (status →
`verified`), `batch_verified`, `needs_discovery`, `status`, `status_reason`,
`cycle`.

---

## Appendix — the run loop as one table

| step | node | 3 agents | gate | on failure |
| --- | --- | --- | --- | --- |
| 1 | `discover` | archaeology / behavior / security | fixed point → END | — |
| 2 | `refine` | synthesize / plan / test-design | actionable? | `human_review` |
| 3 | `review` | architecture / correctness / risk | any reject → refine | `human_review` after `max_self_loops` |
| 4 | `patch` | generate / refactor / reconcile | anchors apply? | `human_review` |
| 5 | `qa` | static / sandbox / adversarial | convergence | `continue` / `escalated` / `human_review` |

## Appendix — test-name convention

`sandbox.test_name_for_marker("text/title_case")` → `test_text_title_case`. A
repository that follows this convention lets QA attribute a failing test to the
defect that predicted it, which sharpens the credit rule. It is a convention,
not a requirement: without it the rule falls back to marker retraction plus
"some previously failing test now passes".

## Appendix — retracting a declaration by hand

```python
from qedloop.core import strip_marker

code = {"src/mod.py": open("src/mod.py", encoding="utf-8").read()}
fixed = strip_marker(code, "mathx/clamp_inverted", ["src/mod.py"])
```

This is what Phase 4 does automatically for the defect it just fixed.
