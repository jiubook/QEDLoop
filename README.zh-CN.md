# QEDLoop

**自迭代软件开发闭环**。QED = *quod erat demonstrandum*，「证毕」。名字本身就是规矩：拿不出证明，就不许说收敛。

> English version: [README.md](README.md)

一个跑在真实仓库上的 AI 闭环：**五个阶段 × 每阶段三个 agent**，共享一条状态总线，收敛与否由**实测证据**决定而不是由模型说了算。

```
discover ─► refine ─► review ─► patch ─► qa ─┐
    ▲          ▲         │                   │
    │          └── 拒绝 ─┘                   │  质量未达标 → 下一轮
    └────────────────────────────────────────┘
     converged / escalated / human_review
```

两条回边各有一个预算封顶：`reject` 退回 Phase 2，`质量未达标` 回到 Phase 1 开新一轮（两者的守卫见 `docs/architecture.md` 的 loops 表）。

- **Phase 1 发现**：三个透镜各自找新缺陷；同一个缺陷的多份报告折叠成一条账目；每轮只锁定**一个**目标。
- **Phase 2 细化**：把缺陷变成可证伪的需求 + 有序实施步骤 + 测试设计。
- **Phase 3 评审**：架构与正确性两个 agent **投票**，风险 agent 评估爆炸半径；被否决则回到 Phase 2（有次数上限）。
- **Phase 4 改码**：两个 agent 各写一份补丁（最小改动 vs 重构），第三个裁决；胜出者以**锚定式编辑**应用到候选树。
- **Phase 5 验证**：在仓库副本上真实跑测试（改前 + 改后），三个 agent 解读数据，门控决定收敛、继续或交人。

**纯 Python 3.9+，零必需依赖**（只有验证候选树时需要 pytest）。整个闭环可以用**确定性 mock provider 离线跑通**，不花一分钱就能看清它怎么工作。

---

## 与 LangGraph 的对应关系

| LangGraph | 本项目 |
| --- | --- |
| `StateGraph(State)` | `qedloop/state.py` 的 `CHANNELS` 通道注册表 |
| 带注解的 reducer | `qedloop/core.py` 的 `merge_ledger("issues"/"findings"/...)` |
| `add_node` / `add_edge` | `StateGraph.add_node` / `add_edge` |
| `add_conditional_edges` | `StateGraph.add_conditional_edges`（`review` 与 `qa` 都用它） |
| `START` / `END` | `qedloop.graph.START` / `END` |
| `compile().invoke` | `StateGraph.compile()` → `CompiledGraph.run()` |
| `.stream()` | `CompiledGraph.stream()`（逐节点产出事件） |
| `recursion_limit` | `max_steps` + 每节点 `max_visits` |

---

## 快速开始（60 秒）

```powershell
cd QEDLoop

# 1. 看看演示靶场里有什么（3 个注入缺陷，测试初始为红）
python run.py check --target examples/buggy_service

# 2. 离线跑闭环（确定性，不需要 API key）
python run.py run --target examples/buggy_service --provider mock

# 3. 读结果
start runs\<run_id>\REPORT.md

# 4. 先预演再落盘（每个文件留 .qedloop.bak 备份）
python run.py apply --run runs\<run_id> --dry-run
python run.py apply --run runs\<run_id>
```

演示靶场三轮收敛，收尾输出：

```
status           : converged
reason           : quality 1.00 >= target 0.80 with a compiling tree, a green suite and 9 approvals
changed files    : src/tinylib/mathx.py, src/tinylib/stats.py, src/tinylib/text.py
```

---

## 命令

| 命令 | 用途 |
| --- | --- |
| `run.py run --target <目录>` | 对目标仓库跑闭环，产物写入 `runs/<run_id>/` |
| `run.py apply --run <目录>` | 把**已收敛**的候选树写回目标（`--dry-run` 预演，默认留备份） |
| `run.py channels [--config <文件>] [--probe]` | 列出所有模型渠道及其地址；`--probe` 逐个发一次真实请求验证连通 |
| `run.py agents` | 列出 15 个 agent 角色的透镜、mode 与职责 |
| `run.py check --target <目录>` | 基线体检：文件数、缺陷标记、测试是否红 |

常用参数：`--config`、`--brief`、`--max-iterations`、`--quality-target`、`--min-approvals`、`--max-self-loops`、`--single-cycle`、`--no-tests`、`--token-budget`、`--include/--exclude`、`--quiet`。

### 退出码

| 状态 | 码 | 含义 |
| --- | --- | --- |
| `converged` | 0 | 能编译 + 测试全绿 + 有评审通过 + 质量达标 |
| `continue` | 0 | 预算还有，报告会说明还差什么 |
| `human_review` | 1 | 无进展或单轮模式，需要人看一眼 |
| `escalated` | 3 | 迭代预算耗尽 |
| `error` | 4 | 节点抛异常；部分运行结果仍会落盘 |

### 运行期间怎么看进展、怎么中断

不加 `--quiet` 时，每个阶段节点与每个 agent 的结果都会**实时逐行打印**（下面是
`python run.py run --target examples/buggy_service --provider mock` 的真实输出）：

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

基线那两行不是装饰：第一步要复制整棵目标仓库再跑一遍它的套件，在大仓库上是**几分钟的静默**，
而静默正是让人对着一次正常运行的闭环按 `Ctrl+C` 的原因。

`quiet`（配置里的 `run.quiet: true`）抑制的正是这份实时输出，运行结束时仍会打印结果摘要；
命令行 `--quiet` 与它是同一个开关。

`trace.jsonl` 里是同样的信息加上完整字段（否决理由全文、三张评审票、每个 agent 的 token
与耗时），运行中另开一个窗口可以流式跟读：

```powershell
# 注意：`runs\<run_id>\trace.jsonl` 这种写法不能照抄 —— `<` `>` 在 Windows 路径里是
# 非法字符，PowerShell 还会把它们当成重定向符，照抄会报「路径中具有非法字符」。
# 下面这版直接找最新被写入的那个 trace 文件（正在跑的那次永远是最新的）：
$run = Get-ChildItem runs -Recurse -Depth 2 -Filter trace.jsonl | Sort-Object LastWriteTime | Select-Object -Last 1
Get-Content $run.FullName -Wait -Tail 20
```

（`-Depth 2` 同时覆盖 `runs\<run_id>\` 与 `runs\<目标 id>\<run_id>\` 两种布局；`-Tail 20`
先回放最后 20 条再跟着写。`Ctrl+C` 退出跟读，不会影响正在跑的闭环。）

**中断**：在控制台按 `Ctrl+C`。当前节点会停下，但这次运行不会丢——状态记为
`human_review`、理由是 `interrupted by the operator after N node(s)`，`state.json` 与
`REPORT.md` 照常写出，退出码 1。中断的运行**故意不写候选树**，所以 `apply`（包括
`--allow-unverified`）一律拒绝：被验证过才是可落盘的前提，而中断恰恰意味着什么都没验证。

两个实测细节：中断在**模型调用**之间送达很快（实测 2.0s 内结束，`trace.jsonl` 里留下一条
`refine/start` 而没有对应的 `done`）；但如果正好卡在测量套件的那一步，Windows 下它要等那次
pytest 的等待返回才送达（实测子进程跑满 20s，父进程 20.6s 才返回）——结果仍是上面那份，只是慢。
按第二次 `Ctrl+C` 会直接结束进程，那就只剩 `trace.jsonl` 可读了。

---

## 多模型渠道与自定义 API 地址

框架通过一个 **渠道（channel）** 抽象访问任何 OpenAI 兼容端点：官方 API、公司网关、第三方聚合、本地 Ollama / LM Studio 都只是配置差异，不需要改代码。

### 内置渠道

```
python run.py channels
```

| 名称 | 类型 | 默认地址 | 需要 key |
| --- | --- | --- | --- |
| `openai` | openai | `https://api.openai.com/v1` | 是（`OPENAI_API_KEY`） |
| `deepseek` | deepseek | `https://api.deepseek.com/v1` | 是（`DEEPSEEK_API_KEY`） |
| `ollama` | openai-compat | `http://127.0.0.1:11434/v1` | 否 |
| `lmstudio` | openai-compat | `http://127.0.0.1:1234/v1` | 否 |
| `mock` | mock | — | 否（离线确定性） |
| `auto` | — | — | 自动挑第一个可用渠道，都没有就用 mock |

### 在自己的配置里声明渠道

任何配置文件都可以加一节 `providers:`（`--config` 指到的那个文件）：

```yaml
providers:
  company-gateway:                # 名字随你起
    kind: openai-compat           # 任何 POST {base_url}/chat/completions 的服务
    base_url: https://llm.corp.example.com/v1
    model: gpt-4o-mini
    api_key_env: CORP_LLM_KEY     # key 从环境变量读，不写进文件
    timeout: 120
    cache: true

  local-ollama:
    kind: openai-compat
    base_url: http://127.0.0.1:11434/v1
    model: qwen2.5-coder:14b
    api_key_env: ""               # 留空 = 不需要 key
    timeout: 300                  # 本地模型慢，别提前超时

  openrouter:
    kind: openai-compat
    base_url: https://openrouter.ai/api/v1
    model: anthropic/claude-3.5-sonnet
    api_key_env: OPENROUTER_API_KEY
```

然后按名字使用：

```powershell
python run.py channels --config my.yml --probe        # 先看连通性
python run.py run --target ./myrepo --config my.yml --provider company-gateway
python run.py run --target ./myrepo --config my.yml --provider local-ollama --model qwen2.5-coder:32b
```

字段说明：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `kind` | 否 | `openai-compat`（默认）/ `openai` / `deepseek` / `mock`；别名 `compat`、`openai-compatible` 都认 |
| `base_url` | 否 | 自定义 API 地址，结尾斜杠会被规范化掉 |
| `model` | 否 | 模型 id |
| `api_key_env` | 否 | 从哪个环境变量读 key；写 `""` 表示该端点不需要 key —— 此时**不发 Authorization 头**，也不会借用环境里的 `OPENAI_API_KEY`（本地服务不会被喂别人的 key） |
| `api_key` | 否 | 直接写 key（不推荐，会被提交进版本库） |
| `timeout` | 否 | 单次请求超时秒数，`llm_timeout` 未设时生效 |
| `max_tokens` | 否 | 该模型的**输出上限**；声明后覆盖每个 agent 的默认预算（1400）。思考模型的思维链 token 也算在输出里，所以开了思考就必须调大 |
| `temperature` / `cache` | 否 | 默认采样温度、该渠道是否复用相同提示词的缓存 |
| `note` | 否 | 给自己看的备注 |

### 命令行覆盖与优先级

```powershell
python run.py run --target ./myrepo `
  --provider company-gateway `
  --model gpt-4o `
  --base-url https://another-gateway.example/v1 `
  --api-key sk-... `
  --llm-timeout 240
```

每个字段的优先级：**显式参数 > 渠道配置 > 环境变量 > 内置默认**。

可用的环境变量：`LLM_PROVIDER`、`LLM_MODEL`、`LLM_BASE_URL`、`OPENAI_API_KEY`、`LLM_API_KEY`、`DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`。其中任一渠道的 `base_url` 都可以写成 `${VAR:-默认值}` 形式，由配置加载器展开。

### 两个刻意的设计

1. **`auto` 不会去连本地预设**。内置的 `ollama` / `lmstudio` 标记为 `auto_select: false`：新机器上跑 `--provider auto` 时，如果它去尝试连一个没启动的本地端口，每次运行都会卡在死端口上。你自己在配置里声明的渠道则允许被 `auto` 选中。
2. **key 永远不会被打印**。`channels` 只显示「有 / 没有」以及从哪个环境变量读；`--probe` 也只显示连通结果。

---

## 在自己的仓库上使用

**不需要把仓库移进 `QEDLoop`。** `--target` 接受任意路径：

```powershell
# 绝对路径：任何目录下都能跑
python run.py check --target D:\work\myrepo

# 相对路径：相对「当前工作目录」，不是相对 QEDLoop 自己的目录
cd D:\work
python D:\work\QEDLoop\run.py check --target myrepo

# 或者直接进入目标仓库
cd D:\work\myrepo
python D:\work\QEDLoop\run.py check --target .
```

run 产物写在**当前工作目录**的 `runs/`（`--out` 可改），不会污染目标仓库；`apply` 默认写回 `state.json` 里记录的绝对 target。

**目标仓库的专属设定放在本仓库里，不写进目标仓库。** 任何一个被测仓库的配置——以及给 agent 的项目约束——都放在 `targets/<目标 id>/` 下，跑完闭环后那个仓库的 `git status` 不会有任何变化：

```
targets/
  README.md       约定说明
  eer-ai/
    loop.yml      python run.py run --config targets\eer-ai\loop.yml
    brief.md      注入到每个 agent 的项目约束（loop.yml 的 brief: 指向它）
```

`brief.md` 是「光看代码看不出来」的硬规则——不许碰生成物、不新增依赖、改某个 schema 必须配套哪几步迁移。它会作为 `<project_constraints>` 段落进到**每个** agent 的系统提示（上限 4000 字符），并被明确告知：这是规则、不是证据。相对路径相对配置文件所在目录（`brief: brief.md` 就是同目录那份）；文件读不到会直接报错退出，不会静默当成「没有约束」。也可以用 `--brief <路径>` 临时指定。

> `brief.md` 喂给的是**闭环里那 15 个 agent**。给**你自己的 coding agent**（Claude Code 等）的
> 项目约定是另一份：仓库里带的是模板 `CLAUDE.md.sample`，复制成 `CLAUDE.md` 再填 —— 里面
> 预留了「如何写这个文件」的说明，以及在占位符没填时让 agent 主动提醒你的指令。
> `CLAUDE.md` 在 `.gitignore` 里，属于本地文件。

```powershell
python run.py check --target D:\work\myrepo                                # 有没有可度量的缺陷？
python run.py run --target D:\work\myrepo --provider mock --single-cycle   # 只看发现结果，不写文件
python run.py run --config targets\myrepo\loop.yml --provider company-gateway
python run.py apply --run runs\<run_id> --dry-run                          # 预演
python run.py apply --run runs\<run_id>                                    # 落盘
```

### 目标仓库需要什么

1. **pytest 能发现的测试**（`tests/test_*.py` 或 `test_*.py`）。没有测试也能跑，但无法收敛 —— Phase 5 需要真实测量结果。
2. **最好是当前红的**。全绿仓库会立刻以 `converged`（不动点）结束、零改动，这是正确答案。
3. 可选：用 `# BUG: 名称 -- 说明` 声明缺陷。有声明才能被精确归属与验证；没有声明的缺陷需要整套测试变绿才能判定修复。

### 大仓库

整棵树塞不进 prompt，假装塞得进去就会让 agent 报告自己没读过的文件。所以每次 prompt 都带一份**文件清单（MANIFEST）**（有上限，优先列出正在展示的文件与测试文件），正文预算花在关键文件上，超大文件做**头尾裁剪**而不是丢弃；没能读到的部分会被明确点名，并告知模型「基于没读到的代码下判断时要降低 confidence」。

仓库很大时，缩小范围让预算买到深度：

```powershell
python run.py run --target . --include "src/core/*.py,src/api/*.py" --exclude "tests/fixtures/*"
```

另有两项测量自动完成，都会写进报告：

- **基线测量**：任何 agent 启动前先跑一遍目标测试，discovery 据此推理（「测试是绿的，去找没覆盖到的地方」），而不是靠猜；
- **改前/改后在仓库副本里跑**：沙箱看到的是与项目自身 CI 相同的资源与配置。只拿到 `.py` 文件的沙箱会报出**根本不存在**的失败。

---

## 产出物

```
runs/<run_id>/
  REPORT.md                  结论、证据、逐轮经过、逐 agent 表
  state.json                 状态总线中可上报的那一片（不含文件正文）
  trace.jsonl                所有节点与 agent 事件，追加写，可重放
  candidate/                 Phase 5 实际验证过的那棵树
  candidate.manifest.json    基线哈希 + verified 标记（apply 会校验）
  candidate.diff             候选树相对基线的 unified diff
  run.meta.json              指针与最终生效的策略
```

### 关于 `candidate.corrupted/`

若某个运行目录里出现 **`candidate.corrupted/` 与 `candidate.diff.corrupted`**，那是**已隔离的
坏产物：不要使用，更不要 `apply`**。

原因是 `write_text` 的默认行为会做平台换行翻译，而候选正文**已经带着目标仓库的 CRLF 约定**，
于是每个 `\r\n` 被写成 `\r\r\n` —— 按通用换行读回来是**两个**换行，等于在**每一行后面凭空
插一个空行**。`sandbox.materialise`（测量副本）、`orchestrator.keep_candidate`（候选树）、
`candidate.diff`、以及报告第 4 节嵌入的 diff 都中招过。

这个缺陷自己很难露头：**Python 不在乎空行，所以测试套件全程全绿**，而且当时那条行尾断言
只数 `\r\n`，而 `\r\r\n` **含有** `\r\n`，所以它照样通过。唯一露馅的地方是 `git diff
--numstat`：实测那份补丁落盘时是 **74 insertions / 5 deletions**，而模型写的是约 30 / 6。

- **已修复**，并补了直接检查 `\r\r\n` 与换行总数的回归测试（见 `docs/phases.md` 的 Phase 5 小节）。
- 这些目录**不会被删除**：它们就是出过问题的证据。`state.json` / `trace.jsonl` /
  `candidate.manifest.json` / `REPORT.md` 原样保留（注意这批 `REPORT.md` 第 4 节的 diff
  带同样的空行）。
- 对它们执行 `apply` 会干净地拒绝：`this run produced no candidate tree`。
- **修复之后新产生的运行不受影响**：`candidate/`、`candidate.diff`、`REPORT.md` 一律是纯
  LF（`CR=0`），落盘前可用 `git diff --numstat` 复核行数。

---

## 三个关键设计决定

1. **一轮只处理一个 issue**。评审判决、补丁、测试结果都只属于一个缺陷，才能被归属和重放；一次发现里的其它问题留在账目里当 `open`，成为后续轮次的目标。
2. **模型只能"解读"证据，不能"生产"证据**。测试数字来自真实 pytest；门控直接读数字，agent 投票只能影响评分。
3. **`# BUG:` 是声明，不是测量**。修好一个缺陷后会**撤回该声明注释**（`core.strip_marker`，只删注释、能区分整行注释块与行尾注释），再配合「至少一个原先失败的测试现在通过」才判定 verified。

质量分是三者的加权（测试 0.5 / 评审 0.3 / 静态 0.2），但**三项硬性要求独立于分数之外**：候选树必须能编译、测试必须全绿、必须有评审通过 —— 分数再高也盖不住它们。

---

## 项目结构

```
QEDLoop/
  run.py                     入口
  qedloop/
    graph.py                 StateGraph / CompiledGraph（节点、条件边、守卫）
    state.py                 状态总线：通道 + reducer
    core.py                  记录、id、缺陷标记扫描与撤回、diff
    llm.py                   渠道抽象 + provider（OpenAI 兼容 HTTP / DeepSeek / mock / 缓存 / 回放）
    prompts.py               15 个 agent 契约与提示词构建
    crew.py                  扇出、容错 JSON 提取、归一化
    sandbox.py               真实 pytest 测量、静态分析、锚定式补丁
    policy.py                两个门控与质量评分（纯函数）
    phases/                  discover / refine / review / patch / qa
    orchestrator.py          运行目录、tracing、产物、apply
    report.py, cli.py, config.py
  docs/architecture.md       为什么这样设计（英文）
  docs/phases.md             逐阶段契约与 agent 职责（英文）
  examples/buggy_service/    演示靶场：3 个注入缺陷 + 失败测试
  examples/loop.buggy_service.yml
  tests/                     测试即规格
```

---

## 测试

```powershell
python -m pytest tests -q
```

测试套件就是规格说明：reducer 语义、图路由与循环守卫、两个门控、锚定编辑的安全性、缺陷标记扫描与撤回、agent JSON 契约、渠道解析与自定义地址、provider 行为、产物写入、`apply`，以及一次完整的离线端到端收敛。

---

## 已知限制

- **单进程、单目标**：`apply` 就是文件拷贝，没有 git 集成、分支或 PR。
- **候选树由锚定式查找/替换产生**，不是真正的 patch 工具：没有唯一锚点的改动无法表达，也不能新建文件。
- **成功依赖仓库自身的测试**。没有任何测试能观察到的缺陷会被报告、打补丁，然后被 QA 正确地拒绝 —— 正确但没用。给 discovery 一个像样的测试套件是使用者的责任。
- **缺陷归属依赖标记**。报告里能看到是哪个信号给某次修复记了分，但「缺陷确实消失」除「标记被撤回 + 测试有改善」之外没有更强的证据。
- **无跨运行记忆**：每次运行都从磁盘上的树重新开始，两次运行会重新推导出同样的 issue。
- **agent 契约是 JSON 文本**，用容错提取器解析，而不是约束解码。

---

## 更多细节

英文的深入文档：`docs/architecture.md`（状态总线、评分、失败处理、扩展点、真实规模仓库上的三项机制）与 `docs/phases.md`（逐阶段契约）。

在**本仓库**里工作的 AI 助手，先读 `CLAUDE.md.sample` —— 那是一份模板，复制成 `CLAUDE.md`
后填上你自己项目的入口、命令与不变量即可（`CLAUDE.md` 已在 `.gitignore` 里，属于本地文件）。

---

## 许可

**AGPL-3.0-only**（`LICENSE` 是 AGPL-3.0 全文；`pyproject.toml` 里声明的是
`AGPL-3.0-only`，源码文件逐个带 `SPDX-License-Identifier: AGPL-3.0-only` 头）。用它改
自己的仓库、内部使用、二次分发都没问题；**但如果你把它作为网络服务提供给别人用**，
AGPL 要求你向使用者提供修改后的完整源码 —— 这是 AGPL 与 GPL 的唯一实质区别，也是选它的
原因。

`only` 表示**不授权**按更高版本的 AGPL 使用（例如将来的 AGPL-4.0）；要放开就把
`pyproject.toml` 与文件头里的 `-only` 换成 `-or-later`。
