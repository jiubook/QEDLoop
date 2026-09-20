# targets/ —— 目标仓库的专属设定

**约定：任何针对某个被测仓库的东西都放在这里，不写进那个仓库。**

```
targets/
  README.md            本文件：约定说明
  eer-ai/
    loop.yml           QEDLoop 运行配置（--config 指向它）
    brief.md           注入到每个 agent 的项目约束（loop.yml 的 brief: 指向它）
    skill.md           需要时放更长的操作手册（目前没有机制会读，只给人看）
```

为什么这么分：

- 被测仓库保持**内容零污染** —— 跑完一次闭环，那个仓库的 `git status` 不该有任何变化（`run` 全程只读副本，`apply` 才写，且由你逐次决定）。
- 目标仓库的配置属于**这个工作区**，不属于它自己：换个项目只要新增一个 `targets/<id>/`，不会在各个仓库里散落配置文件。
- 一个目标仓库往往不止一份文件（配置 + 约束 + 手册），所以按目录组织，而不是平铺 `xxx.yml`。

## 用法

```powershell
cd D:\work\QEDLoop

python run.py channels --config targets\eer-ai\loop.yml
python run.py run      --config targets\eer-ai\loop.yml --single-cycle
python run.py run      --config targets\eer-ai\loop.yml
```

> 上面的 `eer-ai` 是本地目标的名字。**`targets/*/` 不进仓库**（里面有本机绝对路径和
> 本机环境变量名，见仓库根 `.gitignore`），所以 clone 下来之后这个目录并不存在，
> 是你自己按同一套结构新建一个 `targets\<你的目标 id>\`。

`loop.yml` 里的 `target` 与 `out_dir` 都写绝对路径，因此在哪个目录调用都成立。
路径规则只有一条要记：**`target` / `out_dir` 相对当前工作目录，`brief` 相对配置文件
所在目录**（前者是你在命令行选定的运行位置，后者是配置自带的附件）。

运行产物按目标分目录：`runs\<target-id>\<run_id>\`。

## `brief.md` —— 给 agent 的项目约束

`loop.yml` 里的 `brief:` 指向这份文件，它的内容会作为 `<project_constraints>` 段落
**注入到每个 agent 的系统提示**（15 个 agent × 每一轮）。所以它必须短：上限 4000 字符，
超了会截断并在提示里注明。写「光看代码看不出来」的硬规则，例如

- 这个仓库只支持 Windows、不引入新的第三方依赖
- 不许碰生成物（`version.json`、`profiles.json`）与哪些目录不在范围内
- 改某个 schema 必须配套哪几步迁移

三条设计约定：

1. **相对路径相对配置文件所在目录**，不是当前工作目录 —— `brief: brief.md` 就是同目录
   那份。`target` / `out_dir` 则是运行期地址，相对当前工作目录，所以本目录里都写绝对路径。
2. **读不到就报错**（`config error: brief file not found: ...`，退出码 2），不会静默当成
   「没有约束」—— 静默忽略一份维护者以为已经生效的规则，比直接失败危险得多。
3. **约束不是证据**：注入的段落里明确写了这一点。agent 不能拿 brief 当「这里有缺陷」的
   依据，测试结论也只能来自实测。

运行报告与 `state.json` 会记录用的是哪份 brief、多少字符（`brief: {source, chars}`），
但不重复正文：正文在状态总线上（`state["brief"]`），也出现在每一次模型请求里，落盘的
只有「这次运行受哪份规则约束」这个事实。

## 目标仓库自己的静态检查（可选）

`loop.yml` 里可以设一条 `lint_command`：闭环会在**候选树副本**里执行它，退出码非零即计入
`block_reasons`——与「能编译」「测试全绿」同级，是硬性条件，不是分数项。

为什么需要它：闭环只证明「能编译 + 测试全绿」，两者都看不见 lint 规则。实测代价——某轮
补丁的**唯一**缺陷是 `TC003`（标准库 import 该放进 `TYPE_CHECKING` 块），507 个测试全绿，
而 EER-Ai 自己的 ruff 拒绝它，提交上去 CI 就是红的。

| 占位符 | 展开为 |
| --- | --- |
| `{python}` | 沙箱解释器，与跑测试用的是同一个 |
| `{files}` | **本轮改动的文件**（一个文件一个参数） |

三条约定：

1. **用 `{python}`，不要用 `uv run` / `npx` / `cargo`**。沙箱副本排除了 `.venv` 与
   `node_modules`，这些入口会**每轮重建一次环境**（联网、分钟级），而且用的是与实测不同
   的解释器——那就又变成「两个环境给出两个答案」。
2. **尽量写上 `{files}`**。只检查补丁碰过的文件，失败才归因得到这一轮；全仓库检查会撞上
   闭环无权修改的既有违规（`run.include` 之外的文件），变成永远修不掉的 blocker。
3. **命令必须只读，不要加 `--fix`**。会改文件的检查既报失败又顺手修好，判定就落在一个
   已经不存在的树上，而且「落盘的字节 == Phase 5 验证过的字节」这条承诺会被破坏。

**配了但跑不起来（命令写错、解释器找不到）算没通过**，不会静默放行——否则一个 typo 就
等于把这道门悄悄关掉了。

版本差异要知道：命令用的是环境里的 linter，目标仓库 CI 钉的可能是另一个版本（EER-Ai 的
pre-commit 是 `ruff-pre-commit v0.15.0`，而环境里是 ruff 0.15.9）。这道门拦得住「明显过
不了 lint」的补丁，**不保证与 CI 逐条等价**。

## 还可以放什么

- `skill.md`：需要时放更长的操作手册（例如这个仓库的发布流程），但目前没有机制会读它，
  只有人和 AI 助手会看。要让它进提示词，走 `brief:`（并注意 4000 字符的上限）。

给 AI 助手用的项目级约定（本仓库自己的）走 `../CLAUDE.md.sample` 那份模板：复制成
`../CLAUDE.md` 再填。两者读者不同 —— `brief.md` 喂给**闭环里那 15 个 agent**，
`CLAUDE.md` 喂给**帮你改 QEDLoop 代码的 agent**。
