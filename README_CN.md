# KittyCode

KittyCode 是一个运行在终端里的轻量级 AI 编程代理，目标是在尽量精简代码的前提下保留清晰、直接的核心运行逻辑，包括主代理循环、本地工具调用、上下文压缩、命令行交互，以及对 OpenAI 兼容接口和 Anthropic 接口的支持。

## 快速开始

1. 安装 KittyCode：

```bash
pip install kittycode
```

2. 通过引导式配置完成初始化：

```bash
kittycode --config
```

这是创建或修复 `~/.kittycode/config.json` 的推荐方式。

3. 启动交互式终端界面：

```bash
kittycode
```

## 项目背景

KittyCode 受到 [NanoCoder](https://github.com/he-yufeng/NanoCoder) 的启发，延续了“小型终端编程代理”这一思路，但把项目进一步收缩到更核心、更直接的运行时能力。

KittyCode 的基本工作方式如下：

- 用户输入先通过当前配置的模型接口发送出去。
- 模型可以直接回答，也可以发起工具调用。
- 本地执行工具后，再把结果回填给模型继续推理。
- 直到模型返回普通文本，当前任务才结束。

这个项目刻意保持小而清晰，只保留最核心的运行时能力，包括代理主循环、命令行界面、会话保存、上下文压缩，以及默认工具集。

## 功能特点

- 保留核心 agent loop，并支持多工具并行执行。
- LLM 适配层同时支持 OpenAI 兼容接口和 Anthropic 接口。
- 内置 Shell 执行、文件操作、网页搜索、网页抓取、TODO 跟踪和子代理等核心工具。
- 支持交互式 REPL 和单次命令模式，并可通过 `Esc` 中断当前任务。
- 支持长会话上下文压缩。
- 支持保存和恢复历史会话。

## 环境要求

- Python 3.10 及以上
- OpenAI 兼容接口或 Anthropic 接口所需的 API Key

## Skills 机制

KittyCode 启动时会扫描 `~/.kittycode/skills` 目录。每个 skill 使用一个独立文件夹，文件夹顶层需要有一个 `SKILL.md`。

目录结构示例：

```text
~/.kittycode/skills/
	example-skill/
		SKILL.md
		other-files...
```
运行时会从每个 `SKILL.md` 开头读取 `name` 和 `description`，并把得到的 skill 列表缓存在进程内存中。每一轮对话时，系统 prompt 最前面都会包含以下字段：

- `name`
- `description`
- `path`

这样模型就能先看到本地有哪些 skill，再按需读取对应目录下的 `SKILL.md` 和其他相关文件。

KittyCode 只会在启动时加载一次 skill，并在整个进程生命周期内保持这份快照不变。新增或修改 skill 后，需要重启 KittyCode 才会生效。

你也可以在 CLI 中直接通过 `/<skill 名称>` 使用某个已加载的 skill。

- `/<skill 名称>`：选中该 skill，下一条普通输入会自动带上这个 skill。
- `/<skill 名称> <任务>`：立即用这个 skill 执行当前请求。

## 使用方法

启动交互式终端界面：

```bash
kittycode
```
当 CLI 正在执行时，可以按 `Esc` 中断当前任务。这个中断是协作式的：KittyCode 会在下一处安全检查点停止，例如 LLM 流式输出过程中、工具调度前后或下一轮循环开始前。如果某个底层外部调用本身是阻塞的，仍然可能要等该调用返回后才会完全结束。

也可以直接通过模块启动：

```bash
python -m kittycode
```
单次执行一个提示词并退出：

```bash
kittycode -p "Explain the project structure"
```
恢复历史会话：

```bash
kittycode -r session_1234567890
```
运行配置向导：

```bash
kittycode --config
```
## 交互命令

在 REPL 中可用的命令有：

- `/help`
- `/reset`
- `/skills`
- `/<skill 名称>`
- `/model <name>`
- `/tokens`
- `/compact`
- `/save`
- `/sessions`
- `/quit`

`/skills` 命令会输出启动时已加载的 skill 列表。
当输入以 `/` 开头时，CLI 还会根据前缀自动补全可用命令和 skill。

## 目录结构

- `kittycode/__init__.py`：对外导出与旧模块路径兼容层
- `kittycode/main.py`：CLI 入口封装
- `kittycode/cli.py`：交互式与单次命令入口
- `kittycode/config/__init__.py`：config.json 加载
- `kittycode/llm/__init__.py`：provider 适配与流式解析
- `kittycode/prompt/__init__.py`：system/user prompt 构建
- `kittycode/skills/__init__.py`：本地 skill 发现与缓存
- `kittycode/runtime/`：代理循环、上下文压缩、中断、会话与日志支持
- `kittycode/tools/`：内置工具集合
- `tests/`：核心运行时与工具测试

## 开发与测试

运行测试：

```bash
python -m pytest -q
```
当前测试主要覆盖导出 API、config.json 读取、provider 转换辅助逻辑、上下文压缩、会话辅助函数、默认工具注册表，以及 skill 发现与 prompt 注入逻辑。
