# KittyCode

KittyCode 是一个运行在终端里的轻量级 AI 编程代理，目标是在尽量精简代码的前提下保留清晰、直接的核心运行逻辑，包括主代理循环、本地工具调用、上下文压缩、命令行交互，以及对 OpenAI 兼容接口和 Anthropic 接口的支持。

## 项目背景

KittyCode 的基本工作方式如下：

- 用户输入先通过当前配置的模型接口发送出去。
- 模型可以直接回答，也可以发起工具调用。
- 本地执行工具后，再把结果回填给模型继续推理。
- 直到模型返回普通文本，当前任务才结束。

这个项目刻意保持小而清晰，只保留最核心的运行时能力，包括代理主循环、命令行界面、会话保存、上下文压缩，以及默认工具集。

## 功能特点

- 保留核心 agent loop，并支持多工具并行执行。
- LLM 适配层同时支持 OpenAI 兼容接口和 Anthropic 接口。
- 内置 Bash、读文件、写文件、精确替换编辑、Glob 搜索、Grep 搜索、子代理等工具。
- 启动时自动扫描 `~/.kittycode/skills`，并在每轮系统 prompt 开头注入可用 skill 列表。
- 支持交互式 REPL 和单次命令模式。
- 启动时展示 ANSI 彩色像素猫。
- 支持在执行过程中按 `Esc` 中断当前任务。
- 支持长会话上下文压缩。
- 支持保存和恢复历史会话。

## 环境要求

- Python 3.10 及以上
- OpenAI 兼容接口或 Anthropic 接口所需的 API Key

## 安装方法

进入项目目录后，以可编辑模式安装：

```bash
cd KittyCode
python -m pip install -e .
```

如果还要安装测试依赖：

```bash
python -m pip install -e .[dev]
```

## 配置说明

KittyCode 启动时会从 `~/.kittycode/config.json` 读取配置。

支持的字段包括：

- `interface`：`openai` 或 `anthropic`
- `api_key`
- `model`
- `base_url`
- `max_tokens`
- `temperature`
- `max_context`

OpenAI 兼容接口示例：

```json
{
	"interface": "openai",
	"api_key": "sk-...",
	"model": "gpt-4o",
	"base_url": "https://api.openai.com/v1",
	"max_tokens": 4096,
	"temperature": 0,
	"max_context": 128000
}
```

Anthropic 接口示例：

```json
{
	"interface": "anthropic",
	"api_key": "sk-ant-...",
	"model": "claude-3-7-sonnet-latest",
	"base_url": "https://api.anthropic.com",
	"max_tokens": 4096,
	"temperature": 0,
	"max_context": 128000
}
```

如果需要，也可以通过命令行参数临时覆盖这些配置，例如 `--model`、`--interface`、`--base-url`、`--api-key`。

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

每轮对话前，KittyCode 都会检查 skill 目录是否发生变化；如果新增或修改了 skill，会自动刷新内存缓存，因此不需要重启进程。

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

临时覆盖模型、接口类型或接口地址：

```bash
kittycode --interface anthropic --model claude-3-7-sonnet-latest
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

`/skills` 命令会在检测到 skill 目录变化时刷新本地缓存，并输出当前已加载的 skill 列表。
当输入以 `/` 开头时，CLI 还会根据前缀自动补全可用命令和 skill。

## 如果执行 `pip install -e .` 后仍然找不到 `kittycode`

项目本身已经在 `pyproject.toml` 中声明了控制台入口：

```toml
[project.scripts]
kittycode = "kittycode.cli:main"
```

所以如果仍然不能直接执行 `kittycode`，通常不是因为项目没有声明入口，而是 Python 本地安装路径或 shell `PATH` 的问题。

先确认你实际使用的是哪个 Python/pip：

```bash
python3 -m pip --version
python3 -m site --user-base
```

在 macOS 上，用户级安装通常会把脚本放到类似下面的目录：

```bash
~/Library/Python/3.11/bin
```

如果这个目录不在 `PATH` 中，即使 editable install 成功，shell 也依然找不到 `kittycode`。

对于 `zsh`，可以把对应目录加入配置：

```bash
export PATH="$HOME/Library/Python/3.11/bin:$PATH"
```

然后重新加载 shell：

```bash
source ~/.zshrc
```

如果安装过程本身在往 `~/Library/Python/.../site-packages` 写 `.pth` 文件时就报权限错误，那要先修复本地 Python 环境权限，或者改用虚拟环境后再重新安装。

## 目录结构

- `kittycode/agent.py`：核心代理循环
- `kittycode/llm.py`：流式 LLM 封装
- `kittycode/context.py`：上下文压缩
- `kittycode/cli.py`：交互式与单次命令入口
- `kittycode/session.py`：会话持久化
- `kittycode/tools/`：内置工具集合
- `tests/`：核心运行时与工具测试

## 开发与测试

运行测试：

```bash
python -m pytest -q
```

当前测试主要覆盖导出 API、config.json 读取、provider 转换辅助逻辑、上下文压缩、会话辅助函数、默认工具注册表，以及 skill 发现与 prompt 注入逻辑。
