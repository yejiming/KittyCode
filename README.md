# KittyCode

KittyCode is a minimal terminal coding agent focused on a compact, readable implementation. It keeps the core runtime straightforward, with an agent loop, local tools, context compression, a small command-line interface, and support for both OpenAI-compatible and Anthropic APIs.

## Background

KittyCode is inspired by [NanoCoder](https://github.com/he-yufeng/NanoCoder) and keeps the same general idea of a small terminal coding agent while simplifying the project around the core runtime.

KittyCode follows a simple terminal-agent runtime model:

- A user message is sent through the configured model interface.
- The model can either answer directly or call tools.
- Tool calls are executed locally and the results are fed back into the conversation.
- The loop continues until the model returns plain text.

The project is intentionally small. It includes the core agent runtime, a compact CLI, session persistence, context compression, and a default built-in tool set.

## Features

- Minimal agent loop with optional parallel execution for multiple tool calls.
- LLM adapter that supports both OpenAI-compatible and Anthropic interfaces.
- Core built-in tools for shell execution, file operations, web search, web fetch/crawling, TODO tracking, and sub-agents.
- Interactive REPL and one-shot command mode, with `Esc` interrupt support.
- Context compression to keep long sessions manageable.
- Session save and resume support.

## Requirements

- Python 3.10 or newer
- An API key for either an OpenAI-compatible endpoint or an Anthropic-compatible endpoint

## Installation

Install from PyPI:

```bash
pip install kittycode
```

## Configuration

KittyCode reads startup configuration from `~/.kittycode/config.json`.

Supported fields:

- `interface`: `openai` or `anthropic`
- `api_key`
- `model`
- `base_url`
- `max_tokens`
- `temperature`
- `max_context`

OpenAI-compatible example:

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

Anthropic example:

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

The CLI still allows explicit overrides such as `--model`, `--interface`, `--base-url`, and `--api-key`.

## Skills

At startup, KittyCode scans `~/.kittycode/skills` for skill folders. Each skill should live in its own directory and include a `SKILL.md` file at the top level.

Expected layout:

```text
~/.kittycode/skills/
	example-skill/
		SKILL.md
		other-files...
```

KittyCode reads the leading `name` and `description` fields from each `SKILL.md`, keeps the resulting skill list in memory, and inserts the list into the system prompt at the start of every round. The prompt includes:

- `name`
- `description`
- `path`

This allows the model to see which local skills are available and decide when to read and use them.

Before each round, KittyCode checks whether the skill directory changed and reloads the cached skill metadata when needed, so adding or editing skills does not require restarting the process.

You can also invoke a loaded skill directly from the CLI with `/<skill name>`.

- `/<skill name>` selects that skill for your next non-command message.
- `/<skill name> <task>` runs the next request immediately with that skill.

## Usage

Run the interactive terminal UI:

```bash
kittycode
```

When the CLI is busy, press `Esc` to interrupt the current run. The stop is cooperative: KittyCode cancels at the next safe checkpoint between LLM streaming, tool dispatch, and loop rounds. A blocking external call may still need to return before the run fully stops.

You can also use the module entry point:

```bash
python -m kittycode
```

Run a one-shot prompt and exit:

```bash
kittycode -p "Explain the project structure"
```

Resume a saved session:

```bash
kittycode -r session_1234567890
```

Override model, interface, or endpoint from the command line:

```bash
kittycode --interface anthropic --model claude-3-7-sonnet-latest
```

## Interactive Commands

Inside the REPL, KittyCode supports:

- `/help`
- `/reset`
- `/skills`
- `/<skill name>`
- `/model <name>`
- `/tokens`
- `/compact`
- `/save`
- `/sessions`
- `/quit`

The `/skills` command refreshes the local skill cache if the skill directory has changed and then prints the currently loaded skills.
Slash commands also support prefix matching while typing, so entering `/` shows matching commands and skills through completion suggestions.

## Project Layout

- `kittycode/agent.py`: core agent loop
- `kittycode/llm.py`: streaming LLM wrapper
- `kittycode/context.py`: context compression
- `kittycode/cli.py`: interactive and one-shot CLI
- `kittycode/session.py`: session persistence
- `kittycode/tools/`: built-in tools
- `tests/`: focused runtime and tool tests

## Development

Run the test suite:

```bash
python -m pytest -q
```

The current test suite covers the exported API, config-file behavior, provider conversion helpers, context compression, session helpers, the default tool registry, and skill discovery/prompt injection.
