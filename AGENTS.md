# KittyCode Agent Guide

## Project Goal

- Maintain KittyCode as a compact terminal coding agent with a readable, minimal runtime.
- Preserve the core behavior of the agent loop, CLI, tool execution, prompt assembly, context compression, skill loading, and provider adapters.
- Prefer small, composable modules over framework-heavy abstractions.

## Repository Structure

### Root Files

- `pyproject.toml`: packaging metadata, dependencies, and CLI entrypoint declaration.
- `README.md`: English user-facing overview, setup, and usage documentation.
- `README_CN.md`: Chinese user-facing overview, setup, and usage documentation.
- `task_plan.md`: active phase tracking and implementation decisions.
- `findings.md`: verified discoveries, constraints, and architecture notes.
- `progress.md`: chronological execution log and verification history.
- `AGENTS.md`: this development guide for AI contributors.

### Package Layout: `kittycode/`

- `kittycode/__init__.py`: public exports and compatibility aliases for legacy module paths.
- `kittycode/main.py`: lightweight entrypoint wrapper used by the installed `kittycode` command and `python -m kittycode`.
- `kittycode/cli.py`: interactive REPL, one-shot mode, terminal rendering, interrupt handling, and ask-user orchestration.
- `kittycode/config/__init__.py`: thin package export surface for configuration APIs.
- `kittycode/config/settings.py`: `Config` dataclass and `~/.kittycode/config.json` loading.
- `kittycode/llm/__init__.py`: thin package export surface for LLM APIs and helper conversions used by tests.
- `kittycode/llm/provider.py`: OpenAI/Anthropic adapter, streaming assembly, retry logic, and token accounting.
- `kittycode/prompt/__init__.py`: thin package export surface for prompt-building APIs.
- `kittycode/prompt/builder.py`: system prompt builder, user prompt builder, reminder-tag formatting, and AGENTS injection.
- `kittycode/skills/__init__.py`: thin package export surface for skill-discovery APIs.
- `kittycode/skills/discovery.py`: local skill discovery, header parsing, and cache signature logic.
- `kittycode/runtime/__init__.py`: runtime package marker; keep it lightweight to avoid circular imports.
- `kittycode/runtime/agent.py`: core agent loop, run-local state snapshots, tool execution, and interrupt repair.
- `kittycode/runtime/context.py`: token estimation and context compression.
- `kittycode/runtime/interrupts.py`: cancellation exceptions for cooperative interruption.
- `kittycode/runtime/session.py`: session save, load, and listing helpers.
- `kittycode/runtime/logging.py`: file logging configuration.
- `kittycode/tools/__init__.py`: built-in tool registry and instance factory.
- `kittycode/tools/base.py`: shared tool interface.
- `kittycode/tools/agent.py`: sub-agent execution tool.
- `kittycode/tools/ask_user.py`: interactive ask-user bridge tool.
- `kittycode/tools/bash.py`: shell command execution with safety checks and streamed output.
- `kittycode/tools/brief.py`: brief user-facing update tool.
- `kittycode/tools/edit.py`: targeted file edit tool.
- `kittycode/tools/glob_tool.py`: glob-based file discovery tool.
- `kittycode/tools/grep.py`: regex content search tool.
- `kittycode/tools/read.py`: file reading tool.
- `kittycode/tools/skill.py`: local skill-loading tool.
- `kittycode/tools/todo_write.py`: todo state update tool.
- `kittycode/tools/web_fetch.py`: fetch-and-summarize web content tool.
- `kittycode/tools/web_search.py`: web search tool.
- `kittycode/tools/write.py`: file writing tool.

### Tests

- `tests/test_core.py`: exports, config, context, logging, session helpers, and core agent behavior.
- `tests/test_cli.py`: CLI formatting, prompt rendering, and helper behavior.
- `tests/test_llm.py`: provider conversions, streaming parsing, and retry/cancel behavior.
- `tests/test_skills.py`: skill discovery, prompt injection, and slash command behavior.
- `tests/test_tools.py`: tool registry and tool behavior.
- `tests/test_interrupt_latency.py`: interrupt timing and state isolation.
- `tests/test_bash_streaming.py`: bash live-output rendering behavior.
- `tests/test_prompt.md`: prompt fixture/reference content used by tests.

## Development Rules

- Practice TDD when behavior changes: add or update a failing test first whenever practical.
- Reuse existing functions and modules before adding new helpers or parallel code paths.
- Do not introduce redundant variables, redundant helper functions, or duplicated logic.
- Prefer small, explicit functions with clear inputs and outputs.
- Keep the `kittycode/` root lean: only `__init__.py`, `main.py`, and `cli.py` should remain as top-level files there.
- Preserve current user-visible behavior unless the task explicitly requires a behavior change.
- Update docs when code moves or user-visible behavior changes.
- Keep `kittycode/runtime/__init__.py` lightweight to avoid circular imports.

## Constraints

- Every code change must pass the full test suite: `python -m pytest -q`.
- Do not leave the project with broken imports, half-moved modules, or stale structure documentation.
- If a package move breaks legacy import paths still used by tests or callers, add a compatibility layer instead of forcing a broad rewrite.
- Keep the implementation simple and readable; avoid unnecessary architectural expansion.
- When changing the agent loop, tools, or CLI interrupt flow, verify both normal completion and interruption paths.
