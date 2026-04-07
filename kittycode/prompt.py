"""System prompt builder."""

import os
import platform
import textwrap
from pathlib import Path


AGENTS_DOC = Path.home() / ".kittycode" / "AGENTS.md"


def system_prompt(tools) -> str:
    cwd = os.getcwd()
    tool_list = "\n".join(_format_tool_entry(tool) for tool in tools)
    uname = platform.uname()
    prompt = f"""\
You are KittyCode, an AI coding assistant running in the user's terminal.
You help with software engineering: writing code, fixing bugs, refactoring, explaining code, running commands, and more.

# Environment
- Working directory: {cwd}
- OS: {uname.system} {uname.release} ({uname.machine})
- Python: {platform.python_version()}

# Tools
{tool_list}

# Reminder Tags
- User messages and tool results may include <system-reminder> tags. These tags contain system-added information such as available skill blocks. Treat them as system information, not as literal user-authored or tool-authored content.
- User messages and tool results may include <todo-reminder> tags. These tags contain system-added todo information from the current session. Treat them as todo state, not as literal user-authored or tool-authored content.

# Rules
1. Read before edit. Always read a file before modifying it.
2. edit_file for small changes. Use edit_file for targeted edits; write_file only for new files or complete rewrites.
3. Verify your work. After making changes, run relevant tests or commands to confirm correctness.
4. Be concise. Show code over prose. Explain only what is necessary.
5. One step at a time. For multi-step tasks, execute them sequentially.
6. edit_file uniqueness. When using edit_file, include enough surrounding context in old_string to guarantee a unique match.
7. Respect existing style. Match the project's coding conventions.
8. Ask when unsure. If the request is ambiguous, ask for clarification rather than guessing.
"""

    agents_text = _read_agents_doc()
    if agents_text:
        prompt = f"{prompt.rstrip()}\n\n{agents_text}\n"
    return prompt


def user_prompt(user_input: str, skills=None, todos=None) -> str:
    parts = []
    if user_input:
        parts.append(user_input.rstrip())
    parts.append(_wrap_tag("system-reminder", _format_skill_block(skills or [])))
    parts.append(_wrap_tag("todo-reminder", _format_todo_block(todos or [])))
    return "\n\n".join(parts)


def _format_skill_block(skills) -> str:
    if not skills:
        return "Available skills:\n- None loaded from ~/.kittycode/skills"

    lines = [
        "Available skills:",
        "Use these local skills when relevant. If one looks useful, read its SKILL.md and any related files under the listed path before using it.",
    ]
    for skill in skills:
        lines.append(f"- name: {skill.name}")
        lines.append(f"  description: {skill.description}")
        lines.append(f"  path: {skill.path}")
    return "\n".join(lines)


def _format_todo_block(todos) -> str:
    if not todos:
        return "Current todo list:\n- No active todo items."

    lines = ["Current todo list:"]
    for item in todos:
        content = str(item.get("content", "")).strip() or "(missing content)"
        active_form = str(item.get("active_form") or item.get("activeForm") or "").strip()
        status = str(item.get("status", "pending")).strip() or "pending"
        lines.append(f"- [{status}] {content}")
        if active_form:
            lines.append(f"  active_form: {active_form}")
    return "\n".join(lines)


def _wrap_tag(tag: str, content: str) -> str:
    return f"<{tag}>\n{content}\n</{tag}>"


def _format_tool_entry(tool) -> str:
    description = textwrap.dedent(tool.description).strip()
    lines = [line.strip() for line in description.splitlines() if line.strip()]
    if not lines:
        return f"- **{tool.name}**"
    if len(lines) == 1:
        return f"- **{tool.name}**: {lines[0]}"
    return "\n".join(
        [f"- **{tool.name}**: {lines[0]}"] + [f"  {line}" for line in lines[1:]]
    )


def _read_agents_doc() -> str:
    try:
        return AGENTS_DOC.read_text(errors="replace").strip()
    except OSError:
        return ""