"""System prompt builder."""

import os
import platform
import textwrap


def system_prompt(tools, skills=None) -> str:
    cwd = os.getcwd()
    tool_list = "\n".join(_format_tool_entry(tool) for tool in tools)
    uname = platform.uname()
    skill_block = _format_skill_block(skills or [])

    return f"""\
{skill_block}

You are KittyCode, an AI coding assistant running in the user's terminal.
You help with software engineering: writing code, fixing bugs, refactoring, explaining code, running commands, and more.

# Environment
- Working directory: {cwd}
- OS: {uname.system} {uname.release} ({uname.machine})
- Python: {platform.python_version()}

# Tools
{tool_list}

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


def _format_skill_block(skills) -> str:
    if not skills:
        return "# Available Skills\n- None loaded from ~/.kittycode/skills"

    lines = [
        "# Available Skills",
        "Use these local skills when relevant. If one looks useful, read its SKILL.md and any related files under the listed path before using it.",
    ]
    for skill in skills:
        lines.append(f"- name: {skill.name}")
        lines.append(f"  description: {skill.description}")
        lines.append(f"  path: {skill.path}")
    return "\n".join(lines)


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