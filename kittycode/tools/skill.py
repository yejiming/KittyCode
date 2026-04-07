"""Local skill loading and injection."""

from __future__ import annotations

from pathlib import Path

from ..skills import load_skills
from .base import Tool

_MAX_SKILL_BODY_CHARS = 20_000
_MAX_LISTED_FILES = 24


class SkillTool(Tool):
    name = "skill"
    _parent_agent = None
    description = """
    Load a local skill from ~/.kittycode/skills and inject its instructions
    into the current run. Available skill blocks are surfaced through
    <system-reminder> tags in the conversation. Use this when one of the
    listed skills matches the user's request.
    """
    parameters = {
        "type": "object",
        "properties": {
            "skill": {
                "type": "string",
                "description": "Skill name, for example 'commit' or 'review-pr'",
            },
            "task": {
                "type": "string",
                "description": "Optional task the selected skill should be applied to",
            },
            "args": {
                "type": "string",
                "description": "Optional free-form arguments for compatibility with slash-style skills",
            },
        },
        "required": ["skill"],
    }

    def execute(self, skill: str, task: str | None = None, args: str | None = None) -> str:
        normalized = skill.strip().lstrip("/")
        if not normalized:
            return "Error: skill is required"

        available_skills = self._available_skills()
        selected = _find_skill(normalized, available_skills)
        if selected is None:
            available = ", ".join(skill_def.name for skill_def in available_skills) or "(none)"
            return f'Error: unknown skill "{normalized}". Available skills: {available}'

        skill_doc = Path(selected.path) / "SKILL.md"
        try:
            skill_body = skill_doc.read_text(errors="replace").strip()
        except OSError as exc:
            return f"Error reading {skill_doc}: {exc}"

        if len(skill_body) > _MAX_SKILL_BODY_CHARS:
            skill_body = skill_body[:_MAX_SKILL_BODY_CHARS] + "\n... (skill instructions truncated)"

        task_text = (task or args or "").strip()
        related_files = _list_skill_files(Path(selected.path))

        lines = [
            f'Skill "{selected.name}" selected.',
            f"Description: {selected.description}",
            f"Path: {selected.path}",
        ]
        if related_files:
            lines.append("Related files:")
            lines.extend(f"- {file_path}" for file_path in related_files)

        lines.extend(["", "SKILL.md:", skill_body])

        if task_text:
            lines.extend(["", "Apply this skill to the following task:", task_text])

        lines.extend(
            [
                "",
                "Follow the skill instructions above before continuing with the task.",
            ]
        )
        return "\n".join(lines).strip()

    def _available_skills(self):
        if self._parent_agent is not None:
            return list(getattr(self._parent_agent, "skills", []))
        return load_skills()


def _find_skill(name: str, skills):
    target = name.casefold()
    for skill in skills:
        if skill.name.casefold() == target:
            return skill
        if Path(skill.path).name.casefold() == target:
            return skill
    return None


def _list_skill_files(root: Path) -> list[str]:
    files: list[str] = []
    skill_doc = root / "SKILL.md"
    if skill_doc.is_file():
        files.append(str(skill_doc.resolve()))

    for path in sorted(root.rglob("*")):
        if len(files) >= _MAX_LISTED_FILES:
            break
        if not path.is_file() or path.name == "SKILL.md":
            continue
        if any(part.startswith(".") and part != "." for part in path.relative_to(root).parts):
            continue
        if "__pycache__" in path.parts:
            continue
        files.append(str(path.resolve()))

    return files