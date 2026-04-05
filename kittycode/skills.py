"""Local skill discovery for KittyCode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

SKILLS_DIR = Path.home() / ".kittycode" / "skills"

_NAME_PATTERN = re.compile(r'^\s*name:\s*["\']?(.*?)["\']?\s*$')
_DESCRIPTION_PATTERN = re.compile(r'^\s*description:\s*["\']?(.*?)["\']?\s*$')

_cached_skills: tuple["SkillDefinition", ...] | None = None
_cached_root: Path | None = None
_cached_signature: tuple | None = None


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    description: str
    path: str


def load_skills(skills_dir: Path | str | None = None, force_reload: bool = False) -> list[SkillDefinition]:
    """Load skill metadata from ~/.kittycode/skills into process memory."""
    global _cached_root
    global _cached_skills
    global _cached_signature

    root = Path(skills_dir).expanduser() if skills_dir is not None else SKILLS_DIR
    root = root.resolve()
    signature = _build_signature(root)

    if (
        not force_reload
        and _cached_skills is not None
        and _cached_root == root
        and _cached_signature == signature
    ):
        return list(_cached_skills)

    if not root.exists() or not root.is_dir():
        _cached_root = root
        _cached_skills = ()
        _cached_signature = signature
        return []

    skills: list[SkillDefinition] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        skill_doc = entry / "SKILL.md"
        if not skill_doc.is_file():
            continue
        skills.append(_read_skill(entry, skill_doc))

    _cached_root = root
    _cached_skills = tuple(skills)
    _cached_signature = signature
    return list(_cached_skills)


def _build_signature(root: Path) -> tuple:
    if not root.exists() or not root.is_dir():
        return ("missing", str(root))

    items = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        skill_doc = entry / "SKILL.md"
        if not skill_doc.is_file():
            continue
        stat = skill_doc.stat()
        items.append((entry.name, stat.st_mtime_ns, stat.st_size))
    return tuple(items)


def _read_skill(skill_dir: Path, skill_doc: Path) -> SkillDefinition:
    name, description = _parse_skill_header(skill_doc.read_text(errors="replace"))
    return SkillDefinition(
        name=name or skill_dir.name,
        description=description or "No description provided.",
        path=str(skill_dir.resolve()),
    )


def _parse_skill_header(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    header_lines = lines[:40]

    name = ""
    description = ""

    for line in header_lines:
        if not name:
            match = _NAME_PATTERN.match(line)
            if match:
                name = match.group(1).strip()
                continue
        if not description:
            match = _DESCRIPTION_PATTERN.match(line)
            if match:
                description = match.group(1).strip()

        if name and description:
            return name, description

    for line in lines[:20]:
        stripped = line.strip()
        if stripped.startswith("# "):
            name = name or stripped[2:].strip()
            break

    if not description:
        for line in lines[:40]:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped == "---":
                continue
            description = stripped
            break

    return name, description