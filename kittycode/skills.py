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
    for skill_dir, skill_doc in _iter_skill_docs(root):
        skill = _read_skill(skill_dir, skill_doc)
        if skill is not None:
            skills.append(skill)

    _cached_root = root
    _cached_skills = tuple(skills)
    _cached_signature = signature
    return list(_cached_skills)


def _build_signature(root: Path) -> tuple:
    if not root.exists() or not root.is_dir():
        return ("missing", str(root))

    items = []
    for entry, skill_doc in _iter_skill_docs(root):
        stat = skill_doc.stat()
        items.append((entry.relative_to(root).as_posix(), stat.st_mtime_ns, stat.st_size))
    return tuple(items)


def _iter_skill_docs(root: Path):
    candidates: list[Path] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        candidates.append(entry)
        for child in entry.iterdir():
            if child.is_dir():
                candidates.append(child)

    for skill_dir in sorted(candidates, key=lambda path: path.relative_to(root).as_posix()):
        skill_doc = skill_dir / "SKILL.md"
        if skill_doc.is_file():
            yield skill_dir, skill_doc


def _read_skill(skill_dir: Path, skill_doc: Path) -> SkillDefinition | None:
    name, description = _parse_skill_header(skill_doc.read_text(errors="replace"))
    if not name or not description:
        return None
    return SkillDefinition(
        name=name,
        description=description,
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

    return name, description