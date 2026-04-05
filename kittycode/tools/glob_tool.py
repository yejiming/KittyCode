"""File pattern matching."""

from pathlib import Path

from .base import Tool


class GlobTool(Tool):
    name = "glob"
    description = "Find files matching a glob pattern. Supports ** for recursive matching."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern, for example '**/*.py' or 'src/**/*.ts'",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: cwd)",
            },
        },
        "required": ["pattern"],
    }

    def execute(self, pattern: str, path: str = ".") -> str:
        try:
            base = Path(path).expanduser().resolve()
            if not base.is_dir():
                return f"Error: {path} is not a directory"

            hits = list(base.glob(pattern))
            hits.sort(key=lambda item: item.stat().st_mtime if item.exists() else 0, reverse=True)

            total = len(hits)
            shown = hits[:100]
            result = "\n".join(str(item) for item in shown)
            if total > 100:
                result += f"\n... ({total} matches, showing first 100)"
            return result or "No files matched."
        except Exception as exc:
            return f"Error: {exc}"