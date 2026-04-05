"""File reading with line numbers."""

from pathlib import Path

from .base import Tool


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a file's contents with line numbers. Always read a file before editing it."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file",
            },
            "offset": {
                "type": "integer",
                "description": "Start line (1-based). Default 1.",
            },
            "limit": {
                "type": "integer",
                "description": "Max lines to read. Default 2000.",
            },
        },
        "required": ["file_path"],
    }

    def execute(self, file_path: str, offset: int = 1, limit: int = 2000) -> str:
        try:
            path = Path(file_path).expanduser().resolve()
            if not path.exists():
                return f"Error: {file_path} not found"
            if not path.is_file():
                return f"Error: {file_path} is a directory, not a file"

            text = path.read_text(errors="replace")
            lines = text.splitlines()
            total = len(lines)
            start = max(0, offset - 1)
            chunk = lines[start : start + limit]
            result = "\n".join(f"{start + index + 1}\t{line}" for index, line in enumerate(chunk))

            if total > start + limit:
                result += f"\n... ({total} lines total, showing {start + 1}-{start + len(chunk)})"
            return result or "(empty file)"
        except Exception as exc:
            return f"Error: {exc}"