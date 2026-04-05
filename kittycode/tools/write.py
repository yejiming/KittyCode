"""File creation and overwrite."""

from pathlib import Path

from .base import Tool


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Create a new file or completely overwrite an existing one. "
        "For small edits to existing files, prefer edit_file instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path for the file",
            },
            "content": {
                "type": "string",
                "description": "Full file content to write",
            },
        },
        "required": ["file_path", "content"],
    }

    def execute(self, file_path: str, content: str) -> str:
        try:
            path = Path(file_path).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            return f"Wrote {line_count} lines to {file_path}"
        except Exception as exc:
            return f"Error: {exc}"