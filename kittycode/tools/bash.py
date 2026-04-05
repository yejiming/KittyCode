"""Shell command execution with basic safety checks."""

import os
import queue
import re
import subprocess
import threading
import time

from ..interrupts import CancellationRequested

from .base import Tool

_cwd: str | None = None

_DANGEROUS_PATTERNS = [
    (r"\brm\s+(-\w*)?-r\w*\s+(/|~|\$HOME)", "recursive delete on home/root"),
    (r"\brm\s+(-\w*)?-rf\s", "force recursive delete"),
    (r"\bmkfs\b", "format filesystem"),
    (r"\bdd\s+.*of=/dev/", "raw disk write"),
    (r">\s*/dev/sd[a-z]", "overwrite block device"),
    (r"\bchmod\s+(-R\s+)?777\s+/", "chmod 777 on root"),
    (r":\(\)\s*\{.*:\|:.*\}", "fork bomb"),
    (r"\bcurl\b.*\|\s*(sudo\s+)?bash", "pipe curl to bash"),
    (r"\bwget\b.*\|\s*(sudo\s+)?bash", "pipe wget to bash"),
]


class BashTool(Tool):
    name = "bash"
    description = (
        "Execute a shell command. Returns stdout, stderr, and exit code. "
        "Use this for running tests, installing packages, git operations, and similar tasks."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to run",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 120)",
            },
        },
        "required": ["command"],
    }

    def execute(
        self,
        command: str,
        timeout: int = 120,
        stream_callback=None,
        cancel_event=None,
    ) -> str:
        global _cwd

        warning = _check_dangerous(command)
        if warning:
            return f"Blocked: {warning}\nCommand: {command}\nIf intentional, modify the command to be more specific."

        cwd = _cwd or os.getcwd()

        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=cwd,
            )
        except Exception as exc:
            return f"Error running command: {exc}"

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        line_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        reader_threads = [
            threading.Thread(
                target=_read_stream_lines,
                args=(process.stdout, "stdout", line_queue),
                daemon=True,
            ),
            threading.Thread(
                target=_read_stream_lines,
                args=(process.stderr, "stderr", line_queue),
                daemon=True,
            ),
        ]
        for thread in reader_threads:
            thread.start()

        deadline = time.monotonic() + timeout

        try:
            while process.poll() is None or any(thread.is_alive() for thread in reader_threads) or not line_queue.empty():
                if cancel_event is not None and cancel_event.is_set():
                    _terminate_process(process)
                    raise CancellationRequested()

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _terminate_process(process)
                    return f"Error: timed out after {timeout}s"

                try:
                    source, line = line_queue.get(timeout=min(0.05, remaining))
                except queue.Empty:
                    continue

                if source == "stdout":
                    stdout_lines.append(line)
                    if stream_callback is not None:
                        stream_callback(line)
                else:
                    stderr_lines.append(line)
                    if stream_callback is not None:
                        stream_callback(f"[stderr] {line}")
        finally:
            for thread in reader_threads:
                thread.join(timeout=0.1)

        returncode = process.wait()

        if returncode == 0:
            _update_cwd(command, cwd)

        output = "\n".join(stdout_lines)
        if stderr_lines:
            stderr_output = "\n".join(stderr_lines)
            output += ("\n" if output else "") + f"[stderr]\n{stderr_output}"
        if returncode != 0:
            output += f"\n[exit code: {returncode}]"
        if len(output) > 15_000:
            output = output[:6000] + f"\n\n... truncated ({len(output)} chars total) ...\n\n" + output[-3000:]
        return output.strip() or "(no output)"


def _check_dangerous(command: str) -> str | None:
    for pattern, reason in _DANGEROUS_PATTERNS:
        if re.search(pattern, command):
            return reason
    return None


def _update_cwd(command: str, current_cwd: str):
    global _cwd

    for part in command.split("&&"):
        part = part.strip()
        if not part.startswith("cd "):
            continue
        target = part[3:].strip().strip("'\"")
        if not target:
            continue
        new_dir = os.path.normpath(os.path.join(current_cwd, os.path.expanduser(target)))
        if os.path.isdir(new_dir):
            _cwd = new_dir


def _read_stream_lines(stream, source: str, line_queue: queue.Queue[tuple[str, str]]) -> None:
    if stream is None:
        return

    try:
        for raw_line in iter(stream.readline, ""):
            line_queue.put((source, raw_line.rstrip("\n")))
    finally:
        stream.close()


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1)
