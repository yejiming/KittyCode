"""Tests for CLI presentation and interrupt helpers."""

import re

from rich.console import Console
from rich.panel import Panel

from kittycode import __version__
from kittycode.cli import (
    _BOLD,
    _MarkdownStreamRenderer,
    _PIXEL_CAT_ART,
    _PromptToolkitOutputFile,
    _RESET,
    _brief,
    _build_input_reader,
    _format_tool_call_details,
    _format_tool_call,
    _format_question_prompt,
    _merge_columns,
    _parse_question_answer,
    _render_brief_attachments,
    _render_markdown_as_ansi,
    _rendered_line_count,
    _render_tool_call_details,
    _render_startup_header,
    _rewind_and_clear_lines,
    _show_tool_call,
    _show_help,
    _startup_right_box_lines,
    _startup_left_box_lines,
    _write_assistant_response,
)
from kittycode.config import Config

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def test_pixel_cat_art_has_cat_ear_silhouette():
    lines = _PIXEL_CAT_ART.splitlines()

    assert lines[:3] == [
        "\x1b[38;5;240m /\\_/\\\x1b[0m",
        "\x1b[38;5;240m(\x1b[38;5;215m \x1b[38;5;255mo.o\x1b[38;5;215m \x1b[38;5;240m)\x1b[38;5;215m___________\x1b[0m",
        "\x1b[38;5;240m \x1b[38;5;215m>\x1b[38;5;240m \x1b[38;5;218m^\x1b[38;5;240m           __)\x1b[0m",
    ]


def test_render_startup_header_shows_two_columns_at_standard_terminal_width():
    config = Config(
        interface="anthropic",
        model="claude-3-7-sonnet-latest",
        base_url="https://api.anthropic.com",
    )

    console = Console(record=True, width=80)
    console.print(_render_startup_header(config, width=80))
    output = console.export_text()
    lines = output.splitlines()

    assert "KittyCode" in output
    assert "Model: claude-3-7-sonnet-latest" in output
    assert "Interface: anthropic" in output
    assert "Base:" in output
    assert "https://api.anthropic.com" in output
    assert any("╭" in line for line in lines[:2])
    assert any("│" in line and "Interface: anthropic" in line for line in lines[:4])


def test_render_startup_header_hides_right_column_when_narrow():
    config = Config(
        interface="openai",
        model="gpt-4o",
        base_url="https://api.openai.com/v1",
    )

    console = Console(record=True, width=40)
    console.print(_render_startup_header(config, width=40))
    output = console.export_text()

    assert f"KittyCode v{__version__}" in output
    assert "Model: gpt-4o" in output
    assert "Interface: openai" not in output
    assert "Base: https://api.openai.com/v1" not in output
    assert "Type /help for commands" not in output


def test_startup_left_box_centers_cat_art():
    config = Config(model="gpt-4o")
    lines = _startup_left_box_lines(config)

    assert lines[0].startswith("\x1b[38;5;81m╭")
    assert lines[-1].endswith("╯\x1b[0m")
    assert " /\\_/\\" in lines[1]
    assert lines[1].count(" ") > 4


def test_startup_boxes_share_same_height_and_right_content_is_centered():
    config = Config(interface="openai", model="gpt-4o", base_url="https://api.openai.com/v1")
    left_lines = _startup_left_box_lines(config)
    right_lines = _startup_right_box_lines(config, 60, target_height=len(left_lines))
    merged = _merge_columns(
        left_lines,
        right_lines,
        max(len(line) for line in left_lines),
        2,
    )

    assert len(left_lines) == len(right_lines)
    first_right_content = next(i for i, line in enumerate(right_lines) if "Interface:" in line)
    last_right_content = max(i for i, line in enumerate(right_lines) if "exit." in line)
    assert first_right_content > 1
    assert last_right_content < len(right_lines) - 2
    assert len(merged) == len(left_lines)


def test_startup_right_box_only_bolds_shortcuts_in_help_hint():
    config = Config(interface="openai", model="gpt-4o", base_url="https://api.openai.com/v1")

    right_lines = _startup_right_box_lines(config, 80)
    rendered = "\n".join(right_lines)
    plain_text = ANSI_RE.sub("", rendered)

    assert "Type /help for commands, press Esc to interrupt a run, /quit to exit." in plain_text
    assert f"Type {_BOLD}/help{_RESET} for commands, press {_BOLD}Esc{_RESET} to interrupt a run, {_BOLD}/quit{_RESET} to exit." in rendered
    assert f"{_BOLD}Type /help for commands, press Esc to interrupt a run, /quit to exit.{_RESET}" not in rendered


def test_build_input_reader_keeps_history_path():
    reader = _build_input_reader("/tmp/kittycode-history", lambda: {"/help"})

    assert reader.history_path == "/tmp/kittycode-history"


def test_build_input_reader_uses_prompt_toolkit_session():
    reader = _build_input_reader("/tmp/kittycode-history", lambda: {"/help", "/quit"})

    assert reader.session.completer is not None
    assert reader.session.history is not None


def test_prompt_toolkit_output_file_forwards_raw_writes_and_flushes():
    events = []

    class FakeOutput:
        def write_raw(self, text):
            events.append(("write_raw", text))

        def flush(self):
            events.append(("flush",))

    wrapper = _PromptToolkitOutputFile(FakeOutput())

    assert wrapper.write("abc") == 3
    wrapper.flush()
    assert wrapper.isatty() is True
    assert events == [("write_raw", "abc"), ("flush",)]


def test_show_help_renders_without_name_error():
    _show_help()


def test_format_question_prompt_lists_options_and_free_text_hint():
    prompt = _format_question_prompt(
        {
            "header": "Library",
            "question": "Which library should we use?",
            "options": [
                {"label": "requests", "description": "HTTP client", "recommended": True},
                {"label": "urllib", "description": "stdlib", "recommended": False},
            ],
            "multiSelect": False,
            "allowFreeformInput": True,
        }
    )

    assert "1. requests (recommended) - HTTP client" in prompt
    assert "2. urllib - stdlib" in prompt
    assert "You can also enter free text." in prompt


def test_parse_question_answer_supports_single_and_multi_select():
    single = {
        "options": [
            {"label": "A"},
            {"label": "B"},
        ],
        "multiSelect": False,
        "allowFreeformInput": False,
    }
    multi = {
        "options": [
            {"label": "A"},
            {"label": "B"},
            {"label": "C"},
        ],
        "multiSelect": True,
        "allowFreeformInput": False,
    }

    assert _parse_question_answer("2", single) == "B"
    assert _parse_question_answer("1,3", multi) == "A, C"


def test_render_brief_attachments_formats_attachment_lines():
    lines = _render_brief_attachments(
        {
            "attachments": [
                {"path": "/tmp/file.txt", "size": 42, "is_image": False},
            ]
        }
    )

    assert lines == ["[dim]Attachment:[/dim] /tmp/file.txt (42 bytes, image=False)"]


def test_format_tool_call_shows_full_todo_write_payload():
    rendered = _format_tool_call(
        "todo_write",
        {
            "todos": [
                {
                    "content": "Inspect runtime behavior for todo rendering",
                    "active_form": "Inspecting runtime behavior for todo rendering",
                    "status": "in_progress",
                },
                {
                    "content": "Verify that every todo item is visible in the CLI output",
                    "active_form": "Verifying that every todo item is visible in the CLI output",
                    "status": "pending",
                },
            ]
        },
    )

    assert rendered.startswith("todo_write(")
    assert "Verify that every todo item is visible in the CLI output" in rendered
    assert "..." not in rendered


def test_format_tool_call_details_keeps_full_payload():
    long_value = "retain-full-payload-" * 30

    rendered = _format_tool_call_details(
        "web_fetch",
        {
            "url": "https://example.com",
            "prompt": long_value,
        },
    )

    assert rendered[0] == ("tool", "web_fetch")
    assert ("url", "https://example.com") in rendered
    assert ("prompt", long_value) in rendered


def test_render_tool_call_details_returns_panel_with_full_arguments():
    panel = _render_tool_call_details(
        "write_file",
        {
            "file_path": "/tmp/demo.txt",
            "content": "full body",
        },
    )
    console = Console(record=True, width=200)

    console.print(panel)
    output = console.export_text()

    assert isinstance(panel, Panel)
    assert "Tool Call: write_file" in output
    assert "Tool" in output
    assert "file_path" in output
    assert "/tmp/demo.txt" in output
    assert "content" in output
    assert "full body" in output
    assert '"file_path"' not in output


def test_show_tool_call_prints_summary_then_detail_panel():
    class FakeIO:
        def __init__(self):
            self.values = []

        def print(self, value):
            self.values.append(value)

    io = FakeIO()

    _show_tool_call(io, "grep", {"pattern": "needle", "path": "src"})

    assert io.values[0].startswith("\n[dim]> grep(")
    assert isinstance(io.values[1], Panel)


def test_write_assistant_response_preserves_ansi_and_appends_newline():
    writes = []

    _write_assistant_response(writes.append, "**hello**")

    assert len(writes) == 1
    assert "hello" in ANSI_RE.sub("", writes[0])
    assert writes[0].endswith("\n")


def test_render_markdown_as_ansi_renders_markdown_content():
    rendered = _render_markdown_as_ansi("# Title\n\n**bold**")

    plain = ANSI_RE.sub("", rendered)

    assert "Title" in plain
    assert "bold" in plain
    assert rendered.endswith("\n")


def test_rendered_line_count_accounts_for_terminal_wrapping_and_ansi_width():
    rendered = "\x1b[1m1234567890\x1b[0m\n"

    assert _rendered_line_count(rendered, width=5) == 2


def test_rewind_and_clear_lines_returns_terminal_control_sequence():
    sequence = _rewind_and_clear_lines(2)

    assert sequence.startswith("\x1b[2A")
    assert "\r\x1b[2K" in sequence
    assert sequence.endswith("\r")


def test_markdown_stream_renderer_rewrites_rendered_block():
    """Incremental renderer emits changed tail only on updates."""
    emitted: list[str] = []
    clock = iter([0.0, 1.0])

    writer = _MarkdownStreamRenderer(
        emitted.append,
        render=lambda text: f"<{text}>",
        refresh_interval=0.0,
        now=lambda: next(clock),
        terminal_width=80,
    )

    writer.write("hello")
    # First render: full emit
    assert len(emitted) == 1
    assert "<hello>" in emitted[0]

    writer.write(" world")
    # Second render: rewinds old tail and emits new
    assert len(emitted) > 1

    writer.finish()


def test_markdown_stream_renderer_rewinds_all_wrapped_rows():
    """When a logical line wraps, the rewind accounts for all physical rows."""
    emitted: list[str] = []
    clock = iter([0.0, 1.0])

    writer = _MarkdownStreamRenderer(
        emitted.append,
        render=lambda text: text,
        refresh_interval=0.0,
        now=lambda: next(clock),
        terminal_width=5,
    )

    writer.write("123456789")
    first_emit = emitted[0]
    assert "123456789" in first_emit

    writer.write("0")
    # Rewind should cover 2 physical rows (9 chars at width 5 = 2 rows)
    # and new content "1234567890" with 2 physical rows is then emitted.
    assert len(emitted) > 1
    combined = "".join(emitted)
    assert "1234567890" in combined

    writer.finish()


def test_markdown_stream_renderer_no_duplicate_on_long_content():
    """Long content that exceeds terminal height must not produce duplicates."""
    emitted: list[str] = []
    tick = 0.0

    def clock():
        nonlocal tick
        tick += 1.0
        return tick

    writer = _MarkdownStreamRenderer(
        emitted.append,
        render=lambda text: text,
        refresh_interval=0.0,
        now=clock,
        terminal_width=40,
    )

    # Build up content that is much taller than a typical terminal (>50 lines)
    for i in range(60):
        writer.write(f"Line {i}\n")

    writer.finish()

    # Collect all non-control-sequence text that was emitted
    full_output = "".join(emitted)
    for i in range(60):
        count = full_output.count(f"Line {i}\n")
        assert count == 1, f"'Line {i}' appeared {count} times (expected 1)"
