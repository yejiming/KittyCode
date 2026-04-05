"""Tests for CLI presentation and interrupt helpers."""

import re

from rich.console import Console

from kittycode.cli import (
    _BOLD,
    _PIXEL_CAT_ART,
    _RESET,
    _build_input_reader,
    _merge_columns,
    _pixel_cat_banner,
    _render_startup_header,
    _show_help,
    _startup_right_box_lines,
    _startup_left_box_lines,
)
from kittycode.config import Config

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def test_pixel_cat_banner_contains_ansi_blocks():
    banner = _pixel_cat_banner()

    assert "\x1b[" in banner
    assert "/\\_/\\\\" in banner
    assert "KittyCode" in banner


def test_pixel_cat_art_has_cat_ear_silhouette():
    lines = _PIXEL_CAT_ART.splitlines()

    assert lines[:3] == [
        "\x1b[38;5;240m /\\_/\\\\\x1b[0m",
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

    assert "KittyCode v0.1.0" in output
    assert "Model: gpt-4o" in output
    assert "Interface: openai" not in output
    assert "Base: https://api.openai.com/v1" not in output
    assert "Type /help for commands" not in output


def test_startup_left_box_centers_cat_art():
    config = Config(model="gpt-4o")
    lines = _startup_left_box_lines(config)

    assert lines[0].startswith("\x1b[38;5;81m╭")
    assert lines[-1].endswith("╯\x1b[0m")
    assert " /\\_/\\\\" in lines[1]
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


def test_show_help_renders_without_name_error():
    _show_help()
