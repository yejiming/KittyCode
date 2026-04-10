"""Tests for CLI presentation and interrupt helpers."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from rich.panel import Panel
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import FloatContainer
from prompt_toolkit.layout.menus import CompletionsMenu

from kittycode import __version__
from kittycode.cli import (
    _APP_STYLE,
    _MarkdownStreamRenderer,
    _PIXEL_CAT_ART,
    _PromptToolkitOutputFile,
    _build_cli_runtime,
    _brief,
    _build_input_reader,
    _build_history_line_metadata,
    _compose_footer_line,
    _create_agent,
    _format_tool_call_details,
    _format_tool_call,
    _filter_think_display_text,
    _format_model_choices,
    _format_question_prompt,
    _input_area_height_for_text,
    _last_line_start_offset,
    _merge_prompt_toolkit_styles,
    _merge_columns,
    _normalize_output_text,
    _parse_question_answer,
    _parse_args,
    _load_config,
    _render_brief_attachments,
    _render_markdown_to_plain_text,
    _render_tool_call_details,
    _render_startup_header,
    _repl,
    _resume_agent_session,
    _show_tool_call,
    _show_help,
    _style_startup_line,
    _startup_right_box_lines,
    _startup_left_box_lines,
    _style_history_markdown_line,
    _write_assistant_response,
    SlashCommandCompleter,
    render_message_to_text,
    main,
)
from kittycode.config import Config

def test_history_app_style_uses_low_saturation_role_palette():
    assert _APP_STYLE.style_rules == [
        ("history.system", "fg:#56b6c2"),
        ("history.user", "fg:#27cd96"),
        ("history.assistant", "fg:#d68786"),
        ("history.assistant.label", "bold"),
        ("history.tool", "fg:#56b6c2"),
        ("footer", "fg:#d68786"),
        ("footer.label", "fg:#d68786 bold"),
        ("history.markdown.heading1", "bold underline"),
        ("history.markdown.heading2", "bold"),
        ("history.markdown.heading3", "bold"),
        ("history.markdown.heading4", "bold"),
        ("history.markdown.heading5", "bold"),
        ("history.markdown.heading6", "bold"),
        ("history.markdown.list_marker", "fg:#95a88f"),
        ("history.markdown.quote", "fg:#93a2ab"),
        ("history.markdown.emphasis", "italic"),
        ("history.markdown.strong", "bold"),
        ("history.markdown.strike", "strike"),
        ("history.markdown.code", "fg:#b6a58e"),
        ("history.markdown.fence", "fg:#8d98a5"),
        ("history.markdown.codeblock", "fg:#9ea7b2"),
        ("input.rule", "fg:#6b7280"),
        ("startup.text", "fg:#6b7280"),
        ("startup.frame", "fg:#d68786"),
        ("startup.cat", "fg:#d68786"),
    ]


def test_merge_prompt_toolkit_styles_combines_classes_in_prompt_toolkit_format():
    merged = _merge_prompt_toolkit_styles(
        "class:history.assistant",
        "class:history.markdown.heading1",
    )

    assert merged == "class:history.assistant,history.markdown.heading1"


def test_merge_prompt_toolkit_styles_preserves_inline_styles():
    merged = _merge_prompt_toolkit_styles(
        "class:history.user",
        "bold",
        "fg:#ffffff",
    )

    assert merged == "class:history.user bold fg:#ffffff"


def test_style_startup_line_colors_frame_and_cat_but_keeps_text_gray():
    cat_fragments = _style_startup_line("│ /\\_/\\\\                     │  │ Interface: openai │", "class:startup.text")
    text_fragments = _style_startup_line("│          KittyCode v0.3.0          │", "class:startup.text")

    assert cat_fragments[0] == ("class:startup.text,startup.frame", "│")
    assert cat_fragments[2] == ("class:startup.text,startup.cat", "/")
    assert ("class:startup.text", "I") in cat_fragments
    assert ("class:startup.text,startup.frame", "│") in cat_fragments[-3:]
    assert any(fragment == ("class:startup.text", "K") for fragment in text_fragments)


def test_style_startup_line_keeps_non_cat_slashes_gray():
    fragments = _style_startup_line("│ Type /help for commands │", "class:startup.text")

    assert ("class:startup.text,startup.cat", "/") not in fragments
    assert ("class:startup.text", "/") in fragments


def test_pixel_cat_art_has_cat_ear_silhouette():
    lines = _PIXEL_CAT_ART.splitlines()

    assert lines[:3] == [
        " /\\_/\\\\",
        "( o.o )___________",
        " > ^           __)",
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


def test_render_startup_header_falls_back_to_single_left_box_before_right_box_gets_cramped():
    config = Config(
        interface="openai",
        model="gpt-4o",
        base_url="https://api.openai.com/v1",
    )

    console = Console(record=True, width=64)
    console.print(_render_startup_header(config, width=64))
    output = console.export_text()

    assert f"KittyCode v{__version__}" in output
    assert "Model: gpt-4o" in output
    assert "Interface: openai" not in output
    assert "Base: https://api.openai.com/v1" not in output


def test_render_startup_header_switches_to_compact_box_when_terminal_is_too_narrow():
    config = Config(
        interface="openai",
        model="gpt-4o",
        base_url="https://api.openai.com/v1",
    )

    console = Console(record=True, width=32)
    console.print(_render_startup_header(config, width=32))
    output = console.export_text()
    lines = output.splitlines()

    assert f"KittyCode v{__version__}" in output
    assert "Model: gpt-4o" in output
    assert " /\\_/\\\\" not in output
    assert all(len(line) <= 32 for line in lines if line)


def test_render_startup_header_uses_minimal_text_header_at_extreme_narrow_widths():
    config = Config(
        interface="openai",
        model="gpt-4o",
        base_url="https://api.openai.com/v1",
    )

    console = Console(record=True, width=12)
    console.print(_render_startup_header(config, width=12))
    output = console.export_text()
    lines = output.splitlines()

    assert "KittyCode" in output
    assert "╭" not in output
    assert "╯" not in output
    assert all(len(line) <= 12 for line in lines if line)


def test_startup_left_box_centers_cat_art():
    config = Config(model="gpt-4o")
    lines = _startup_left_box_lines(config)

    assert lines[0].startswith("╭")
    assert lines[-1].endswith("╯")
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
    assert "Type /help for commands, press Esc to interrupt a run, /quit to" in rendered
    assert "exit." in rendered


def test_startup_right_box_adapts_to_narrow_available_width():
    config = Config(interface="openai", model="gpt-4o", base_url="https://api.openai.com/v1")

    right_lines = _startup_right_box_lines(config, 28)

    assert max(len(line) for line in right_lines) <= 28
    assert any("Interface:" in line for line in right_lines)
    assert any("Base:" in line for line in right_lines)


def test_build_input_reader_keeps_history_path():
    reader = _build_input_reader("/tmp/kittycode-history", lambda: {"/help"})

    assert reader.history_path == "/tmp/kittycode-history"
    assert reader.application is not None
    assert reader.history_buffer is not None


def test_build_input_reader_uses_prompt_toolkit_session():
    reader = _build_input_reader("/tmp/kittycode-history", lambda: {"/help", "/quit"})

    assert reader.session.completer is not None
    assert reader.session.history is not None
    assert reader.layout is not None
    assert reader.input_buffer.history is reader.session.history


def test_build_input_reader_uses_rule_lines_and_bold_chevron_prompt():
    reader = _build_input_reader("/tmp/kittycode-history", lambda: {"/help", "/quit"})
    prompt_processor = reader.input_area.control.input_processors[-1]

    assert reader._input_top_rule_window is not None
    assert reader._input_bottom_rule_window is not None
    assert prompt_processor.text == [("bold", "> ")]


def test_history_render_width_prefers_history_window_render_width():
    reader = _build_input_reader("/tmp/kittycode-history", lambda: {"/help"})
    reader.history_window.render_info = SimpleNamespace(window_width=37)

    assert reader._history_render_width() == 37


def test_input_area_content_width_prefers_input_window_render_width():
    reader = _build_input_reader("/tmp/kittycode-history", lambda: {"/help"})
    reader.input_area.window.render_info = SimpleNamespace(window_width=29)

    assert reader._input_area_content_width() == 29


def test_build_input_reader_wraps_layout_in_float_container_with_completion_menu():
    reader = _build_input_reader("/tmp/kittycode-history", lambda: {"/help", "/quit"})

    assert isinstance(reader.layout.container, FloatContainer)
    assert len(reader.layout.container.floats) == 1
    completion_float = reader.layout.container.floats[0]
    assert isinstance(completion_float.content, CompletionsMenu)
    assert completion_float.attach_to_window is reader.input_area.window


def test_compose_footer_line_places_left_center_and_right_sections():
    line = _compose_footer_line(
        width=100,
        left="KittyCode v1.0.0",
        center="Author: Jimmy Ye",
        right="Read: 10  Write: 5",
    )

    assert len(line) == 100
    assert "KittyCode v1.0.0" in line
    assert "Author: Jimmy Ye" in line
    assert "Read: 10  Write: 5" in line


def test_compose_footer_line_prioritizes_wider_right_token_section():
    line = _compose_footer_line(
        width=100,
        left="KittyCode v1.0.0",
        center="Author: Jimmy Ye",
        right="Read: 123456789012345  Write: 987654321098765",
    )

    assert "Read: 123456789012345  Write: 987654321098765" in line


def test_reader_footer_shows_version_author_and_token_totals():
    reader = _build_input_reader(
        "/tmp/kittycode-history",
        lambda: {"/help", "/quit"},
        token_provider=lambda: (10, 5),
    )
    reader.application.output = SimpleNamespace(get_size=lambda: SimpleNamespace(columns=100))

    fragments = reader._render_footer_fragments()
    footer_text = "".join(text for _style, text in fragments)

    assert "KittyCode v" in footer_text
    assert "Author: Jimmy Ye" in footer_text
    assert "Read: 10  Write: 5" in footer_text
    assert "Total:" not in footer_text


def test_input_area_height_for_text_grows_with_multiline_content():
    height = _input_area_height_for_text(
        "first line\nsecond line\nthird line\nfourth line",
        content_width=40,
        prompt_width=6,
    )

    assert height == 4


def test_input_area_height_for_text_accounts_for_wrapped_long_line():
    height = _input_area_height_for_text(
        "x" * 90,
        content_width=24,
        prompt_width=6,
    )

    assert height == 4


def test_reader_updates_input_area_height_when_buffer_text_changes():
    reader = _build_input_reader("/tmp/kittycode-history", lambda: {"/help", "/quit"})
    reader.application.output = SimpleNamespace(get_size=lambda: SimpleNamespace(columns=30))

    reader.input_buffer.set_document(Document("x" * 80, cursor_position=80), bypass_readonly=True)

    assert reader._input_area_height == 4
    assert hasattr(reader.input_area.window.height, "preferred")
    assert reader.input_area.window.height.preferred == 4


def test_reader_uses_single_line_height_for_empty_input():
    reader = _build_input_reader("/tmp/kittycode-history", lambda: {"/help", "/quit"})

    assert reader._input_area_height == 1


def test_submit_current_input_validates_buffer():
    reader = _build_input_reader("/tmp/kittycode-history", lambda: {"/help", "/quit"})
    called = []
    reader.input_buffer.validate_and_handle = lambda: called.append("submit")

    reader._submit_current_input()

    assert called == ["submit"]


def test_insert_input_newline_adds_line_break():
    reader = _build_input_reader("/tmp/kittycode-history", lambda: {"/help", "/quit"})

    reader.input_buffer.set_document(Document("hello", cursor_position=5), bypass_readonly=True)
    reader._insert_input_newline()

    assert reader.input_buffer.text == "hello\n"


def test_slash_command_completer_returns_prefix_matches():
    completer = SlashCommandCompleter(lambda: {"/help", "/hello", "/quit"})

    completions = list(
        completer.get_completions(
            Document("/he", cursor_position=3),
            CompleteEvent(text_inserted=True),
        )
    )

    assert [completion.text for completion in completions] == ["/hello", "/help"]


def test_reader_up_down_navigates_completion_menu_when_open():
    reader = _build_input_reader("/tmp/kittycode-history", lambda: {"/help", "/quit"})
    actions: list[str] = []

    reader.input_buffer.complete_state = object()
    reader.input_buffer.complete_previous = lambda: actions.append("previous")
    reader.input_buffer.complete_next = lambda: actions.append("next")

    reader._navigate_completion_or_input_history(-1)
    reader._navigate_completion_or_input_history(1)

    assert actions == ["previous", "next"]


def test_reader_up_down_keybindings_are_eager_on_input_area():
    reader = _build_input_reader("/tmp/kittycode-history", lambda: {"/help", "/quit"})

    bindings = {
        binding.keys[0]: binding
        for binding in reader._build_key_bindings().bindings
        if binding.keys and binding.keys[0] in (Keys.Up, Keys.Down)
    }

    assert bindings[Keys.Up].eager()
    assert bindings[Keys.Down].eager()


def test_reader_up_down_uses_history_when_completion_menu_closed(monkeypatch):
    reader = _build_input_reader("/tmp/kittycode-history", lambda: {"/help", "/quit"})
    deltas: list[int] = []

    reader.input_buffer.complete_state = None
    monkeypatch.setattr(reader, "_navigate_input_history", lambda delta: deltas.append(delta))

    reader._navigate_completion_or_input_history(-1)
    reader._navigate_completion_or_input_history(1)

    assert deltas == [-1, 1]


def test_reader_write_raw_preserves_raw_text_in_history_buffer():
    reader = _build_input_reader("/tmp/kittycode-history", lambda: {"/help"})

    reader.write_raw("\x1b[31mhello\x1b[0m")

    assert reader._transient_output is not None
    assert reader._transient_output.role == "assistant"
    assert reader._transient_output.text == "\x1b[31mhello\x1b[0m"
    assert reader.history_buffer.text == "\x1b[31mhello\x1b[0m"


def test_last_line_start_offset_returns_start_of_final_line():
    assert _last_line_start_offset("") == 0
    assert _last_line_start_offset("hello") == 0
    assert _last_line_start_offset("one\ntwo") == 4


def test_build_history_line_metadata_marks_assistant_label_line():
    metadata = _build_history_line_metadata(
        SimpleNamespace(role="assistant", kind="plain", text="hello"),
        "hello",
    )

    assert metadata == [{"base_style": "class:history.assistant"}]


def test_reader_clear_history_resets_buffer_contents():
    reader = _build_input_reader("/tmp/kittycode-history", lambda: {"/help"})

    reader.write("assistant output")
    reader.write_raw("streaming tail")
    reader.clear_history()

    assert reader.history_buffer.text == ""
    assert reader._history_items == []
    assert reader._transient_output is None


def test_reader_commits_user_and_system_history_as_discrete_blocks():
    reader = _build_input_reader("/tmp/kittycode-history", lambda: {"/help"})

    reader._commit_prompt_input("/help")
    reader.print("[yellow]Conversation reset.[/yellow]")

    assert [(item.role, item.text) for item in reader._history_items] == [
        ("user", "/help"),
        ("system", "Conversation reset."),
    ]
    assert reader.history_buffer.text == "> /help\n\nConversation reset."
    assert reader._history_line_metadata == [
        {"base_style": "class:history.user"},
        {},
        {"base_style": "class:history.system"},
    ]


def test_reader_finalizes_transient_message_before_next_committed_entry():
    reader = _build_input_reader("/tmp/kittycode-history", lambda: {"/help"})

    reader.write_raw("partial")
    reader.write("final answer")

    assert [(item.role, item.text) for item in reader._history_items] == [
        ("assistant", "partial"),
        ("assistant", "final answer"),
    ]
    assert reader._transient_output is None
    assert reader.history_buffer.text == "partial\n\nfinal answer"
    assert reader.history_buffer.document.cursor_position == len("partial\n\n")


def test_reader_live_tool_output_stays_in_shared_transcript_while_active_and_after_finish():
    reader = _build_input_reader("/tmp/kittycode-history", lambda: {"/help"})

    reader.start_live_tool_output()
    reader.append_live_tool_output("one")
    reader.append_live_tool_output("two")

    assert reader._transient_output is not None
    assert reader._transient_output.role == "tool"
    assert reader._transient_output.text == "one\ntwo"
    assert reader.history_buffer.text == "Tool Output\none\ntwo"

    reader.finish_live_tool_output()

    assert [(item.role, item.text) for item in reader._history_items] == [
        ("tool", "one\ntwo"),
    ]
    assert reader._transient_output is None
    assert reader.history_buffer.text == "Tool Output\none\ntwo"


def test_render_message_to_text_renders_markdown_for_history():
    rendered = render_message_to_text(
        "assistant",
        "markdown",
        "# Title\n\n- item\n> quote\n`code`\n```py\nprint(1)\n```",
        width=80,
    )

    assert "KittyCode" not in rendered
    assert "Title" in rendered
    assert "• item" in rendered
    assert "print(1)" in rendered
    assert "# Title" not in rendered
    assert "```py" not in rendered


def test_reader_write_raw_markdown_renders_markdown_in_history_buffer():
    reader = _build_input_reader("/tmp/kittycode-history", lambda: {"/help"})

    reader.write_raw("# Title\n\n- item", kind="markdown")

    assert reader._transient_output is not None
    assert reader._transient_output.kind == "markdown"
    assert "Title" in reader.history_buffer.text
    assert "• item" in reader.history_buffer.text
    assert "# Title" not in reader.history_buffer.text


def test_render_message_to_text_prefixes_user_history_with_angle_marker():
    rendered = render_message_to_text("user", "plain", "hello", width=80)

    assert rendered == "> hello"


def test_build_history_line_metadata_maps_rendered_markdown_lines_to_styles():
    markdown = "# Title\n\n- item\n> quote\n```py\nprint(1)\n```"
    rendered = render_message_to_text("assistant", "markdown", markdown, width=80)

    metadata = _build_history_line_metadata(
        SimpleNamespace(role="assistant", kind="markdown", text=markdown),
        rendered,
    )

    markdown_kinds = [entry.get("markdown_kind") for entry in metadata if entry.get("markdown")]

    assert "heading" in markdown_kinds
    assert "list" in markdown_kinds
    assert "quote" in markdown_kinds
    assert "codeblock" in markdown_kinds


def test_build_history_line_metadata_marks_rendered_table_lines_as_table():
    markdown = "| Name | Value |\n| --- | --- |\n| Alpha | 1 |\n| B | 22 |"
    rendered = render_message_to_text("assistant", "markdown", markdown, width=80)

    metadata = _build_history_line_metadata(
        SimpleNamespace(role="assistant", kind="markdown", text=markdown),
        rendered,
    )

    table_kinds = [entry.get("markdown_kind") for entry in metadata if entry.get("markdown")]

    assert "table" in table_kinds


def test_build_history_line_metadata_keeps_raw_markdown_line_for_inline_styling():
    markdown = "给你一段**粗体字**。"
    rendered = render_message_to_text("assistant", "markdown", markdown, width=80)

    metadata = _build_history_line_metadata(
        SimpleNamespace(role="assistant", kind="markdown", text=markdown),
        rendered,
    )

    assert metadata[0]["raw_text"] == markdown


def test_history_line_metadata_stays_aligned_after_markdown_then_user_message():
    reader = _build_input_reader("/tmp/kittycode-history", lambda: {"/help"})

    reader.write_raw("# Title\n\n- item", kind="markdown")
    reader.finalize_active_output()
    reader._commit_prompt_input_ui("/help")

    assert reader.history_buffer.text.split("\n") == [
        "                                   Title                                    ",
        "",
        " • item                                                                     ",
        "",
        "> /help",
    ]
    assert reader._history_line_metadata == [
        {
            "base_style": "class:history.assistant",
            "markdown": True,
            "markdown_kind": "heading",
            "raw_text": "# Title",
        },
        {"base_style": "class:history.assistant"},
        {
            "base_style": "class:history.assistant",
            "markdown": True,
            "markdown_kind": "list",
            "raw_text": "- item",
        },
        {},
        {"base_style": "class:history.user"},
    ]


def test_style_history_markdown_line_highlights_heading():
    fragments = _style_history_markdown_line(
        "# Title",
        {"base_style": "class:history.assistant", "markdown": True, "markdown_kind": "heading", "raw_text": "# Title"},
    )

    assert fragments == [("class:history.assistant,history.markdown.heading1", "# Title")]


def test_style_history_markdown_line_uses_distinct_style_for_second_level_heading():
    fragments = _style_history_markdown_line(
        "Subtitle",
        {"base_style": "class:history.assistant", "markdown": True, "markdown_kind": "heading", "raw_text": "## Subtitle"},
    )

    assert fragments == [("class:history.assistant,history.markdown.heading2", "Subtitle")]


def test_style_history_markdown_line_highlights_list_marker_and_inline_code():
    fragments = _style_history_markdown_line(
        " • use rg",
        {
            "base_style": "class:history.assistant",
            "markdown": True,
            "markdown_kind": "list",
            "raw_text": "- use `rg`",
        },
    )

    assert fragments[0] == ("class:history.assistant,history.markdown.list_marker", " • ")
    assert fragments[1] == ("class:history.assistant", "use ")
    assert fragments[2] == ("class:history.assistant,history.markdown.code", "rg")


def test_style_history_markdown_line_highlights_strong_text_when_raw_markdown_is_available():
    fragments = _style_history_markdown_line(
        "给你一段粗体字。",
        {
            "base_style": "class:history.assistant",
            "markdown": True,
            "markdown_kind": "markdown",
            "raw_text": "给你一段**粗体字**。",
        },
    )

    assert fragments == [
        ("class:history.assistant", "给你一段"),
        ("class:history.assistant,history.markdown.strong", "粗体字"),
        ("class:history.assistant", "。"),
    ]


def test_style_history_markdown_line_highlights_emphasis_and_strike_when_raw_markdown_is_available():
    fragments = _style_history_markdown_line(
        "这是斜体和删除线",
        {
            "base_style": "class:history.assistant",
            "markdown": True,
            "markdown_kind": "markdown",
            "raw_text": "这是*斜体*和~~删除线~~",
        },
    )

    assert fragments == [
        ("class:history.assistant", "这是"),
        ("class:history.assistant,history.markdown.emphasis", "斜体"),
        ("class:history.assistant", "和"),
        ("class:history.assistant,history.markdown.strike", "删除线"),
    ]


def test_style_history_markdown_line_preserves_rendered_table_rows():
    fragments = _style_history_markdown_line(
        " Name  Value ",
        {
            "base_style": "class:history.assistant",
            "markdown": True,
            "markdown_kind": "markdown",
            "raw_text": "| Name | Value |",
        },
    )

    assert fragments == [("class:history.assistant", " Name  Value ")]


def test_style_history_markdown_line_preserves_rendered_table_borders():
    fragments = _style_history_markdown_line(
        "Alpha │ 1",
        {
            "base_style": "class:history.assistant",
            "markdown": True,
            "markdown_kind": "table",
        },
    )

    assert fragments == [("class:history.assistant", "Alpha │ 1")]


def test_style_history_markdown_line_highlights_inline_markdown_inside_table_cells():
    fragments = _style_history_markdown_line(
        "Alpha │ rg",
        {
            "base_style": "class:history.assistant",
            "markdown": True,
            "markdown_kind": "table",
            "raw_text": "| **Alpha** | `rg` |",
        },
    )

    assert fragments == [
        ("class:history.assistant,history.markdown.strong", "Alpha"),
        ("class:history.assistant", " "),
        ("class:history.assistant", "│"),
        ("class:history.assistant", " "),
        ("class:history.assistant,history.markdown.code", "rg"),
    ]


def test_style_history_markdown_line_highlights_quote_and_code_fence_blocks():
    quote = _style_history_markdown_line(
        "> quoted",
        {"base_style": "class:history.assistant", "markdown": True, "markdown_kind": "quote"},
    )
    fence = _style_history_markdown_line(
        "```py",
        {"base_style": "class:history.assistant", "markdown": True, "markdown_kind": "fence"},
    )
    codeblock = _style_history_markdown_line(
        "print(1)",
        {"base_style": "class:history.assistant", "markdown": True, "markdown_kind": "codeblock"},
    )

    assert quote == [("class:history.assistant,history.markdown.quote", "> quoted")]
    assert fence == [("class:history.assistant,history.markdown.fence", "```py")]
    assert codeblock == [("class:history.assistant,history.markdown.codeblock", "print(1)")]


def test_reader_prompt_persists_input_history(tmp_path):
    history_path = tmp_path / "kittycode-history"
    reader = _build_input_reader(str(history_path), lambda: {"/help"})
    reader.application.run = lambda: "/help"

    result = reader.prompt("You >")

    assert result == "/help"
    assert history_path.exists()
    assert "/help" in history_path.read_text()


def test_reader_navigation_uses_persisted_input_history(tmp_path):
    history_path = tmp_path / "kittycode-history"
    reader = _build_input_reader(str(history_path), lambda: {"/help"})
    reader.session.history.append_string("/help")
    reader.session.history.append_string("/quit")
    reader._load_input_history()

    reader._navigate_input_history(-1)
    assert reader.input_buffer.text == "/quit"

    reader._navigate_input_history(-1)
    assert reader.input_buffer.text == "/help"

    reader._navigate_input_history(1)
    assert reader.input_buffer.text == "/quit"


def test_reader_loads_file_history_before_first_input_navigation(tmp_path):
    history_path = tmp_path / "kittycode-history"
    from prompt_toolkit.history import FileHistory

    history = FileHistory(str(history_path))
    history.append_string("/help")
    history.append_string("/quit")

    reader = _build_input_reader(str(history_path), lambda: {"/help"})
    reader._load_input_history()
    reader._navigate_input_history(-1)

    assert reader.input_buffer.text == "/quit"


def test_create_agent_builds_llm_with_config_values(monkeypatch):
    captured = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured["llm_kwargs"] = kwargs

    class FakeAgent:
        def __init__(self, *, llm, max_context_tokens):
            captured["agent_llm"] = llm
            captured["max_context_tokens"] = max_context_tokens

    monkeypatch.setattr("kittycode.cli.LLM", FakeLLM)
    monkeypatch.setattr("kittycode.cli.Agent", FakeAgent)

    config = Config(
        interface="anthropic",
        model="claude-test",
        api_key="secret",
        base_url="https://example.test",
        temperature=0.25,
        max_tokens=2048,
        max_context_tokens=999,
    )

    agent = _create_agent(config)

    assert isinstance(agent, FakeAgent)
    assert captured["llm_kwargs"] == {
        "model": "claude-test",
        "api_key": "secret",
        "interface": "anthropic",
        "base_url": "https://example.test",
        "temperature": 0.25,
        "max_tokens": 2048,
    }
    assert captured["max_context_tokens"] == 999


def test_resume_agent_session_attaches_loaded_messages(monkeypatch):
    class FakeAgent:
        def __init__(self):
            self.messages = []

    monkeypatch.setattr("kittycode.cli.load_session", lambda session_id: ([{"role": "user", "content": "hi"}], "gpt-test"))

    agent = FakeAgent()
    resumed = _resume_agent_session(agent, "session-1")

    assert resumed is agent
    assert agent.messages == [{"role": "user", "content": "hi"}]


def test_build_cli_runtime_applies_overrides_and_resume(monkeypatch):
    base_config = Config(api_key="initial", model="base-model", interface="openai")
    base_config.models = [
        SimpleNamespace(
            interface="openai",
            provider="openai",
            api_key="initial",
            model_name="base-model",
            base_url="https://base.test/v1",
        )
    ]

    monkeypatch.setattr("kittycode.cli._load_config", lambda: base_config)

    fake_agent = SimpleNamespace(messages=[])
    monkeypatch.setattr("kittycode.cli._create_agent", lambda config: fake_agent)

    resumed = SimpleNamespace(messages=[])

    def fake_resume(agent, session_id):
        resumed.messages = [{"role": "assistant", "content": "restored"}]
        agent.messages = resumed.messages
        return agent

    monkeypatch.setattr("kittycode.cli._resume_agent_session", fake_resume)

    args = SimpleNamespace(
        model="override-model",
        interface="anthropic",
        base_url="https://override.test",
        api_key="override-key",
        prompt=None,
        resume="session-42",
    )

    config, agent = _build_cli_runtime(args)

    assert config is base_config
    assert config.model == "override-model"
    assert config.interface == "anthropic"
    assert config.base_url == "https://override.test"
    assert config.api_key == "override-key"
    assert len(config.models) == 1
    assert config.models[0].interface == "openai"
    assert config.models[0].model_name == "base-model"
    assert agent.messages == [{"role": "assistant", "content": "restored"}]


def test_load_config_exits_with_validation_guidance(monkeypatch):
    printed = []

    monkeypatch.setattr(
        "kittycode.cli.Config.from_file",
        lambda: (_ for _ in ()).throw(ValueError("models[0].model_name is required")),
    )
    monkeypatch.setattr("kittycode.cli.console.print", lambda message="", *args, **kwargs: printed.append(str(message)))

    with pytest.raises(SystemExit, match="1"):
        _load_config()

    assert any("Invalid config file" in line for line in printed)
    assert any("kittycode --config" in line for line in printed)
    assert any("models[0].model_name is required" in line for line in printed)


def test_parse_args_help_mentions_config_and_hides_runtime_override_flags(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["kittycode", "--help"])

    with pytest.raises(SystemExit, match="0"):
        _parse_args()

    captured = capsys.readouterr()
    help_text = captured.out

    assert "--config" in help_text
    assert "Open the guided configuration setup" in help_text
    assert "-m" not in help_text
    assert "--interface" not in help_text
    assert "--base-url" not in help_text
    assert "--api-key" not in help_text


def test_main_routes_prompt_to_run_once(monkeypatch):
    args = SimpleNamespace(prompt="hello", resume=None)
    config = Config(api_key="secret")
    agent = object()
    calls = []

    monkeypatch.setattr("kittycode.cli.configure_logging", lambda: calls.append("logging"))
    monkeypatch.setattr("kittycode.cli._parse_args", lambda: args)
    monkeypatch.setattr("kittycode.cli._build_cli_runtime", lambda parsed_args: (config, agent))
    monkeypatch.setattr("kittycode.cli._run_once", lambda runtime_agent, prompt: calls.append(("run_once", runtime_agent, prompt)))
    monkeypatch.setattr("kittycode.cli._repl", lambda runtime_agent, runtime_config: calls.append(("repl", runtime_agent, runtime_config)))

    main()

    assert calls == ["logging", ("run_once", agent, "hello")]


def test_main_routes_interactive_runtime_to_repl(monkeypatch):
    args = SimpleNamespace(prompt=None, resume=None)
    config = Config(api_key="secret")
    agent = object()
    calls = []

    monkeypatch.setattr("kittycode.cli.configure_logging", lambda: calls.append("logging"))
    monkeypatch.setattr("kittycode.cli._parse_args", lambda: args)
    monkeypatch.setattr("kittycode.cli._build_cli_runtime", lambda parsed_args: (config, agent))
    monkeypatch.setattr("kittycode.cli._run_once", lambda runtime_agent, prompt: calls.append(("run_once", runtime_agent, prompt)))
    monkeypatch.setattr("kittycode.cli._repl", lambda runtime_agent, runtime_config: calls.append(("repl", runtime_agent, runtime_config)))

    main()

    assert calls == ["logging", ("repl", agent, config)]


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


def test_show_help_lists_model_selector_without_trailing_name_contract():
    console = Console(record=True, width=80)

    _show_help(io=console)

    rendered = console.export_text()
    assert "/model         Switch model mid-conversation" in rendered
    assert "/model <name>" not in rendered


def test_format_model_choices_marks_active_entry():
    settings_module = __import__("kittycode.config.settings", fromlist=["StoredModelConfig"])
    config = Config(
        interface="openai",
        model="gpt-4o",
        api_key="sk-openai",
        base_url="https://api.openai.com/v1",
        models=[
            settings_module.StoredModelConfig(
                interface="openai",
                provider="OpenAI",
                api_key="sk-openai",
                model_name="gpt-4o",
                base_url="https://api.openai.com/v1",
            ),
            settings_module.StoredModelConfig(
                interface="anthropic",
                provider="Anthropic",
                api_key="sk-ant",
                model_name="claude-3-7-sonnet-latest",
                base_url="https://api.anthropic.com",
            ),
        ],
    )

    lines, notice = _format_model_choices(config)

    assert lines == [
        "Provider | Model",
        "1. * OpenAI | gpt-4o",
        "2.   Anthropic | claude-3-7-sonnet-latest",
    ]
    assert notice is None


def test_format_model_choices_reports_non_catalog_runtime_and_single_model_notice():
    settings_module = __import__("kittycode.config.settings", fromlist=["StoredModelConfig"])
    config = Config(
        interface="openai",
        model="override-model",
        api_key="override-key",
        base_url="https://override.example/v1",
        models=[
            settings_module.StoredModelConfig(
                interface="openai",
                provider="DeepSeek",
                api_key="sk-test",
                model_name="deepseek-chat",
                base_url="https://api.deepseek.com/v1",
            )
        ],
    )

    lines, notice = _format_model_choices(config)

    assert lines == [
        "Provider | Model",
        "1.   DeepSeek | deepseek-chat",
    ]
    assert "Current runtime is outside the configured model list." in notice
    assert "Only one configured model is available" in notice


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


def test_show_tool_call_prints_detail_panel_only():
    class FakeIO:
        def __init__(self):
            self.values = []

        def print(self, value):
            self.values.append(value)

    io = FakeIO()

    _show_tool_call(io, "grep", {"pattern": "needle", "path": "src"})

    assert len(io.values) == 1
    assert isinstance(io.values[0], Panel)


def test_write_assistant_response_renders_markdown_to_plain_text_and_appends_newline():
    writes = []

    _write_assistant_response(writes.append, "**hello**")

    assert len(writes) == 1
    assert "hello" in writes[0]
    assert "**hello**" not in writes[0]
    assert writes[0].endswith("\n")


def test_filter_think_display_text_hides_content_between_think_markers():
    assert _filter_think_display_text("hello<think>secret</think>world") == "helloworld"
    assert _filter_think_display_text("hello<think>secret<think>world") == "helloworld"
    assert _filter_think_display_text("<think>secret") == ""


def test_write_assistant_response_hides_think_content():
    writes = []

    _write_assistant_response(writes.append, "hello<think>secret</think>world")

    assert len(writes) == 1
    assert writes[0].strip() == "helloworld"
    assert writes[0].endswith("\n")


def test_render_markdown_to_plain_text_renders_markdown_content():
    rendered = _render_markdown_to_plain_text("# Title\n\n**bold**")

    assert "Title" in rendered
    assert "bold" in rendered


def test_render_markdown_to_plain_text_aligns_markdown_table_columns():
    rendered = _render_markdown_to_plain_text(
        "| Name | Value |\n| --- | --- |\n| Alpha | 1 |\n| B | 22 |",
        width=80,
    )
    lines = [line for line in rendered.splitlines() if line.strip()]

    assert "│" in lines[0]
    assert "┼" in lines[1]
    assert lines[2].index("│") == lines[3].index("│")


def test_render_markdown_to_plain_text_renders_inline_markdown_inside_table_cells():
    rendered = _render_markdown_to_plain_text(
        "| Name | Value |\n| --- | --- |\n| **Alpha** | `rg` |",
        width=80,
    )

    assert "**Alpha**" not in rendered
    assert "`rg`" not in rendered
    assert "Alpha" in rendered
    assert "rg" in rendered
    assert "# Title" not in rendered


def test_normalize_output_text_only_normalizes_newlines():
    rendered = "line 1\r\nline 2\rline 3"

    assert _normalize_output_text(rendered) == "line 1\nline 2line 3"


def test_markdown_stream_renderer_emits_incremental_rendered_text():
    emitted: list[str] = []
    clock = iter([0.0, 1.0])

    writer = _MarkdownStreamRenderer(
        emitted.append,
        render=lambda text: text.upper(),
        refresh_interval=0.0,
        now=lambda: next(clock),
        terminal_width=80,
    )

    writer.write("hello")
    assert emitted == ["HELLO"]

    writer.write(" world")
    assert emitted == ["HELLO", " WORLD"]

    writer.finish()


def test_markdown_stream_renderer_preserves_wrapped_text_without_rewind():
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
    assert emitted[0] == "123456789"

    writer.write("0")
    assert emitted == ["123456789", "0"]

    writer.finish()


def test_markdown_stream_renderer_no_duplicate_on_long_content():
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

    full_output = "".join(emitted)
    for i in range(60):
        count = full_output.count(f"Line {i}\n")
        assert count == 1, f"'Line {i}' appeared {count} times (expected 1)"


def test_markdown_stream_renderer_commits_active_reader_output_on_finish():
    reader = _build_input_reader("/tmp/kittycode-history", lambda: {"/help"})

    writer = _MarkdownStreamRenderer(
        reader.write_raw,
        render=lambda text: text,
        refresh_interval=0.0,
        now=lambda: 0.0,
        terminal_width=80,
        on_finish=reader.finalize_active_output,
    )

    writer.write("hello")

    assert reader._transient_output is not None
    assert reader._transient_output.role == "assistant"
    assert reader._history_items == []

    writer.finish()

    assert [(item.role, item.text) for item in reader._history_items] == [
        ("assistant", "hello"),
    ]
    assert reader._transient_output is None


def test_markdown_stream_renderer_on_text_keeps_reader_transcript_without_duplicates():
    reader = _build_input_reader("/tmp/kittycode-history", lambda: {"/help"})

    writer = _MarkdownStreamRenderer(
        lambda _text: None,
        refresh_interval=0.0,
        now=lambda: 0.0,
        terminal_width=80,
        on_finish=reader.finalize_active_output,
        on_text=lambda text: reader.write_raw(text, kind="markdown"),
    )

    writer.write("hello")
    writer.write(" world")

    assert reader._transient_output is not None
    assert reader._transient_output.role == "assistant"
    assert reader._transient_output.kind == "markdown"


def test_markdown_stream_renderer_hides_think_content_from_transcript():
    reader = _build_input_reader("/tmp/kittycode-history", lambda: {"/help"})

    writer = _MarkdownStreamRenderer(
        lambda _text: None,
        refresh_interval=0.0,
        now=lambda: 0.0,
        terminal_width=80,
        on_finish=reader.finalize_active_output,
        on_text=lambda text: reader.write_raw(text, kind="markdown"),
    )

    writer.write("hello<think>secret")
    writer.write("</think>world")

    assert reader._transient_output is not None
    assert reader._transient_output.text == "helloworld"

    writer.finish()

    assert [(item.role, item.kind, item.text) for item in reader._history_items] == [
        ("assistant", "markdown", "helloworld"),
    ]


def test_repl_command_controls_render_via_reader_print(monkeypatch):
    class FakeReader:
        def __init__(self, values):
            self.values = iter(values)
            self.printed = []
            self.cleared = 0

        def print(self, value):
            self.printed.append(value)

        def prompt(self, _message):
            return next(self.values)

        def clear_history(self):
            self.cleared += 1

    class FakeContext:
        def maybe_compress(self, messages, llm):
            return True

    settings_module = __import__("kittycode.config.settings", fromlist=["StoredModelConfig"])

    reconfigured = []
    writes = []

    agent = SimpleNamespace(
        skills=[],
        llm=SimpleNamespace(
            model="gpt-4o",
            total_prompt_tokens=10,
            total_completion_tokens=5,
            reconfigure=lambda **kwargs: reconfigured.append(kwargs),
        ),
        messages=[{"role": "user", "content": "hi"}],
        context=FakeContext(),
        reset=lambda: None,
    )
    config = Config(
        interface="openai",
        model="gpt-4o",
        api_key="sk-openai",
        base_url="https://api.openai.com/v1",
        models=[
            settings_module.StoredModelConfig(
                interface="openai",
                provider="OpenAI",
                api_key="sk-openai",
                model_name="gpt-4o",
                base_url="https://api.openai.com/v1",
            ),
            settings_module.StoredModelConfig(
                interface="anthropic",
                provider="Anthropic",
                api_key="sk-ant",
                model_name="claude-3-7-sonnet-latest",
                base_url="https://api.anthropic.com",
            ),
        ],
    )
    config.write = lambda: writes.append("saved")
    reader = FakeReader(["/tokens", "/model", "2", "/compact", "/reset", "/quit"])

    monkeypatch.setattr("kittycode.cli._build_input_reader", lambda *args, **kwargs: reader)
    monkeypatch.setattr("kittycode.runtime.context.estimate_tokens", lambda _messages: 42)

    _repl(agent, config)

    assert any("Tokens used this session:" in value for value in reader.printed)
    assert any("Provider | Model" in value for value in reader.printed)
    assert any("Switched to [cyan]Anthropic/claude-3-7-sonnet-latest[/cyan]" == value for value in reader.printed)
    assert any("[green]Compressed: 42 -> 42 tokens (1 messages)[/green]" == value for value in reader.printed)
    assert any("[yellow]Conversation reset.[/yellow]" == value for value in reader.printed)
    assert reconfigured == [
        {
            "model": "claude-3-7-sonnet-latest",
            "api_key": "sk-ant",
            "interface": "anthropic",
            "base_url": "https://api.anthropic.com",
        }
    ]
    assert writes == ["saved"]
    assert config.model == "claude-3-7-sonnet-latest"
    assert reader.cleared == 1


def test_repl_model_command_shows_single_model_message_without_switching(monkeypatch):
    class FakeReader:
        def __init__(self, values):
            self.values = iter(values)
            self.printed = []

        def print(self, value):
            self.printed.append(value)

        def prompt(self, _message):
            return next(self.values)

        def clear_history(self):
            pass

    settings_module = __import__("kittycode.config.settings", fromlist=["StoredModelConfig"])
    reconfigured = []
    agent = SimpleNamespace(
        skills=[],
        llm=SimpleNamespace(
            model="deepseek-chat",
            total_prompt_tokens=0,
            total_completion_tokens=0,
            reconfigure=lambda **kwargs: reconfigured.append(kwargs),
        ),
        messages=[],
        context=SimpleNamespace(maybe_compress=lambda messages, llm: False),
        reset=lambda: None,
    )
    config = Config(
        interface="openai",
        model="deepseek-chat",
        api_key="sk-test",
        base_url="https://api.deepseek.com/v1",
        models=[
            settings_module.StoredModelConfig(
                interface="openai",
                provider="DeepSeek",
                api_key="sk-test",
                model_name="deepseek-chat",
                base_url="https://api.deepseek.com/v1",
            )
        ],
    )
    reader = FakeReader(["/model", "/quit"])

    monkeypatch.setattr("kittycode.cli._build_input_reader", lambda *args, **kwargs: reader)

    _repl(agent, config)

    assert any("Provider | Model" in value for value in reader.printed)
    assert "[dim]Only one configured model is available, so switching is unavailable.[/dim]" in reader.printed
    assert reconfigured == []
