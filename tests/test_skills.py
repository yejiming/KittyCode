"""Tests for local skill discovery and prompt injection."""

from types import SimpleNamespace

from prompt_toolkit.document import Document

from kittycode.cli import (
    SlashCommandCompleter,
    _build_skill_request,
    _format_skills,
    _match_skill_command,
    _resolve_command_prefix,
    _slash_command_names,
)
from pathlib import Path

from kittycode.agent import Agent
from kittycode.prompt import system_prompt, user_prompt
from kittycode.skills import SkillDefinition, load_skills
from kittycode.tools import ALL_TOOLS


class DummyLLM:
    def chat(self, *args, **kwargs):
        raise AssertionError("DummyLLM.chat should not be called in these tests")


class RecordingLLM:
    def __init__(self):
        self.captured_messages = []

    def chat(self, messages, *args, **kwargs):
        self.captured_messages.append(messages)
        return SimpleNamespace(
            tool_calls=[],
            message={"role": "assistant", "content": "done"},
            content="done",
        )


def test_load_skills_reads_front_matter(tmp_path):
    skill_dir = tmp_path / "example-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: Example Skill\n"
        "description: Example description for testing.\n"
        "---\n"
        "\n"
        "# Body\n"
    )

    skills = load_skills(tmp_path, force_reload=True)

    assert len(skills) == 1
    assert skills[0].name == "Example Skill"
    assert skills[0].description == "Example description for testing."
    assert skills[0].path == str(skill_dir.resolve())


def test_load_skills_falls_back_when_header_missing(tmp_path):
    skill_dir = tmp_path / "fallback-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Fallback Skill\n\nFallback description.\n")

    skills = load_skills(tmp_path, force_reload=True)

    assert skills[0].name == "Fallback Skill"
    assert skills[0].description == "Fallback description."


def test_load_skills_auto_reloads_after_change(tmp_path):
    skill_dir = tmp_path / "reload-skill"
    skill_dir.mkdir()
    skill_doc = skill_dir / "SKILL.md"
    skill_doc.write_text("name: Reload Skill\ndescription: Old description\n")

    first = load_skills(tmp_path, force_reload=True)
    assert first[0].description == "Old description"

    skill_doc.write_text("name: Reload Skill\ndescription: New description\n")

    second = load_skills(tmp_path)
    assert second[0].description == "New description"


def test_system_prompt_mentions_reminder_tags_instead_of_embedding_skills():
    prompt = system_prompt(
        ALL_TOOLS[:1],
        [SkillDefinition(name="Skill A", description="Does A", path="/tmp/skill-a")],
    )

    assert "<system-reminder>" in prompt
    assert "<todo-reminder>" in prompt
    assert "available skill blocks" in prompt
    assert "name: Skill A" not in prompt


def test_user_prompt_appends_skill_and_todo_reminders():
    prompt = user_prompt(
        "Inspect the repository",
        [SkillDefinition(name="Skill A", description="Does A", path="/tmp/skill-a")],
        [
            {
                "content": "Run tests",
                "active_form": "Running tests",
                "status": "in_progress",
            }
        ],
    )

    assert prompt.startswith("Inspect the repository")
    assert "<system-reminder>" in prompt
    assert "name: Skill A" in prompt
    assert "description: Does A" in prompt
    assert "path: /tmp/skill-a" in prompt
    assert "<todo-reminder>" in prompt
    assert "[in_progress] Run tests" in prompt
    assert "active_form: Running tests" in prompt


def test_agent_appends_skill_and_todo_reminders_to_chat_messages(monkeypatch):
    loaded_skill = SkillDefinition(name="Skill B", description="Does B", path="/tmp/skill-b")
    monkeypatch.setattr(
        "kittycode.agent.load_skills",
        lambda force_reload=False: [loaded_skill],
    )

    llm = RecordingLLM()
    agent = Agent(llm=llm)
    agent.todos = [
        {
            "content": "Inspect runtime",
            "active_form": "Inspecting runtime",
            "status": "in_progress",
        }
    ]

    reply = agent.chat("hello")

    assert agent.skills == [loaded_skill]
    assert reply == "done"
    assert llm.captured_messages
    user_message = llm.captured_messages[0][1]
    assert user_message["role"] == "user"
    assert user_message["content"].startswith("hello")
    assert "name: Skill B" in user_message["content"]
    assert "<todo-reminder>" in user_message["content"]
    assert "Inspect runtime" in user_message["content"]


def test_agent_refreshes_skills_each_user_turn(tmp_path, monkeypatch):
    skill_dir = tmp_path / "live-skill"
    skill_dir.mkdir()
    skill_doc = skill_dir / "SKILL.md"
    skill_doc.write_text("name: Live Skill\ndescription: First version\n")

    monkeypatch.setattr(
        "kittycode.agent.load_skills",
        lambda force_reload=False: load_skills(tmp_path, force_reload=force_reload),
    )

    llm = RecordingLLM()
    agent = Agent(llm=llm)
    agent.chat("first")
    assert "First version" in llm.captured_messages[0][1]["content"]

    skill_doc.write_text("name: Live Skill\ndescription: Updated version\n")

    agent.chat("second")
    assert "Updated version" in llm.captured_messages[1][-1]["content"]


def test_format_skills_output():
    text = _format_skills([
        SkillDefinition(name="Skill C", description="Does C", path="/tmp/skill-c"),
    ])

    assert "1. Skill C" in text
    assert "/Skill C" in text
    assert "Does C" in text
    assert "/tmp/skill-c" in text


def test_format_skills_empty():
    assert _format_skills([]) == "No skills loaded from ~/.kittycode/skills"


def test_match_skill_command_without_task():
    skill = SkillDefinition(name="Skill D", description="Does D", path="/tmp/skill-d")

    matched = _match_skill_command("/Skill D", [skill])

    assert matched == (skill, "")


def test_match_skill_command_with_task():
    skill = SkillDefinition(name="Skill E", description="Does E", path="/tmp/skill-e")

    matched = _match_skill_command("/Skill E refactor this file", [skill])

    assert matched == (skill, "refactor this file")


def test_build_skill_request():
    skill = SkillDefinition(name="Skill F", description="Does F", path="/tmp/skill-f")

    request = _build_skill_request(skill, "inspect the project")

    assert 'Use the local skill "Skill F"' in request
    assert "Skill path: /tmp/skill-f" in request
    assert request.endswith("inspect the project")


def test_slash_command_names_include_builtins_and_skills():
    skill = SkillDefinition(name="Skill G", description="Does G", path="/tmp/skill-g")

    names = _slash_command_names([skill])

    assert "/help" in names
    assert "/quit" in names
    assert "/Skill G" in names


def test_resolve_command_prefix_unique_match():
    resolved, matches = _resolve_command_prefix("/q", [])

    assert resolved == "/quit"
    assert matches == ["/quit"]


def test_resolve_command_prefix_ambiguous_match():
    resolved, matches = _resolve_command_prefix("/s", [])

    assert resolved is None
    assert "/save" in matches
    assert "/sessions" in matches
    assert "/skills" in matches


def test_slash_command_completer_prefix_matches_skills_and_commands():
    skill = SkillDefinition(name="Skill H", description="Does H", path="/tmp/skill-h")
    completer = SlashCommandCompleter(lambda: _slash_command_names([skill]))

    completions = list(completer.get_completions(Document(text="/sk", cursor_position=3), None))
    completion_texts = [completion.text for completion in completions]

    assert "/skills" in completion_texts
    assert "/Skill H" in completion_texts