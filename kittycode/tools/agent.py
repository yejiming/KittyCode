"""Sub-agent spawning."""

from .base import Tool


class AgentTool(Tool):
    name = "agent"
    description = """
    Spawn a sub-agent to handle a complex sub-task independently.
    The sub-agent has its own context and tool access. Use this for:
    researching a codebase, implementing a multi-step change in isolation,
    or any task that would benefit from a fresh context window.
    """
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What the sub-agent should accomplish",
            },
        },
        "required": ["task"],
    }

    _parent_agent = None

    def execute(self, task: str, cancel_event=None) -> str:
        if self._parent_agent is None:
            return "Error: agent tool not initialized (no parent agent)"

        from ..agent import Agent

        parent = self._parent_agent
        sub_agent = Agent(
            llm=parent.llm.clone() if hasattr(parent.llm, "clone") else parent.llm,
            tools=[type(tool)() for tool in parent.tools if tool.name != "agent"],
            max_context_tokens=parent.context.max_tokens,
            max_rounds=20,
        )
        sub_agent.ask_user_handler = getattr(parent, "ask_user_handler", None)
        sub_agent.on_brief_message = getattr(parent, "on_brief_message", None)

        try:
            result = sub_agent.chat(task, cancel_event=cancel_event)
            if len(result) > 5000:
                result = result[:4500] + "\n... (sub-agent output truncated)"
            return f"[Sub-agent completed]\n{result}"
        except Exception as exc:
            return f"Sub-agent error: {exc}"
