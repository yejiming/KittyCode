"""Core agent loop.

This is the heart of KittyCode.

    user message -> LLM (with tools) -> tool calls? -> execute -> loop
                                      -> text reply? -> return to user

It keeps looping until the LLM responds with plain text, which means it is
done working and ready to report back.
"""

import concurrent.futures
import copy
import inspect

from .context import ContextManager
from .interrupts import CancellationRequested
from .llm import LLM
from .prompt import system_prompt, user_prompt
from .skills import load_skills
from .tools import create_tool_instances, get_tool
from .tools.agent import AgentTool
from .tools.base import Tool


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        max_context_tokens: int = 128_000,
        max_rounds: int = 50,
    ):
        self.llm = llm
        self.tools = list(tools) if tools is not None else create_tool_instances()
        self.messages: list[dict] = []
        self.todos: list[dict] = []
        self.brief_messages: list[dict] = []
        self.ask_user_handler = None
        self.on_brief_message = None
        self.context = ContextManager(max_tokens=max_context_tokens)
        self.max_rounds = max_rounds
        self.skills = []
        self._system = ""
        self.refresh_skills(force_reload=True)

        for tool in self.tools:
            tool.bind_agent(self)

    def _full_messages(self) -> list[dict]:
        return [{"role": "system", "content": self._system}] + self.messages

    def _build_user_message(self, user_input: str) -> str:
        self.refresh_skills()
        return user_prompt(user_input, self.skills, self.todos)

    def _tool_schemas(self) -> list[dict]:
        return [tool.schema() for tool in self.tools]

    def fork(self):
        worker = Agent(
            llm=self.llm.clone() if hasattr(self.llm, "clone") else self.llm,
            tools=[type(tool)() for tool in self.tools],
            max_context_tokens=self.context.max_tokens,
            max_rounds=self.max_rounds,
        )
        worker.messages = copy.deepcopy(self.messages)
        worker.todos = copy.deepcopy(self.todos)
        worker.brief_messages = copy.deepcopy(self.brief_messages)
        worker.ask_user_handler = self.ask_user_handler
        worker.on_brief_message = self.on_brief_message
        worker.skills = list(self.skills)
        worker._system = self._system
        return worker

    def chat(self, user_input: str, on_token=None, on_tool=None, on_tool_output=None, cancel_event=None) -> str:
        """Process one user message. May involve multiple LLM/tool rounds."""
        self.messages.append({"role": "user", "content": self._build_user_message(user_input)})
        self.context.maybe_compress(self.messages, self.llm)

        try:
            self._raise_if_cancelled(cancel_event)

            for _ in range(self.max_rounds):
                self._raise_if_cancelled(cancel_event)
                response = self.llm.chat(
                    messages=self._full_messages(),
                    tools=self._tool_schemas(),
                    on_token=on_token,
                    cancel_event=cancel_event,
                )

                if not response.tool_calls:
                    self.messages.append(response.message)
                    return response.content

                self.messages.append(response.message)

                if len(response.tool_calls) == 1:
                    tool_call = response.tool_calls[0]
                    if on_tool:
                        on_tool(tool_call.name, tool_call.arguments)
                    self._raise_if_cancelled(cancel_event)
                    result = self._exec_tool(tool_call, cancel_event=cancel_event, on_output=on_tool_output)
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )
                else:
                    results = self._exec_tools_parallel(
                        response.tool_calls,
                        on_tool=on_tool,
                        on_tool_output=on_tool_output,
                        cancel_event=cancel_event,
                    )
                    for tool_call, result in zip(response.tool_calls, results):
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result,
                            }
                        )

                self.context.maybe_compress(self.messages, self.llm)

            return "(reached maximum tool-call rounds)"
        except CancellationRequested:
            return "(interrupted)"

    def _exec_tool(self, tool_call, cancel_event=None, on_output=None) -> str:
        """Execute a single tool call and return the result string."""
        self._raise_if_cancelled(cancel_event)
        tool = get_tool(tool_call.name, self.tools)
        if tool is None:
            return f"Error: unknown tool '{tool_call.name}'"
        try:
            execute_kwargs = dict(tool_call.arguments)
            parameters = inspect.signature(tool.execute).parameters
            if on_output is not None and "stream_callback" in parameters:
                execute_kwargs["stream_callback"] = lambda text: on_output(tool_call.name, text)
            if cancel_event is not None and "cancel_event" in parameters:
                execute_kwargs["cancel_event"] = cancel_event
            return tool.execute(**execute_kwargs)
        except TypeError as exc:
            return f"Error: bad arguments for {tool_call.name}: {exc}"
        except Exception as exc:
            return f"Error executing {tool_call.name}: {exc}"

    def _exec_tools_parallel(self, tool_calls, on_tool=None, on_tool_output=None, cancel_event=None) -> list[str]:
        """Run multiple tool calls concurrently using threads."""
        for tool_call in tool_calls:
            if on_tool:
                on_tool(tool_call.name, tool_call.arguments)
        self._raise_if_cancelled(cancel_event)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(self._exec_tool, tool_call, cancel_event, on_tool_output) for tool_call in tool_calls]
            return [future.result() for future in futures]

    def reset(self):
        """Clear conversation history."""
        self.messages.clear()

    def refresh_skills(self, force_reload: bool = False):
        """Refresh cached skill metadata and rebuild the system prompt."""
        self.skills = load_skills(force_reload=force_reload)
        self._system = system_prompt(self.tools, self.skills)

    @staticmethod
    def _raise_if_cancelled(cancel_event):
        if cancel_event is not None and cancel_event.is_set():
            raise CancellationRequested()
