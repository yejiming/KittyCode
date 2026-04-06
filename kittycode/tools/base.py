"""Base class for all tools."""

from abc import ABC, abstractmethod


class Tool(ABC):
    """Minimal tool interface."""

    name: str
    description: str
    parameters: dict

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Run the tool and return a text result."""

    def bind_agent(self, agent) -> None:
        if hasattr(self, "_parent_agent"):
            self._parent_agent = agent

    def schema(self) -> dict:
        description = self.description.strip()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": description,
                "parameters": self.parameters,
            },
        }