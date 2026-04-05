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

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }