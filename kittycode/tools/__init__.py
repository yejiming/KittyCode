"""Tool registry."""

from .agent import AgentTool
from .bash import BashTool
from .edit import EditFileTool
from .glob_tool import GlobTool
from .grep import GrepTool
from .read import ReadFileTool
from .write import WriteFileTool

_TOOL_TYPES = [
    BashTool,
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    GlobTool,
    GrepTool,
    AgentTool,
]

 
def create_tool_instances():
    return [tool_type() for tool_type in _TOOL_TYPES]


ALL_TOOLS = create_tool_instances()

def get_tool(name: str, tools=None):
    """Look up a tool by name."""
    for tool in tools or ALL_TOOLS:
        if tool.name == name:
            return tool
    return None
