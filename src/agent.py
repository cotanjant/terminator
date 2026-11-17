from pathlib import Path
from typing import Annotated
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from src.sandbox import Sandbox


class State(TypedDict):
    messages: Annotated[list, add_messages]


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompt.md"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")
DEFAULT_MODEL = "google_genai:gemini-3.5-flash-lite"


def text_of(content) -> str:
    """Extract plain text from string or block-list message content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(
        b.get("text", "") if isinstance(b, dict) and b.get("type") == "text" else (b if isinstance(b, str) else "")
        for b in content
    )


def build_graph(sandbox: Sandbox, model_name: str):
    """Build and compile the LangGraph agent wired to the given Sandbox."""

    @tool
    def run_command(cmd: str) -> str:
        """Runs a command in sandbox and returns output."""
        return sandbox.execute(cmd)

    tools = [run_command]
    llm = init_chat_model(model_name).bind_tools(tools)

    def agent(state: State):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
        return {"messages": [llm.invoke(msgs)]}

    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")

    return g.compile(checkpointer=MemorySaver())
