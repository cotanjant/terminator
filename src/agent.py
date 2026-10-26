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

SYSTEM_PROMPT = open("prompt.md", encoding="utf-8").read()

def build_graph(sandbox: Sandbox, model_name: str):
    @tool
    def run_command(cmd: str) -> str:
        """
            Runs a command in sandbox and returns output
        """

        return sandbox.execute(cmd)

    tools = [run_command]
    llm = init_chat_model(model_name).bind_tools(tools)

    def agent(state: State):
        msgs = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ] + state["messages"]

        return {
            "messages": [llm.invoke(msgs)]
        }

    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "agent")
    # if there is tool_call in the last message redirect to "tools" if not END
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")

    return g.compile(checkpointer=MemorySaver())
