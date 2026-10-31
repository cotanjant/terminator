import uuid

from src.agent import build_graph
from .sandbox import Sandbox

def text_of(content) -> str:
    # some models return different outputs
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if b.get("type") == "text")


def show(m):
    if m.type == "ai":
        for tc in m.tool_calls:
            print(f"$ - TOOL USE {tc['name']}")
            print(f"$ - ARGS {tc['args']}")

        if text := text_of(m.content):
            print(text)
    elif m.type == "tool":
        print(m.content)


def main() -> None:
    sandbox = Sandbox()
    graph = build_graph(sandbox, "openrouter:openrouter/free")  # let's get LE CHONK
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": 40,
    }

    try:
        while True:
            user = input("> ").strip()
            if user in ("exit", "quit"):
                break

            for state in graph.stream(
                {"messages": [{"role": "user", "content": user}]},
                config,
                stream_mode="values",
            ):
                show(state["messages"][-1])
    finally:
        sandbox.close()
