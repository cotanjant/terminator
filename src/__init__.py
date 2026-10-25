from os import read
from langchain.agents import create_agent
from .tools import execute

with open("prompt.md", "r", encoding="utf-8") as file:
    content = file.read()

def main() -> None:
    agent = create_agent(
        model="openrouter:openrouter/free",
        tools=[
            execute
        ],
        system_prompt=content,
    )


    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Create a directory with mkdir and show me the output"
                }
            ]
        }
    )

    print(result["messages"][-1].content_blocks) # this content_blocks is the way of us to read json as blocks
