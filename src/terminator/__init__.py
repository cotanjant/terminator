from os import read
from terminator.tools import get_weather
from langchain.agents import create_agent

with open("prompt.md", "r", encoding="utf-8") as file:
    content = file.read()

def main() -> None:
    agent = create_agent(
        model="openrouter:openrouter/free",
        tools=[
            get_weather
        ],
        system_prompt=content,
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather in izmir"
                }
            ]
        }
    )

    print(result["messages"][-1].content_blocks) # this content_blocks is the way of us to read json as blocks
