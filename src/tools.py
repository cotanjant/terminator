from langchain.tools import tool

# we could add tavily for web search
@tool
def get_weather(city: str) -> str:
    """
        Get the weather at the given city
    """
    return f"It's always sunny in {city}!"

@tool
def run_command(args: str) -> str:
    """
        runs command on container
    """
