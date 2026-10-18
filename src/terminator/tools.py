from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """
        Get the weather at the given city
    """
    return f"It's always sunny in {city}!"
