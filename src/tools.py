import atexit

from langchain.tools import tool
from .sandbox import Sandbox

sandbox = Sandbox()
atexit.register(sandbox.close) # when app shuts down, close the container

@tool
def execute(command: str) -> str:
    """
    Runs a command in a alpine linux container
    - no internet
    - no bash, only sh
    - every command returns stdout+stderr
    """

    return sandbox.execute(command)
