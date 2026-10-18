"""Container sandbox lifecycle for the Terminator agent.

A sandbox is a throwaway Docker container that the agent uses as its
environment.  This module owns the whole lifecycle -- create -> start ->
publish ports -> stop -> remove -- and stays free of any agent framework
(LangChain/LangGraph) so it can be exercised on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import docker

DEFAULT_IMAGE = "alpine:latest"
DEFAULT_PORT = 4444
DEFAULT_MEM_LIMIT = "512m"
DEFAULT_CPU_QUOTA = 50_000


@dataclass
class Sandbox:
    """A running, ephemeral container used as the agent's environment."""

    container: Any = field(repr=False)
    image: str
    host_port: int | None = None

    @property
    def id(self) -> str:
        return self.container.id

    @property
    def status(self) -> str:
        return self.container.status

    def stop(self) -> None:
        self.container.stop()

    def remove(self, force: bool = False) -> None:
        self.container.remove(force=force)


def _client() -> docker.DockerClient:
    return docker.from_env()


def start_sandbox(
    image: str = DEFAULT_IMAGE,
    command: str | list[str] | None = None,
    port: int = DEFAULT_PORT,
    environment: dict[str, str] | None = None,
    name: str | None = None,
    client: docker.DockerClient | None = None,
) -> Sandbox:
    """Create and start a fresh sandbox container.

    The container is *not* removed here -- call :meth:`Sandbox.remove` (or
    :func:`stop_and_remove`) when you are done with it.
    """
    docker_client = client or _client()

    create_kwargs: dict[str, Any] = {
        "image": image,
        "command": command,
        "ports": {f"{port}/tcp": ("127.0.0.1", port)} if port else None,
        "mem_limit": DEFAULT_MEM_LIMIT,
        "cpu_quota": DEFAULT_CPU_QUOTA,
        "network_mode": None,
        "environment": environment or {},
        "detach": True,
    }
    if name:
        create_kwargs["name"] = name

    container = docker_client.containers.create(**create_kwargs)
    container.start()
    container.reload()

    host_port = _host_port(container, port)
    return Sandbox(container=container, image=image, host_port=host_port)


def _host_port(container: Any, port: int) -> int | None:
    """Read the host port Docker published for ``port``.

    The published-port mapping is only populated *after* the container has
    been started, so it must not be trusted from the object returned by
    ``create``.
    """
    ports = container.attrs.get("NetworkSettings", {}).get("Ports") or {}
    bindings = ports.get(f"{port}/tcp")
    if not bindings:
        return None
    return int(bindings[0]["HostPort"])


def stop_and_remove(sandbox: Sandbox, force: bool = True) -> None:
    """Best-effort teardown of a sandbox started by :func:`start_sandbox`."""
    try:
        sandbox.stop()
    finally:
        sandbox.remove(force=force)


def main() -> None:
    """Smoke test: start a sandbox and print its published port.

    The container needs a long-running command, otherwise Docker clears the
    published-port mapping the moment the container exits.
    """
    sandbox = start_sandbox(command="sleep 300")
    try:
        print(f"started sandbox {sandbox.id[:12]} on host port {sandbox.host_port}")
    finally:
        stop_and_remove(sandbox)


if __name__ == "__main__":
    main()
