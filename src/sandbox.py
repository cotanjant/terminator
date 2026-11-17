import os
from pathlib import Path

import docker
from docker.errors import ImageNotFound

DEFAULT_IMAGE = "terminator-sandbox:22.04"
REPO_ROOT = Path(__file__).resolve().parent.parent


class Sandbox:
    """Throwaway, network-isolated Ubuntu Docker container for running agent commands."""

    def __init__(self, image: str | None = None, workspace: str | None = None):
        self.client = docker.from_env()
        self.last_exit_code: int = 0
        image = image or os.getenv("TERMINATOR_IMAGE") or DEFAULT_IMAGE

        self._ensure_image(image)

        volumes = None
        if workspace:
            host_path = str(Path(workspace).resolve())
            volumes = {host_path: {"bind": "/terminator", "mode": "rw"}}

        self.container = self.client.containers.run(
            image,
            command="sleep infinity",
            detach=True,
            network_mode="none",
            mem_limit="512m",
            nano_cpus=1_000_000_000,
            pids_limit=256,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges"],
            working_dir="/terminator" if workspace else "/",
            volumes=volumes,
        )

    def _ensure_image(self, image: str) -> None:
        """Build the default sandbox image from Dockerfile.sandbox if not yet present."""
        try:
            self.client.images.get(image)
        except ImageNotFound:
            if image == DEFAULT_IMAGE:
                self.client.images.build(
                    path=str(REPO_ROOT),
                    dockerfile="Dockerfile.sandbox",
                    tag=DEFAULT_IMAGE,
                    rm=True,
                )
            else:
                raise

    def execute(self, cmd: str, timeout: int = 60) -> str:
        """Execute a shell command inside the sandbox with a timeout."""
        result = self.container.exec_run(
            ["timeout", str(timeout), "sh", "-c", cmd],
            demux=True,
        )
        out, err = result.output
        self.last_exit_code = result.exit_code
        text = (out or b"").decode(errors="replace") + (err or b"").decode(errors="replace")
        if result.exit_code in (124, 143):
            text += "\ntimeout"
        return text[:10_000] or f"no output, exit code: {result.exit_code}"

    def close(self):
        """Force-remove the sandbox container."""
        self.container.remove(force=True)
