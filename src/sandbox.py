from os import error
from base64 import decode

import docker

class Sandbox:
    def __init__(self, image="ubuntu:22.04", workspace=None):
        self.client = docker.from_env()
        # we don't give agent's sandbox a volume
        # we want this to be a throwaway execution container

        # this one starts container
        self.container = self.client.containers.run(
            image,
            command="sleep infinity", # to wait it to get up
            detach=True,
            network_mode="none",
            mem_limit="512m",
            nano_cpus=1_000_000_000, # 1 cpu
            pids_limit=256,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges"],
            working_dir="/terminator" if workspace else "/",
            # like i said no volumes
            # if we add volumes add this line
            # volumes = volumes
            # declare the volumes variable up
        )

    # timeout's limits are 60 because a minute = 60 seconds you knıw :(
    def execute(self, cmd: str, timeout: int = 60) -> str:
        result = self.container.exec_run(
            ["timeout", str(timeout), "sh", "-c", cmd],
            demux=True,
        )
        out, err = result.output
        text = (out or b"").decode(errors="replace") + (err or b"").decode(errors="replace")
        if result.exit_code in (124, 143):
            text += "\ntimeout"
        return text[:10_000] or f"no output, exit code: {result.exit_code}"

    def close(self):
        self.container.remove(force=True)
