# workspaces

## Description

Ways to run a `Workspace`'s commands beyond the bubblewrap recipe in `docs/engine.md`. A workspace
is the library's tools over a backend: `Local` runs each command on this machine, given to the
words of its `shell`, so a sandbox is words in front of the shell.

## A container per command

```python
from rfnry_engine import Local, Workspace

work = "/srv/tenants/acme/work"
local = Local(work, env={"PATH": "/usr/bin:/bin", "DOCKER_CONFIG": "/etc/acme/docker"}, shell=[
    "docker", "run", "--rm", "-i", "--init", "--network", "none",
    "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only", "--tmpfs", "/tmp",
    "--memory", "1g", "--cpus", "1", "--pids-limit", "256", "--user", "1000:1000",
    "-e", f"HOME={work}", "-v", f"{work}:{work}", "-w", work,
    "acme/workspace:1", "timeout", "-s", "KILL", "600", "bash", "-c",
])
workspace = Workspace(local, scope="acme", timeout=600)
```

Each command gets a fresh container from an image you build — a world of its own: no network, no
other tenant, nothing of the service — and `--rm` removes it when the command ends. The model sees
none of this: it has the same tools, and its command runs inside.

The image is the toolchain: a Dockerfile of yours with what your agents need — Python, Go, a C++
compiler. The directory is mounted at the same path, so `Write` and the shell see the same files,
and `--user` is the user that owns it, so a file written on either side stays editable on both.
`env=` is the environment of the `docker` client, not of the command. The client reads its
configuration from `$HOME/.docker`, and `Local`'s default `HOME` is the workspace: a command that
wrote `.docker/config.json` there put `HTTP_PROXY` into the next command's container. So
`DOCKER_CONFIG` points outside the workspace, and the command's own variables go in with `-e`.
Podman reads `$HOME/.config/containers/containers.conf` the same way, and that file can add mounts.

The client passes the engine's SIGTERM on to the container, so the timeout and `stop.now()` stop
it. SIGKILL stops only the client, so a command that ignores SIGTERM would keep its container
running; `timeout -s KILL` inside ends it at the same limit. Each command takes about a quarter of
a second to start.

The caps bound one command; how many run at once is yours to cap — a semaphore around runs — and
`--cgroup-parent` can put every workspace container under one slice, where a limit on the slice
bounds them all. A container shares the host's kernel; for code from many tenants, the gVisor
runtime (`--runtime runsc`) is the stronger boundary. The Docker socket is root on the host, so a
service that can reach it is as good as root; rootless Docker, or Podman — `podman run` takes the
same flags — avoids that.

A container kept for a tenant or a conversation keeps what commands installed between them: start
it with `--init`, and put `docker exec -i -w /srv/tenants/acme/work acme-work timeout -s KILL 600`
in front of the shell. The engine's signals do not reach inside `docker exec`, so that time limit
is the only one that stops a command, `stop.now()` leaves a command running until then, and
starting and removing the container is yours.
