import asyncio
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

from rfnry_engine import Engine, Local, ToolCall, Workspace

from providers import AnthropicProvider

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Task:
    id: str
    status: str = "running"
    pending: dict[str, str] = field(default_factory=dict)
    reply: str | None = None
    run_id: str | None = None


IMAGE = "acme-books:1"
TIMEOUT = 120


def workspace(work: Path) -> Workspace:
    path = str(work)
    local = Local(
        work,
        env={
            "PATH": os.environ["PATH"],
            "DOCKER_CONFIG": os.environ.get("DOCKER_CONFIG", str(Path.home() / ".docker")),
        },
        shell=[
            *("docker", "run", "--rm", "-i", "--init", "--network", "none", "--cap-drop", "ALL"),
            *("--security-opt", "no-new-privileges", "--read-only", "--tmpfs", "/tmp"),
            *("--memory", "1g", "--cpus", "1", "--pids-limit", "256"),
            *("--user", f"{os.getuid()}:{os.getgid()}", "-e", f"HOME={path}"),
            *("-v", f"{path}:{path}", "-w", path, IMAGE, "timeout", "-s", "KILL", str(TIMEOUT)),
            *("bash", "-c"),
        ],
    )
    return Workspace(local, scope="acme-books", timeout=TIMEOUT)


engine = Engine(ROOT / "agents", generation=AnthropicProvider())
tasks: dict[str, Task] = {}
waiting: dict[str, tuple[Task, asyncio.Future[bool]]] = {}
working: set[asyncio.Task] = set()


async def run(request: str) -> dict:
    task = Task(id=uuid4().hex[:8])
    tasks[task.id] = task
    work = ROOT / ".work" / task.id
    work.mkdir(parents=True)
    shutil.copy(ROOT / "data" / "ledger.csv", work / "ledger.csv")

    async def approve(agent: str, call: ToolCall) -> bool:
        if call.name != "Shell":
            return True
        approval_id = uuid4().hex[:8]
        waiting[approval_id] = (task, asyncio.get_running_loop().create_future())
        task.pending[approval_id] = call.args["command"]
        return await waiting[approval_id][1]

    async def bookkeep() -> None:
        try:
            async with workspace(work) as room:
                engine_run = await engine.run(
                    "bookkeeper",
                    scope="acme-books",
                    input=request,
                    workspace=room,
                    approve=approve,
                )
            task.status, task.reply, task.run_id = "done", engine_run.reply, engine_run.run_id
        except Exception as error:
            task.status, task.reply = "failed", str(error)

    job = asyncio.create_task(bookkeep())
    working.add(job)
    job.add_done_callback(working.discard)
    return await settled(task)


async def answer(approval_id: str, allow: bool) -> dict:
    task, decision = waiting.pop(approval_id)
    del task.pending[approval_id]
    decision.set_result(allow)
    return await settled(task)


def status(task_id: str) -> dict:
    return asdict(tasks[task_id])


async def settled(task: Task) -> dict:
    while task.status == "running" and not task.pending:
        await asyncio.sleep(0.2)
    return asdict(task)
