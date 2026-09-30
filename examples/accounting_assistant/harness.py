import asyncio
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

from rfnry_engine import Engine, ToolCall, Workspace

from providers import AnthropicProvider


@dataclass
class Task:
    id: str
    status: str = "running"
    pending: dict[str, str] = field(default_factory=dict)
    reply: str | None = None
    run_id: str | None = None


engine = Engine("agents", generation=AnthropicProvider())
tasks: dict[str, Task] = {}
waiting: dict[str, tuple[Task, asyncio.Future[bool]]] = {}
working: set[asyncio.Task] = set()


async def start(request: str) -> dict:
    task = Task(id=uuid4().hex[:8])
    tasks[task.id] = task
    work = Path(".work") / task.id
    work.mkdir(parents=True)
    shutil.copy("data/ledger.csv", work / "ledger.csv")

    async def approve(agent: str, call: ToolCall) -> bool:
        if call.name != "Shell":
            return True
        approval_id = uuid4().hex[:8]
        waiting[approval_id] = (task, asyncio.get_running_loop().create_future())
        task.pending[approval_id] = call.args["command"]
        return await waiting[approval_id][1]

    async def run() -> None:
        try:
            done = await engine.run(
                "bookkeeper",
                scope="acme-books",
                input=request,
                workspace=Workspace(work, unsandboxed=True),
                approve=approve,
            )
            task.status, task.reply, task.run_id = "done", done.reply, done.run_id
        except Exception as error:
            task.status, task.reply = "failed", str(error)

    job = asyncio.create_task(run())
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
