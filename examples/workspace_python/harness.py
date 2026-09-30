from collections.abc import Awaitable, Callable

from rfnry_engine import Engine, GenerationModel, Run, ToolCall, Workspace


def build(generation: GenerationModel) -> Engine:
    return Engine("agents", generation=generation)


async def code(
    engine: Engine,
    task: str,
    work: str,
    approve: Callable[[str, ToolCall], Awaitable[bool]] | None,
) -> Run:
    workspace = Workspace(work, unsandboxed=True, timeout=60)
    return await engine.run("coder", scope="demo", input=task, workspace=workspace, approve=approve)
