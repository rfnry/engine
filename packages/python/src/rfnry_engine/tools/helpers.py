import asyncio
from typing import Any, cast

from pydantic import BaseModel

from rfnry_engine.errors import InputError
from rfnry_engine.helpers import HELPER_ID, Helper, Helpers, Unanswered
from rfnry_engine.tools.tools import Tool
from rfnry_engine.tools.workspace import JOB_TOOL, JOBS_TOOL, JobArgs, Workspace

HELPER_TOOLS = frozenset({JOB_TOOL, JOBS_TOOL})
ASK_TOOL = "Ask"
ENDED = "and whether it is running or, once it has ended, how: finished, failed or stopped."


class HelperJobArgs(JobArgs):
    send: str | None = None


class AskArgs(BaseModel):
    question: str


def _helper(deadline: float | None) -> str:
    upto = "" if deadline is None else f", up to {deadline:g},"
    left = "" if deadline is None else f", for what is left of the call's {deadline:g} seconds"
    return (
        "whether it is running, its agent, its input, under asking the question it waits on you to "
        "answer through Ask (null when none), and once it has ended its reply (a decision agent's "
        f"answers), its error, or what it had said when it was stopped. wait is seconds{upto} to "
        "wait first; it returns as soon as the helper ends or any notice from a helper reaches "
        "you, and at once while the helper waits on your answer. "
        "stop=true stops the helper and every helper it started, after the wait, and waits for it "
        f"to end{left}. send answers the question under asking; when there is none, it gives a "
        "running generation helper a message, which it reads before its next model call; a "
        "decision helper takes none. Its end reaches you as a notice whether or not you read it "
        "here."
    )


def _described(workspace: Workspace | None, deadline: float | None) -> tuple[str, str]:
    helper = _helper(deadline)
    if workspace is None:
        return (
            "Read a helper you started with Delegate in the background, by its id (d1, d2, …): "
            f"{helper}",
            "List the helpers you started in the background: id, agent, input, when it started, "
            f"{ENDED}",
        )
    return (
        "Read a background job of the workspace (j1, j2, …) or a helper you started with Delegate "
        "in the background (d1, d2, …). A job: what it printed since the last read, stdout and "
        "stderr in the order they arrived, whether it is running, and its exit_code once it has "
        f"ended; wait is seconds, up to {workspace.timeout:g}, to wait for it to end first; "
        "stop=true stops it, after the wait; its output keeps its first and last "
        f"{workspace.output_bytes // 2} bytes; send is for a helper only. A helper: {helper}",
        "List the workspace's background jobs and the helpers you started in the background. A "
        "job: id, command, when it started, whether it is running and its exit_code; a job that "
        "ended is listed until Job reads its last output. A helper: id, agent, input, when it "
        f"started, {ENDED}",
    )


def _checked(wait: float, deadline: float | None) -> None:
    if deadline is not None and not 0 <= wait <= deadline:
        raise InputError(
            f"tools/helpers: wait must be between 0 and {deadline:g} seconds; got {wait:g}"
        )
    if not wait >= 0:
        raise InputError(f"tools/helpers: wait must not be negative; got {wait:g}")


def helper_tools(
    helpers: Helpers, workspace: Workspace | None, deadline: float | None
) -> list[Tool]:
    held = {} if workspace is None else {tool.name: tool for tool in workspace.tools()}

    async def job(args: dict[str, Any]) -> dict[str, Any]:
        a = HelperJobArgs.model_validate(args)
        helper = helpers.get(a.id)
        if helper is None:
            commands = [] if workspace is None else [info.job for info in workspace.jobs()]
            if a.send is not None and a.id in commands:
                raise InputError(
                    f"tools/helpers: job {a.id} is a command; send reaches only a helper"
                )
            if workspace is None or a.send is not None or HELPER_ID.fullmatch(a.id) is not None:
                raise InputError(f"tools/helpers: there is no helper {a.id}")
            return cast(dict[str, Any], await held[JOB_TOOL].call(args))
        _checked(a.wait, deadline)
        loop = asyncio.get_running_loop()
        until = None if deadline is None else loop.time() + deadline
        if a.send is not None:
            if helper.kind == "decision":
                raise InputError(
                    f"tools/helpers: helper {a.id} is a decision run and reads no messages"
                )
            if helper.ended is not None:
                raise InputError(
                    f"tools/helpers: helper {a.id} has ended; send reaches only a helper still running"
                )
            await helpers.send(a.id, a.send)
        await helpers.wait(a.id, a.wait)
        if a.stop:
            await helpers.stop_one(a.id, None if until is None else max(0.0, until - loop.time()))
        return helpers.read(a.id)

    async def jobs(args: dict[str, Any]) -> dict[str, Any]:
        commands = (
            []
            if workspace is None
            else cast(dict[str, Any], await held[JOBS_TOOL].call({}))["jobs"]
        )
        return {"jobs": [*commands, *helpers.listed()]}

    reading, listing = _described(workspace, deadline)
    return [
        Tool(JOB_TOOL, reading, body=job, args=HelperJobArgs),
        Tool(JOBS_TOOL, listing, body=jobs),
    ]


def ask_tool(helper: Helper, helpers: Helpers) -> Tool:
    lead = helpers.lead

    async def ask(args: dict[str, Any]) -> str:
        answer = await helpers.ask(helper, AskArgs.model_validate(args).question)
        if isinstance(answer, str):
            return answer
        if answer is Unanswered.STOPPED:
            raise InputError(
                "tools/helpers: your lead was stopped or failed before it answered; you stop with it"
            )
        raise InputError("tools/helpers: your lead ended its turn without answering")

    return Tool(
        ASK_TOOL,
        f"Ask {lead}, the agent that started you in the background, a question, and wait for its "
        f"answer: its answer is this call's result. If {lead} ends its turn without answering, "
        "the call ends with an error, and you go on with what you have.",
        body=ask,
        args=AskArgs,
    )
