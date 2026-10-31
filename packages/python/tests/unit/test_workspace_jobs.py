import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.errors import InputError
from rfnry_engine.tools.tools import Tool
from rfnry_engine.tools.workspace import Local, Workspace


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


async def call(tool: Tool, args: dict[str, Any]) -> dict[str, Any]:
    result = await tool.call(args)
    assert isinstance(result, dict)
    return result


async def until(check: Callable[[], bool], seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not check():
        assert time.monotonic() < deadline
        await asyncio.sleep(0.02)


async def test_jobs_lists_a_running_and_an_ended_job(root: Path) -> None:
    async with Workspace(Local(root), scope="acme") as work:
        tools = {t.name: t for t in work.tools()}
        await call(tools["Shell"], {"command": "sleep 5", "background": True})
        await call(tools["Shell"], {"command": "true", "background": True})
        await until(lambda: not work.jobs()[1].running)
        first, second = work.jobs()
        assert (first.job, first.command, first.running, first.exit_code) == (
            "j1",
            "sleep 5",
            True,
            None,
        )
        assert (second.job, second.command, second.running, second.exit_code) == (
            "j2",
            "true",
            False,
            0,
        )
        assert first.started.tzinfo is not None
        assert second.started.tzinfo is not None


async def test_output_reads_a_job_without_moving_the_model_s_read(root: Path) -> None:
    async with Workspace(Local(root), scope="acme") as work:
        tools = {t.name: t for t in work.tools()}
        await call(
            tools["Shell"],
            {"command": "printf 'a\\n'; sleep 0.3; printf 'b\\n'", "background": True},
        )
        await until(lambda: work.output("j1") == "a\n")
        assert (await call(tools["Job"], {"id": "j1"}))["output"] == "a\n"
        await until(lambda: not work.jobs()[0].running)
        assert work.output("j1") == "a\nb\n"
        assert (await call(tools["Job"], {"id": "j1", "wait": 1}))["output"] == "b\n"


async def test_output_keeps_the_head_and_tail_as_the_model_s_view(root: Path) -> None:
    async with Workspace(Local(root), scope="acme", output_bytes=10) as work:
        tools = {t.name: t for t in work.tools()}
        await call(tools["Shell"], {"command": "printf 'x%.0s' $(seq 100)", "background": True})
        await until(lambda: not work.jobs()[0].running)
        assert work.output("j1") == "xxxxx\n[... 90 bytes omitted ...]\nxxxxx"


async def test_output_of_an_unknown_job_is_an_input_error(root: Path) -> None:
    async with Workspace(Local(root), scope="acme") as work:
        with pytest.raises(InputError, match="tools/workspace: there is no job j9"):
            work.output("j9")


async def test_the_jobs_tool_lists_what_jobs_returns(root: Path) -> None:
    async with Workspace(Local(root), scope="acme") as work:
        tools = {t.name: t for t in work.tools()}
        await call(tools["Shell"], {"command": "sleep 5", "background": True})
        await call(tools["Shell"], {"command": "true", "background": True})
        await until(lambda: not work.jobs()[1].running)
        listed = await call(tools["Jobs"], {})
        assert listed["jobs"] == [
            {**i.model_dump(), "started": i.started.isoformat(timespec="seconds")}
            for i in work.jobs()
        ]
