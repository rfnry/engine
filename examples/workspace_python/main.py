import asyncio
import tempfile
from pathlib import Path

from rfnry_engine import Engine, Workspace

from providers import AnthropicProvider

TASK = """Write stats.py with median(xs) and mode(xs), and test_stats.py that checks both
with plain asserts, including an even-length list and an empty list (which must raise
ValueError). Run the tests until they pass."""


async def main() -> None:
    engine = Engine("agents", generation=AnthropicProvider(), tool_timeout=60.0)

    with tempfile.TemporaryDirectory() as work:
        run = await engine.run("coder", scope="demo", input=TASK, workspace=Workspace(work))
        for step in run.steps:
            target = step.args.get("path") or step.args.get("command")
            print(f"{step.tool:5} {target}  {step.result[:80]!r}")
        print(f"\nfiles: {sorted(p.name for p in Path(work).iterdir())}")
        print(f"reply: {run.reply}")


asyncio.run(main())
