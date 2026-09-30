import asyncio
import sys
import tempfile
from pathlib import Path

from rfnry_engine import ToolCall

from harness import build, code
from providers import AnthropicProvider

TASK = """Write stats.py with median(xs) and mode(xs), and test_stats.py that checks both
with plain asserts, including an even-length list and an empty list (which must raise
ValueError). Run the tests until they pass."""


async def ask_in_the_terminal(agent: str, call: ToolCall) -> bool:
    if call.name in ("Write", "Edit"):
        return True
    question = f"{agent} wants to run: {call.args.get('command')}  allow? [y/n] "
    answer = await asyncio.to_thread(input, question)
    return answer.strip().lower() == "y"


async def main() -> None:
    engine = build(AnthropicProvider())
    approve = ask_in_the_terminal if sys.argv[1:] == ["ask"] else None

    with tempfile.TemporaryDirectory() as work:
        run = await code(engine, TASK, work, approve)
        for step in run.steps:
            target = step.args.get("path") or step.args.get("command")
            print(f"{step.tool:5} {target}  {step.result[:80]!r}")
        print(f"\nfiles: {sorted(p.name for p in Path(work).iterdir())}")
        print(f"reply: {run.reply}")


asyncio.run(main())
