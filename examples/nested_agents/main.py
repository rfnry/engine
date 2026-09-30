import asyncio

from harness import build, edit
from providers import AnthropicProvider, TypeSafeProvider


async def main() -> None:
    engine = build(AnthropicProvider(), TypeSafeProvider())
    for info in engine.agents.values():
        print(f"{info.name:7} {info.kind:10} calls {info.agents}")

    run = await edit(engine, "Apologise to Dana: her order arrived four days late.")

    for step in run.steps:
        print(f"\n{step.tool} {step.args} -> child {step.child_run_id}\n  {step.result}")
    print()
    for child in run.children:
        usage = child.usage
        print(f"child {child.agent} ({child.kind}) · {usage.input} in / {usage.output} out")
    print(f"\n{run.reply}")


asyncio.run(main())
