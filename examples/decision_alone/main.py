import asyncio

from rfnry_engine import Engine

from providers import TypeSafeProvider

MESSAGE = "We were charged twice this month and nobody answers the chat. Get me a manager."


async def main() -> None:
    engine = Engine("agents", decision=TypeSafeProvider())

    run = await engine.run(
        "inbox",
        scope="demo",
        input={
            "message": MESSAGE,
            "plan": "enterprise",
        },
    )

    department = run.answers["department"]
    urgency = run.answers["urgency"]
    person = run.answers["person"]
    print(f"department  {department.choice}  {department.probabilities}", end="")
    print(f"  confidence {department.confidence:.2f}")
    levels = [round(p, 3) for p in urgency.probabilities]
    print(f"urgency     level {urgency.score:.0f}  {levels}  confidence {urgency.confidence:.2f}")
    print(f"person      p(yes) {person.probability:.3f}  confidence {person.confidence:.2f}")
    print(f"{run.model} · {run.usage.input} in / {run.usage.output} out · {run.duration_ms} ms")


asyncio.run(main())
