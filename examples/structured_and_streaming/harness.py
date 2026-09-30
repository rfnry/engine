from collections.abc import Awaitable, Callable

from pydantic import BaseModel
from rfnry_engine import Engine, Event, GenerationModel, Run


class Verdict(BaseModel):
    refund: bool
    amount: float
    reason: str


def build(generation: GenerationModel) -> Engine:
    return Engine("agents", generation=generation)


async def explain(
    engine: Engine, request: str, on_event: Callable[[Event], Awaitable[None]]
) -> Run:
    return await engine.run(
        "reviewer",
        scope="demo",
        input=f"Explain your decision to the customer: {request}",
        on_event=on_event,
    )


async def decide(
    engine: Engine, request: str, on_event: Callable[[Event], Awaitable[None]]
) -> Verdict:
    run = await engine.run(
        "reviewer", scope="demo", input=request, expect=Verdict, on_event=on_event
    )
    return Verdict.model_validate_json(run.reply)
