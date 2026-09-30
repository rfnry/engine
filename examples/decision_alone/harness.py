from rfnry_engine import DecisionModel, Engine, Run


def build(decision: DecisionModel) -> Engine:
    return Engine("agents", decision=decision)


async def sort(engine: Engine, message: str, plan: str) -> Run:
    return await engine.run(
        "inbox",
        scope="demo",
        input={
            "message": message,
            "plan": plan,
        },
    )
