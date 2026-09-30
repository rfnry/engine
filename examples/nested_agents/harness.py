from rfnry_engine import DecisionModel, Engine, GenerationModel, Run


def build(generation: GenerationModel, decision: DecisionModel) -> Engine:
    return Engine("agents", generation=generation, decision=decision)


async def edit(engine: Engine, request: str) -> Run:
    return await engine.run("editor", scope="demo", input=request)
