from rfnry_engine import DecisionModel, Engine, GenerationModel, Run


def build(generation: GenerationModel, decision: DecisionModel) -> Engine:
    return Engine("agents", generation=generation, decision=decision)


async def receive(engine: Engine, message: str) -> Run:
    return await engine.run("frontdesk", scope="demo", input=message)
