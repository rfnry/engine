from pathlib import Path

from rfnry_engine import Engine

from providers import decision, generation

ROOT = Path(__file__).resolve().parent.parent
CALLBACK = "Triage is not sure who should take this: a nurse will call the patient back."

engine = Engine(ROOT / "agents", models={"strong": generation(), "judge": decision()})


async def run(clinic: str, message: str) -> dict:
    engine_run = await engine.run("triage", scope=clinic, input={"message": message})
    routed = engine_run.routed_to
    return {
        "answers": {
            qid: answer.model_dump(exclude={"type"}) for qid, answer in engine_run.answers.items()
        },
        "routed_to": routed.agent if routed else None,
        "reply": routed.reply if routed else CALLBACK,
        "run_id": engine_run.run_id,
    }
