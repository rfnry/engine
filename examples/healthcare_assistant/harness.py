from rfnry_engine import Engine

from providers import decision, generation

CALLBACK = "Triage is not sure who should take this: a nurse will call the patient back."

engine = Engine("agents", generation=generation(), decision=decision())


async def intake(clinic: str, message: str) -> dict:
    run = await engine.run("triage", scope=clinic, input={"message": message})
    routed = run.routed_to
    return {
        "answers": {
            qid: answer.model_dump(exclude={"type"}) for qid, answer in run.answers.items()
        },
        "routed_to": routed.agent if routed else None,
        "reply": routed.reply if routed else CALLBACK,
        "run_id": run.run_id,
    }
