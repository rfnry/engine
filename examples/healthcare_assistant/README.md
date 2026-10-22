# healthcare_assistant

## Description

A triage assistant for a clinic's staff, reading each patient message before a person does. It shows decision models: a decision agent answers typed questions with probabilities and routes the message to a generation agent that drafts for the staff, unless its confidence is under the floor. Look at `Engine(..., models={"strong": ..., "judge": ...})`, whose names the `model:` of each `AGENT.md` picks, `engine_run.answers` and `engine_run.routed_to` in `src/harness.py`, and the `route:` block with its `min_confidence` in `agents/triage/AGENT.md`.

## Running

```bash
export ANTHROPIC_API_KEY=...
export DECISION_API_KEY=...
# optional: export LOCAL_DECISION_MODEL=llama3.1:8b
uv run src/main.py
```

## Example

```bash
curl -s localhost:8000/intake -H 'content-type: application/json' -H 'X-Tenant: clinic-northside' -d '{"message": "Hi, this is Maria Lopez. Can I move my Thursday 10am check-up to sometime next week?"}'
```

```json
{"answers": {"department": {"choice": "scheduling", "probabilities": {"scheduling": 1.0, "nurse_line": 0.0}, "confidence": 1.0}, "urgent": {"probability": 0.02, "confidence": 0.96}}, "routed_to": "scheduling", "reply": "Draft reply:\n\n\"Hi Maria, thanks for reaching out! We'd be happy to move your Thursday 10am check-up to next week. We have [time] available — would that work for you? Let us know and we'll get it booked.\"\n\nFront desk to-do:\n- Cancel/hold the existing Thursday 10am appointment\n- Check availability for next week and offer a specific time to Maria\n- Confirm new appointment once she responds and update the schedule", "run_id": "b4d3b5b2b0b04648aef650e7a8bb5fc7"}
```

```bash
curl -s localhost:8000/intake -H 'content-type: application/json' -H 'X-Tenant: clinic-northside' -d '{"message": "Since starting the new blood pressure pills on Monday I get dizzy when I stand up and nearly fainted this morning."}'
```

```json
{"answers": {"department": {"choice": "nurse_line", "probabilities": {"scheduling": 0.0, "nurse_line": 1.0}, "confidence": 1.0}, "urgent": {"probability": 0.92, "confidence": 0.8400000000000001}}, "routed_to": "nurse_line", "reply": "Summary: Patient started new BP medication Monday, now has orthostatic dizziness and a near-syncope episode this morning.\n\nQuestions to ask:\n- Exact medication name/dose and time of last dose(s)\n- Did you actually fall or hit your head? Any injury?\n…\n\nUrgency assessment: Moderate-urgent — near-syncope with a new antihypertensive suggests symptomatic orthostatic hypotension and fall risk; needs same-day nurse/provider contact and possibly BP check, but call 911/ED now if there was actual loss of consciousness, injury, chest pain, or neuro symptoms.", "run_id": "b9a075476a8344da9bd3b6620f8d0bf5"}
```

```bash
curl -s localhost:8000/intake -H 'content-type: application/json' -H 'X-Tenant: clinic-northside' -d '{"message": "I have a question about my visit last week."}'
```

```json
{"answers": {"department": {"choice": "scheduling", "probabilities": {"scheduling": 0.7, "nurse_line": 0.3}, "confidence": 0.3999999999999999}, "urgent": {"probability": 0.07, "confidence": 0.8599999999999999}}, "routed_to": null, "reply": "Triage is not sure who should take this: a nurse will call the patient back.", "run_id": "912b96dd75b44011977355accb0f13ce"}
```
