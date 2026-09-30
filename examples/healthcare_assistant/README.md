## Running

```bash
export ANTHROPIC_API_KEY=...
export DECISION_API_KEY=...
export LOCAL_DECISION_MODEL=...
uv run main.py
```

## Example

```bash
curl -s localhost:8000/intake -H 'content-type: application/json' -H 'X-Tenant: clinic-northside' -d '{"message": "Hi, this is Maria Lopez. Can I move my Thursday 10am check-up to sometime next week?"}'
```

```json
{"answers": {"department": {"choice": "scheduling", "probabilities": {"scheduling": 1.0, "nurse_line": 0.0}, "confidence": 1.0}, "urgent": {"probability": 0.02, "confidence": 0.96}}, "routed_to": "scheduling", "reply": "Draft reply:\n\n\"Hi Maria, thanks for reaching out. We'd be happy to reschedule your Thursday 10am check-up to next week. Would [time] on [day] work for you? Let us know if that works or if you need a different time.\"\n\nFront desk to-do:\n- Check calendar for open slots next week\n- Confirm patient's preferred day/time\n- Cancel/move existing Thursday 10am appointment\n- Book new appointment and send confirmation to Maria", "run_id": "c5934db21a7a4067b6aca4a744c4ec57"}
```

```bash
curl -s localhost:8000/intake -H 'content-type: application/json' -H 'X-Tenant: clinic-northside' -d '{"message": "Since starting the new blood pressure pills on Monday I get dizzy when I stand up and nearly fainted this morning."}'
```

```json
{"answers": {"department": {"choice": "nurse_line", "probabilities": {"scheduling": 0.0, "nurse_line": 1.0}, "confidence": 1.0}, "urgent": {"probability": 0.93, "confidence": 0.8600000000000001}}, "routed_to": "nurse_line", "reply": "Summary: Patient started new BP medication Monday, now has orthostatic dizziness and a near-syncope episode this morning — needs prompt callback triage.\n\nQuestions to ask:\n- Exact time/duration of dizziness episodes and today's near-fainting event — any actual loss of consciousness, fall, or injury?\n- Current BP and HR if they have a home cuff; can they check now (lying and standing)?\n…\n\nKey driver of urgency: new antihypertensive + orthostatic symptoms progressing to near-syncope suggests drug-induced hypotension — common but requires timely evaluation before another fall/fainting event occurs.", "run_id": "f29dd7d0b48a49daa02caf80767ff078"}
```

```bash
curl -s localhost:8000/intake -H 'content-type: application/json' -H 'X-Tenant: clinic-northside' -d '{"message": "Please call me about my mother."}'
```

```json
{"answers": {"department": {"choice": "nurse_line", "probabilities": {"scheduling": 0.22, "nurse_line": 0.78}, "confidence": 0.56}, "urgent": {"probability": 0.16, "confidence": 0.6799999999999999}}, "routed_to": null, "reply": "Triage is not sure who should take this: a nurse will call the patient back.", "run_id": "a8c9a92655224e22a8861f966faaaa90"}
```
