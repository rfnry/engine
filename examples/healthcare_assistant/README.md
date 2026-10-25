# healthcare_assistant

## Description

A triage assistant for a clinic's staff, reading each patient message before a person does. It shows decision models: a decision agent answers typed questions with probabilities and routes the message to a generation agent that drafts for the staff, unless its confidence is under the floor. Look at `Engine(..., models={"strong": ..., "judge": ...})`, whose names the `model:` of each `AGENT.md` picks, `engine_run.answers` and `engine_run.routed_to` in `src/harness.py`, and the `route:` block with its `min_confidence` in `agents/triage/AGENT.md`.

It also shows a question whose answers live in data. `agents/triage/questions/doctor.json` asks which of the clinic's doctors a message is about, and `doctor.py` beside it fills the options from `data/clinics.json` for the run's scope, so each clinic is asked about its own doctors under one question id; the answer comes back with the others, `none` when no doctor is named. An `X-Tenant` that is not a clinic of `data/clinics.json` is a 404.

And a tool shared from a group folder. `agents/tools/clinic_hours.py`, in the `tools/` folder at the top of the tree, gives the clinic's opening hours to every generation agent under it, the clinic taken from `call["scope"]`, never from what the model writes; triage, a decision agent, is offered no tools. `tools: [clinic_hours]` in `agents/scheduling/AGENT.md` keeps that one tool and leaves out the file tools, since scheduling has no files to read, while `nurse_line`, without `tools:`, keeps every tool it sees.

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
{"answers": {"department": {"choice": "scheduling", "probabilities": {"scheduling": 1.0, "nurse_line": 0.0}, "confidence": 1.0}, "doctor": {"choice": "none", "probabilities": {"dr-osei": 0.0, "dr-lindqvist": 0.0, "dr-haddad": 0.0, "none": 1.0}, "confidence": 1.0}, "urgent": {"probability": 0.02, "confidence": 0.96}}, "routed_to": "scheduling", "reply": "Draft reply to patient:\n\n\"Hi Maria, thanks for reaching out! We'd be happy to reschedule your check-up from Thursday 10am to next week. We have availability on [time] — would that work for you? Let us know and we'll get it booked.\"\n\nFront desk to-do:\n- Check calendar for open slots next week (Mon–Fri 8:00–18:00, Wed until 20:00, Sat 9:00–12:00) and fill in [time] option(s).\n- Confirm patient's preferred day/time before finalizing.\n- Cancel original Thursday 10am appointment once new time is confirmed.\n- Update scheduling system/EHR with new appointment time.", "run_id": "d0fad5f71a0c4cae83cf27ae9b70c61c"}
```

```bash
curl -s localhost:8000/intake -H 'content-type: application/json' -H 'X-Tenant: clinic-northside' -d '{"message": "Since starting the new blood pressure pills on Monday I get dizzy when I stand up and nearly fainted this morning."}'
```

```json
{"answers": {"department": {"choice": "nurse_line", "probabilities": {"scheduling": 0.0, "nurse_line": 1.0}, "confidence": 1.0}, "doctor": {"choice": "none", "probabilities": {"dr-osei": 0.0, "dr-lindqvist": 0.0, "dr-haddad": 0.0, "none": 1.0}, "confidence": 1.0}, "urgent": {"probability": 0.92, "confidence": 0.8400000000000001}}, "routed_to": "nurse_line", "reply": "Callback Prep – Dizziness/Near-Syncope on New BP Medication\n\nSummary: Patient started a new antihypertensive Monday, now reports orthostatic dizziness and a near-fainting episode this morning.\n\nQuestions to ask:\n- Exact medication name, dose, and timing of doses\n- Did fainting actually occur, or just lightheadedness? Any fall or injury?\n…\n\nUrgency assessment:\n- Concerning: new medication + near-syncope suggests symptomatic hypotension, a known medication side effect that can worsen and risks falls/injury.\n…\n\nClinic hours if visit needed: Mon/Tue/Thu 8:00-18:00, Wed 8:00-20:00, Fri 8:00-16:00, Sat 9:00-12:00, closed Sunday.", "run_id": "aea8d7ae74ef40428c84ee82afbac07e"}
```

```bash
curl -s localhost:8000/intake -H 'content-type: application/json' -H 'X-Tenant: clinic-northside' -d '{"message": "I have a question about my visit last week."}'
```

```json
{"answers": {"department": {"choice": "scheduling", "probabilities": {"scheduling": 0.7, "nurse_line": 0.3}, "confidence": 0.3999999999999999}, "doctor": {"choice": "none", "probabilities": {"dr-osei": 0.0, "dr-lindqvist": 0.0, "dr-haddad": 0.0, "none": 1.0}, "confidence": 1.0}, "urgent": {"probability": 0.07, "confidence": 0.8599999999999999}}, "routed_to": null, "reply": "Triage is not sure who should take this: a nurse will call the patient back.", "run_id": "e6e3f09d20194316a3f1384af06ac22f"}
```

```bash
curl -s localhost:8000/intake -H 'content-type: application/json' -H 'X-Tenant: clinic-northside' -d '{"message": "Is Dr. Haddad in on Saturday? I'\''d like him to look at my ECG results."}'
```

```json
{"answers": {"department": {"choice": "scheduling", "probabilities": {"scheduling": 0.68, "nurse_line": 0.32}, "confidence": 0.3600000000000001}, "doctor": {"choice": "dr-haddad", "probabilities": {"dr-osei": 0.0, "dr-lindqvist": 0.0, "dr-haddad": 1.0, "none": 0.0}, "confidence": 1.0}, "urgent": {"probability": 0.06, "confidence": 0.8799999999999999}}, "routed_to": null, "reply": "Triage is not sure who should take this: a nurse will call the patient back.", "run_id": "cd4507c34b74408cbf692cfd081de250"}
```

Triage is sure of the doctor but not of the department: its confidence there, 0.36, is under the 0.6 floor, so the message goes to no one and a nurse calls back. With `X-Tenant: clinic-riverside` the same question asks about that clinic's own doctors: `dr-moreau`, `dr-tanaka` and `none`.
