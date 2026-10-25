# factory_assistant

## Description

A maintenance assistant for technicians across two plants of one manufacturer. It shows tenants: the `X-Tenant` header becomes the run's `scope`, and each plant's own lines, contacts, parts on hand and recent work go in as `context`. Look at `engine.run("maintenance", scope=plant, context=facts)` in `src/harness.py`.

It also shows a tool that computes, in worker processes of its own. `agents/maintenance/tools/vibration_trend.py` reads one machine's daily vibration from `data/readings.json`, for the plant in `call["scope"]`, and returns the mean, the RMS, the slope of a least-squares line and whether the last three readings are over the machine's alert level; `vibration_trend.json` describes it to the model, and its `output_schema` checks every result before the model reads it. Every folder tool runs in a worker process beside the server, never in it, and without `workers=` the whole tree shares one; `workers={"maintenance": 2}` in `src/harness.py` gives the folder a group of two. A tool that computes is better given its own: two technicians' calls compute in two processes at once, where in one they would share its interpreter, and an agent added outside the folder runs its tools in the default worker, never behind these.

## Running

```bash
export ANTHROPIC_API_KEY=...
uv run src/main.py
```

## Example

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -H 'X-Tenant: plant-reno' -d '{"question": "Who is the electrician on call, and where are the spare type J thermocouples?"}'
```

```json
{"reply": "Electrician on call: Luis Park, ext. 231.\n\nSpare thermocouples (type J, TCJ-2): 6 in stock, bin C-14.", "run_id": "19d63e84aa234e90b00fc42fe9c8aeac"}
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -H 'X-Tenant: plant-monterrey' -d '{"question": "Who is the electrician on call, and where are the spare type J thermocouples?"}'
```

```json
{"reply": "Electrician on call: Tomás Garza, ext. 415.\n\nSpare type J thermocouples (TCJ-2): 2 in stock, cabinet E-2.", "run_id": "7842e9f0ac994fb1bd304c3dfaad9852"}
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -H 'X-Tenant: plant-reno' -d '{"question": "M-13 has made short shots since this morning, and barrel zone 3 reads 40 degrees under setpoint."}'
```

```json
{"reply": "M-13, barrel zone 3, short shots, zone reading 40°C under setpoint — likely a heater/thermocouple fault (zone 3's heater band was just replaced 9/12, so check the new install first).\n\nLikely causes, cheapest/easiest check first:\n\n1. **Loose heater band wiring/connector at zone 3** (common after a recent swap) — Lock out and tag the machine, verify zero energy, let barrel cool below 50°C, then visually check the zone 3 heater band terminals and lead wires for a loose or backed-out connection.\n2. **Bad or shifted thermocouple (TCJ-2)** reading low, so controller heats less — With machine still locked out, check TCJ-2 seating in the barrel well; if loose or corroded, swap it. 6 in stock, bin C-14.\n…\n4. **Controller output or SSR for zone 3 not firing** — This means opening the electrical cabinet: stop here, call Luis Park (electrician on call, ext. 231), do not open the cabinet yourself.\n\nSteps:\n1. Radio Dana Ortiz (channel 3) to flag M-13 down.\n2. Get lockout lock/tag from the tool crib at line 2.\n…\n10. Once fixed, remove lockout, restart, and confirm zone 3 climbs to setpoint before running production.", "run_id": "ecc67d8b02aa4dd59732973376d35ae0"}
```
