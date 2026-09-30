## Running

```bash
export ANTHROPIC_API_KEY=...
uv run main.py
```

## Example

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -H 'X-Tenant: plant-reno' -d '{"question": "M-13 has made short shots since this morning, and barrel zone 3 reads 40 degrees under setpoint.", "task": "troubleshoot"}'
```

```json
{"reply": "M-13 short shots + zone 3 reading 40°C under setpoint — most likely the zone 3 heater band or its thermocouple, especially since that band was just replaced (2026-09-12).\n\n1. **Heater band connection loose/failed (check first, cheapest)**\n   - Check: lock out/tag out M-13 per safety rules (heaters can be up to 300°C — let zone 3 cool below 50°C before touching, gloves + face shield). …\n   - If confirmed: re-terminate the connection. If the band itself is bad, swap it — you have 4 spare HB-60 heater bands in bin C-12.\n …\n\nStart with #1 since it's cheapest and ties directly to the recent Sept 12 zone 3 heater band job — a loose connection from that repair is the most probable cause. Lockout locks/tags are at the tool crib by line 2. If you're not sure after checks 1–2, radio Dana Ortiz (shift lead, channel 3) before escalating to electrical.", "run_id": "6baea806975c440a8f723e07d515e8e4"}
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -H 'X-Tenant: plant-monterrey' -d '{"question": "How do I replace the nozzle heater on MA-2?", "task": "procedure"}'
```

```json
{"reply": "Replace nozzle heater on MA-2 (150-ton all-electric, clean room, line A):\n\n1. Notify Rosa Villarreal (shift lead, ext. 410) that MA-2 is going down for a heater swap.\n2. Gown up for clean room entry before approaching line A.\n3. Lock out and tag MA-2 at the main disconnect; verify zero energy (all-electric — servo drives lock out at the disconnect).\n…\n9. Pull a new nozzle heater 25 mm (NH-25) from cabinet E-2 (3 on hand).\n…\n\nIf the heater doesn't reach/hold temperature or a fault returns, stop and call Tomás Garza (electrician on call, ext. 415) — don't open the electrical cabinet yourself.", "run_id": "94c8aac3149940bfbd5197eabd98b3ba"}
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -H 'X-Tenant: plant-reno' -d '{"question": "The cabinet fan on M-21 is rattling. Can I open the cabinet and look?"}'
```

```json
{"reply": "No — don't open the electrical cabinet yourself. Rule: only the electrician on call opens it.\n\nDo this instead:\n- Note it and radio Dana Ortiz (shift lead, ch. 3) to report the rattle on M-21.\n- Call Luis Park, electrician on call, ext. 231, to open the cabinet and inspect/replace the fan.\n- If you must act, lock out/tag out M-21 first (locks at tool crib by line 2) so no one runs it while waiting — but leave the cabinet itself for Luis.", "run_id": "e570df695b65436fb2a502567102fe23"}
```
