## Running

```bash
export ANTHROPIC_API_KEY=...
uv run main.py
```

## Example

```bash
curl -s localhost:8000/tasks -H 'content-type: application/json' -d '{"task": "Total September expenses by account into expenses.csv, and flag any duplicate entries."}'
```

```json
{"id": "d4609775", "status": "running", "pending": {"e6a07943": "cd $(dirname $(find / -name ledger.csv 2>/dev/null|head -1)) 2>/dev/null; cd - >/dev/null; python3 process.py"}, "reply": null, "run_id": null}
```

```bash
curl -s localhost:8000/approvals/e6a07943 -H 'content-type: application/json' -d '{"allow": false}'
```

```json
{"id": "d4609775", "status": "running", "pending": {"308e8652": "python3 process.py"}, "reply": null, "run_id": null}
```

```bash
curl -s localhost:8000/approvals/308e8652 -H 'content-type: application/json' -d '{"allow": true}'
```

```json
{"id": "d4609775", "status": "done", "pending": {}, "reply": "I totaled September expenses by account and wrote them to expenses.csv:\n\n- bank fees: 18.00\n- equipment: 845.00\n- ingredients: 1766.50\n- marketing: 160.00\n- payroll: 6240.00\n- rent: 2400.00\n- repairs: 420.00\n- software: 258.00\n- utilities: 484.45\n\nDuplicate flagged: lines 6 and 7 (both \"Acme Cloud, INV-2291, -129.00\") — same counterparty, reference, and amount. Neither row was deleted, per the rule; the software total above includes both entries as they appear in the ledger.\n\nFile written: expenses.csv (also process.py, the script used).", "run_id": "022fd9808313421bb28cee47843d3d40"}
```
