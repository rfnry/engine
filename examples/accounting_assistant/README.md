## Running

```bash
export ANTHROPIC_API_KEY=...
uv run src/main.py
```

## Example

```bash
curl -s localhost:8000/tasks -H 'content-type: application/json' -d '{"task": "Total September expenses by account into expenses.csv, and flag any duplicate entries."}'
```

```json
{"id": "9c3f9519", "status": "running", "pending": {"bb2560c5": "head -5 ledger.csv; wc -l ledger.csv"}, "reply": null, "run_id": null}
```

```bash
curl -s localhost:8000/approvals/bb2560c5 -H 'content-type: application/json' -d '{"allow": true}'
```

```json
{"id": "9c3f9519", "status": "running", "pending": {"a5aba7f9": "python3 process.py; echo ---; cat expenses.csv"}, "reply": null, "run_id": null}
```

```bash
curl -s localhost:8000/approvals/a5aba7f9 -H 'content-type: application/json' -d '{"allow": true}'
```

```json
{"id": "9c3f9519", "status": "done", "pending": {}, "reply": "I totaled September expenses by account into expenses.csv (accounts and their totals, positive dollars, rounded to cents):\n\n- bank fees: 18.00\n- equipment: 845.00\n- ingredients: 1766.50\n- marketing: 160.00\n- payroll: 6240.00\n- rent: 2400.00\n- repairs: 420.00\n- software: 258.00\n- utilities: 484.45\n\nDuplicate check: lines 6 and 7 in ledger.csv have the same counterparty, reference, and amount — flagging for your review, no rows deleted.\n\nFiles written: expenses.csv, process.py (the script used to generate it).", "run_id": "3c9d46c0cac2459d82382efb36d5a73a"}
```
