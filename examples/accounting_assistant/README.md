# accounting_assistant

## Description

A bookkeeping assistant for an accounting firm's accountants, working on a client's ledger. It shows a workspace with a shell, and approvals: the agent writes and runs scripts in a folder of its own, each command in a fresh container that sees only that folder, and an accountant allows every shell command before it runs. Look at `workspace` and `engine.run(..., workspace=workspace(work), approve=approve)` in `src/harness.py`.

## Running

```bash
export ANTHROPIC_API_KEY=...
docker build -t acme-books:1 .
uv run src/main.py
```

## Example

```bash
curl -s localhost:8000/tasks -H 'content-type: application/json' -d '{"task": "Total September expenses by account into expenses.csv, and flag any duplicate entries."}'
```

```json
{"id": "43bd69b5", "status": "running", "pending": {"8a49333f": "cd $(dirname $(find / -name ledger.csv 2>/dev/null | head -1)) 2>/dev/null; python3 script.py"}, "reply": null, "run_id": null}
```

```bash
curl -s localhost:8000/approvals/8a49333f -H 'content-type: application/json' -d '{"allow": true}'
```

```json
{"id": "43bd69b5", "status": "done", "pending": {}, "reply": "September expenses have been totaled by account in expenses.csv, and a duplicate check was run.\n\nFiles written:\n- **expenses.csv** — totals by account (positive numbers):\n  - bank fees: 18.00\n  - equipment: 845.00\n  - ingredients: 1766.50\n  - marketing: 160.00\n  - payroll: 6240.00\n  - rent: 2400.00\n  - repairs: 420.00\n  - software: 258.00\n  - utilities: 484.45\n\n- **duplicates.txt** — flags one duplicate entry: counterparty \"Acme Cloud\", reference \"INV-2291\", amount -129.00, appearing on lines 6 and 7 of ledger.csv. Neither row was deleted; please review and decide.", "run_id": "6ff03738a7a44d848181a40c2886c953"}
```
