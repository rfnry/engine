# accounting_assistant

## Description

A bookkeeping assistant for an accounting firm's accountants, working on a client's ledger. It shows a workspace with a shell, and approvals: the agent writes and runs scripts in a folder of its own, each command in a fresh container that sees only that folder, and an accountant allows every shell command before it runs. The agent's own folder is mounted read-only in each container at `/agent`, so the firm's `skills/month-close/close.py` runs there as it is, and the accountant approves a script the firm wrote instead of a new one each time. Look at `workspace` and `async with workspace(work) as room:` around `engine.run(..., workspace=room, approve=approve)` in `src/harness.py`; leaving the block closes the workspace and stops any job the agent left running.

## Running

```bash
export ANTHROPIC_API_KEY=...
docker build -t acme-books:1 .
uv run src/main.py
```

## Example

```bash
curl -s localhost:8000/tasks -H 'content-type: application/json' -d '{"task": "Close September 2026 for this client."}'
```

```json
{"id": "8b7ca69a", "status": "running", "pending": {"77e257c3": "cd /tmp && python3 /agent/skills/month-close/close.py 2026-09 --help 2>&1 | head -5; echo ---; python3 /agent/skills/month-close/close.py 2026-09 2>&1; echo RUN; ls"}, "reply": null, "run_id": null}
```

```bash
curl -s localhost:8000/approvals/77e257c3 -H 'content-type: application/json' -d '{"allow": true}'
```

```json
{"id": "8b7ca69a", "status": "running", "pending": {"ab205498": "python3 /agent/skills/month-close/close.py 2026-09"}, "reply": null, "run_id": null}
```

```bash
curl -s localhost:8000/approvals/ab205498 -H 'content-type: application/json' -d '{"allow": true}'
```

```json
{"id": "8b7ca69a", "status": "done", "pending": {}, "reply": "September 2026 is closed. Here's the summary from close.py (rows 18, counting the duplicate row):\n\n**Totals**\n- Money in: $6,196.25\n- Money out: $12,591.95\n- Net: -$6,395.70\n\n**Expenses by account** (written to close-2026-09.csv):\n- bank fees: 18.00\n- equipment: 845.00\n- ingredients: 1,766.50\n- marketing: 160.00\n- payroll: 6,240.00\n- rent: 2,400.00\n- repairs: 420.00\n- software: 258.00\n- utilities: 484.45\n\n**Possible duplicate**\n- Lines 6 and 7: Acme Cloud, INV-2291, -$129.00, both dated 2026-09-05.\n\nLooking at the ledger, these two rows are identical in every field — same date, counterparty, reference, and amount. This reads like the same software invoice (INV-2291) entered or charged twice, not two separate real charges (a second real charge would normally carry a different invoice/reference number or date). The accountant should check Acme Cloud's billing statement/bank feed to confirm whether this invoice was actually charged twice, and if so, reverse one of the entries. Both rows are retained in the ledger and in the totals above (so the $258.00 software expense and the overall money-out figure include both charges of $129.00).\n\nFile written: close-2026-09.csv (expenses by account for September 2026), in the workspace.", "run_id": "7680423a5d204a9faa8429bc86f86512"}
```
