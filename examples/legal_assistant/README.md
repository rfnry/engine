## Running

```bash
export ANTHROPIC_API_KEY=...
export DECISION_API_KEY=...
# optional: export LOCAL_DECISION_MODEL=llama3.1:8b
uv run main.py
```

## Example

```bash
curl -s localhost:8000/classify -H 'content-type: application/json' -H 'X-Tenant: firm-acme' -d '{"clause": "Supplier may terminate this Agreement for convenience on thirty (30) days'"'"' written notice."}'
```

```json
{"path": ["termination", "convenience"], "levels": [{"kind": "termination", "confidence": 1.0, "run_id": "e8a41a3f2cd34eaaa3d0bb8d4f17eeff"}, {"kind": "convenience", "confidence": 1.0, "run_id": "fa7550b2ba9d4781aaf3d5526c83c107"}]}
```

```bash
curl -s localhost:8000/review -H 'content-type: application/json' -H 'X-Tenant: firm-acme' -d '{"contract": "northwind-msa"}'
```

```json
{"issues": [{"clause": 2, "kind": "Payment", "risk": "low", "problem": "Invoices are due in 15 days and late interest is 2% per month, both worse than the playbook's minimum 30-day payment term and 1%/month interest cap.", "redline": "Replace with: Customer shall pay each invoice within thirty (30) days of its date. Overdue amounts bear interest at one percent (1%) per month until paid."}, …, {"clause": 4, "kind": "Indemnity", "risk": "high", "problem": "The indemnity runs only from Customer to Supplier and is not limited to claims caused by Customer's breach, contrary to the playbook's requirement of a mutual indemnity tied to each party's own breach.", "redline": "Replace with: Each party shall defend, indemnify and hold harmless the other party against any third-party claim, loss or expense arising out of its own breach of this Agreement."}, …], "asked": [{"agent": "clauses", "task": "risk", "run_id": "929ce2961e9642299545ea03294877be"}, {"agent": "clauses", "task": "risk", "run_id": "1cc80afae76e4869ac828f21121a6fe5"}, …], "run_id": "9357398faa934c28ae2f702aa98ddcab"}
```

```bash
curl -s localhost:8000/runs/9357398faa934c28ae2f702aa98ddcab/review -H 'content-type: application/json' -H 'X-Tenant: firm-acme' -d '{"stance": "approval", "comment": "Right calls. Send the redlines for clauses 4 to 6 first."}'
```

```json
{"label_id": "cfc23251223147548353731d90a7456f", "run_id": "9357398faa934c28ae2f702aa98ddcab", "scope": "firm-acme", "at": "2026-09-30T04:43:58.455088Z", "source": "human", "review": {"stance": "approval", "comment": "Right calls. Send the redlines for clauses 4 to 6 first."}}
```
