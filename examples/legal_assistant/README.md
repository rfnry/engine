# legal_assistant

## Description

A contract reviewer for a law firm's lawyers, checking a contract against the firm's playbook. It shows the refinery: every run is recorded, a lawyer's review becomes a label on it, and the firm exports datasets with every label and its source. Look at `Refinery(PostgresStorage(pool))` passed to `Engine(..., refinery=)`, `refinery.review(...)` and `refinery.export(..., since=)` in `src/harness.py`.

## Running

```bash
export ANTHROPIC_API_KEY=...
docker run -d --name rfnry-engine-pg -p 127.0.0.1:55435:5432 -e POSTGRES_USER=harness -e POSTGRES_PASSWORD=harness -e POSTGRES_DB=harness postgres:16-alpine
export RFNRY_EXAMPLES_POSTGRES_URL=postgresql://harness:harness@127.0.0.1:55435/harness
uv run src/main.py
```

## Example

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -H 'X-Tenant: firm-acme' -d '{"contract": "northwind-msa", "question": "Is the indemnity acceptable for our client?"}'
```

```json
{"reply": "No — Clause 4 is one-sided: it makes Customer solely indemnify Supplier for any claim relating to the Agreement or Services, with no reciprocal protection.\n\nPlaybook position: indemnity should be mutual, each party indemnifying the other only for third-party claims caused by its own breach. A customer-only, broad \"any claim... relating to\" indemnity is high risk and unacceptable.\n\nRedline: Replace Clause 4 with mutual indemnification — each party defends and indemnifies the other against third-party claims arising from its own breach of the Agreement, negligence, or willful misconduct; delete the one-sided \"arising out of or relating to this Agreement or the Services\" scope for Customer.", "run_id": "1235d019a80d4ac98242d914a7d95aa7"}
```

```bash
curl -s localhost:8000/runs/1235d019a80d4ac98242d914a7d95aa7/review -H 'content-type: application/json' -H 'X-Tenant: firm-acme' -d '{"stance": "objection", "comment": "Right call, but keep the redline to the playbook: breach only.", "correction": "No. Clause 4 is a one-way indemnity from our client, a high risk under the playbook, and it reaches any claim relating to the Agreement. Redline clause 4 to: Each party shall defend, indemnify and hold harmless the other party against any third-party claim, loss or expense arising out of its own breach of this Agreement."}'
```

```json
{"label_id": "d58b5a0d7fe24f1b8c0c23cfe6cf3e04", "run_id": "1235d019a80d4ac98242d914a7d95aa7", "scope": "firm-acme", "at": "2026-09-30T15:54:56.375064Z", "source": "human", "review": {"stance": "objection", "comment": "Right call, but keep the redline to the playbook: breach only.", "correction": "No. Clause 4 is a one-way indemnity from our client, a high risk under the playbook, and it reaches any claim relating to the Agreement. Redline clause 4 to: Each party shall defend, indemnify and hold harmless the other party against any third-party claim, loss or expense arising out of its own breach of this Agreement."}}
```

```bash
curl -s -X POST localhost:8000/export -H 'X-Tenant: firm-acme'
```

```json
{"keys": ["firm-acme/reviewer/generation/20260930T155456384618Z.jsonl", "firm-acme/reviewer/preference/20260930T155456384618Z.jsonl"], "rows": {"generation": 1, "preference": 1}, "watermark": "2026-09-30T15:54:56.375064Z"}
```
