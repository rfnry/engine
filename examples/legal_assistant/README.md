# legal_assistant

## Description

A contract reviewer for a law firm's lawyers, checking a contract against the firm's playbook. It shows the refinery: every run is recorded, a lawyer's review becomes a label on it, and the firm exports datasets with every label and its source. Look at `Refinery(InMemoryStorage())` passed to `Engine(..., refinery=)`, `refinery.review(...)` and `refinery.export(..., since=)` in `src/harness.py`.

## Running

```bash
export ANTHROPIC_API_KEY=...
uv run src/main.py
```

## Example

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -H 'X-Tenant: firm-acme' -d '{"contract": "northwind-msa", "question": "Is the indemnity acceptable for our client?"}'
```

```json
{"reply": "No — Clause 4 is not acceptable. It's a one-sided indemnity, with Customer (our client) alone defending and indemnifying Supplier for any claim relating to the Agreement or Services. Our playbook requires a mutual indemnity, each party indemnifying the other only for third-party claims caused by its own breach.\n\nRedline: Replace clause 4 with mutual indemnification language — each party defends, indemnifies and holds harmless the other from third-party claims arising from its own breach of the Agreement, negligence, or willful misconduct, deleting the one-way obligation on Customer.", "run_id": "50ccff9d9b244cf199add5987ffa6ecd"}
```

```bash
curl -s localhost:8000/runs/50ccff9d9b244cf199add5987ffa6ecd/review -H 'content-type: application/json' -H 'X-Tenant: firm-acme' -d '{"stance": "objection", "comment": "Right call, but keep the redline to the playbook: breach only, and leave the cap alone.", "correction": "No. Clause 4 is a one-way indemnity from our client, a high risk under the playbook, and it reaches any claim relating to the Agreement. Redline clause 4 to: Each party shall defend, indemnify and hold harmless the other party against any third-party claim, loss or expense arising out of its own breach of this Agreement."}'
```

```json
{"label_id": "b2acac606159495c94d75fb81ade752a", "run_id": "50ccff9d9b244cf199add5987ffa6ecd", "scope": "firm-acme", "at": "2026-09-30T15:53:00.098782Z", "source": "human", "review": {"stance": "objection", "comment": "Right call, but keep the redline to the playbook: breach only, and leave the cap alone.", "correction": "No. Clause 4 is a one-way indemnity from our client, a high risk under the playbook, and it reaches any claim relating to the Agreement. Redline clause 4 to: Each party shall defend, indemnify and hold harmless the other party against any third-party claim, loss or expense arising out of its own breach of this Agreement."}}
```

```bash
curl -s -X POST localhost:8000/export -H 'X-Tenant: firm-acme'
```

```json
{"keys": ["firm-acme/reviewer/generation/20260930T155300105750Z.jsonl", "firm-acme/reviewer/preference/20260930T155300105750Z.jsonl"], "rows": {"generation": 1, "preference": 1}, "watermark": "2026-09-30T15:53:00.098782Z"}
```
