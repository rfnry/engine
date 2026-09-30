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
{"reply": "No, Clause 4 is not acceptable and is a high-risk departure from our playbook.\n\n- Playbook requires a mutual indemnity: each party indemnifies for third-party claims caused by its own breach.\n- Clause 4 imposes a one-way indemnity from Customer to Supplier, and it's uncapped and not limited to Supplier's breach (\"arising out of or relating to this Agreement or the Services\").\n\nRedline: Make it mutual — each party defends/indemnifies the other only against third-party claims arising from its own breach, negligence, or willful misconduct, and cross-reference/carve into the liability cap in Clause 5.", "run_id": "4bb4f5bc5747475889e2599f211a84ea"}
```

```bash
curl -s localhost:8000/runs/4bb4f5bc5747475889e2599f211a84ea/review -H 'content-type: application/json' -H 'X-Tenant: firm-acme' -d '{"stance": "objection", "comment": "Right call, but keep the redline to the playbook: breach only, and leave the cap alone.", "correction": "No. Clause 4 is a one-way indemnity from our client, a high risk under the playbook, and it reaches any claim relating to the Agreement. Redline clause 4 to: Each party shall defend, indemnify and hold harmless the other party against any third-party claim, loss or expense arising out of its own breach of this Agreement."}'
```

```json
{"label_id": "53450b95227c4e1cabf4b1a61c95c1ae", "run_id": "4bb4f5bc5747475889e2599f211a84ea", "scope": "firm-acme", "at": "2026-09-30T13:41:25.298067Z", "source": "human", "review": {"stance": "objection", "comment": "Right call, but keep the redline to the playbook: breach only, and leave the cap alone.", "correction": "No. Clause 4 is a one-way indemnity from our client, a high risk under the playbook, and it reaches any claim relating to the Agreement. Redline clause 4 to: Each party shall defend, indemnify and hold harmless the other party against any third-party claim, loss or expense arising out of its own breach of this Agreement."}}
```

```bash
curl -s -X POST localhost:8000/export -H 'X-Tenant: firm-acme'
```

```json
{"keys": ["firm-acme/reviewer/generation/20260930T134125305850Z.jsonl", "firm-acme/reviewer/preference/20260930T134125305850Z.jsonl"], "rows": {"generation": 1, "preference": 1}, "watermark": "2026-09-30T13:41:25.298067Z"}
```
