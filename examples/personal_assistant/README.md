## Running

```bash
export ANTHROPIC_API_KEY=...
uv run main.py
```

## Example

```bash
curl -s localhost:8000/ask -F question="My dishwasher is leaking. Is it still under warranty, and who do I call?" -F file=@data/label.png -F session=jordan-1
```

```json
{"reply": "Yes — your dishwasher (Acme DW-450, serial AX-99812) is still under warranty until 2027-04-02, and leaks are explicitly covered.\n\nCall Acme Appliances service at 1-800-555-0142, quoting serial AX-99812. Keep the dishwasher off/unplugged until the technician arrives.", "run_id": "c866d12417134cd5bfbfca3b5af64d7c"}
```

```bash
curl -s localhost:8000/ask -F question="And if the washing machine breaks too?" -F session=jordan-1
```

```json
{"reply": "The washing machine (Acme WM-700, serial WX-30317) is no longer under warranty — it ended 2025-06-15. You'd need to pay for repair out of pocket or contact Acme Appliances service (1-800-555-0142) for a paid service call.", "run_id": "65a6b0c10435424a92c1911592c4d823"}
```

```bash
curl -s localhost:8000/ask -F question="The shelf said avocados were 2 for 3 dollars. Was I charged right?" -F file=@data/receipt.pdf
```

```json
{"reply": "No. You were charged $1.75 each (4 for $7.00), not the 2-for-$3 price ($1.50 each). At the 2-for-$3 rate, 4 avocados should cost $6.00, so you were overcharged $1.00.", "run_id": "116af7748f73413086b77d2c70d12555"}
```
