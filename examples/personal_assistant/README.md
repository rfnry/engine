# personal_assistant

## Description

A personal assistant for one person's home, receipts and paperwork. It shows multimodal input: a photo or a PDF comes in as a multipart upload and goes to the agent next to the question. Look at the `Text`, `Image` and `Document` parts passed as `input=` to `engine.run(...)` in `src/harness.py`, and `attachment` in `src/utils.py`, which turns the upload into one. A session keeps its conversation and passes it with where to start (`window` in `src/harness.py`), so a long one sends only its recent turns and the agent reads further back with `History`; the date in `context=` restarts the model's cache once a day.

## Running

```bash
export ANTHROPIC_API_KEY=...
uv run src/main.py
```

## Example

```bash
curl -s localhost:8000/ask -F question="My dishwasher is leaking. Is it still under warranty, and who do I call?" -F file=@data/label.png -F session=jordan-1
```

```json
{"reply": "Yes — your Acme DW-450 dishwasher (serial AX-99812) is under warranty until 2027-04-02, and leaks are explicitly covered.\n\nCall Acme Appliances service at 1-800-555-0142, quoting serial AX-99812. Keep the dishwasher switched off until the technician arrives.", "run_id": "96a8b6622005439cbeb9f49fa4123d54"}
```

```bash
curl -s localhost:8000/ask -F question="And if the washing machine breaks too?" -F session=jordan-1
```

```json
{"reply": "Bad news there — the washing machine (Acme WM-700, serial WX-30317) warranty ended 2025-06-15, so it's no longer covered. Any repair would be out of pocket; you'd need to arrange a technician yourself rather than claim through Acme.", "run_id": "948ac8698fec4ed9be93f4c3a6e3fbfe"}
```

```bash
curl -s localhost:8000/ask -F question="The shelf said avocados were 2 for 3 dollars. Was I charged right?" -F file=@data/receipt.pdf
```

```json
{"reply": "No. You bought 4 avocados at $1.75 each = $7.00. At \"2 for $3\" (i.e. $1.50 each), 4 should cost $6.00. You were overcharged $1.00.", "run_id": "117a8f96075d481fb40815ed78c140e2"}
```
