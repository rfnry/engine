# customer_assistant

## Description

The support desk of an online home-goods store, answering customers about their orders. It shows the basic agent: a folder of markdown under `agents/support` that picks its tool by name in `tools:`, and that tool defined in code, answering in one reply or streaming it as it is written. Look at `Engine(..., tools=[Tool("lookup_order", description, body=lookup_order, args=LookupArgs)])` and `engine.run(..., history=, start=, on_event=)` in `src/harness.py`, where `TextDelta` events feed the stream and `window` keeps a long session to its recent turns while the agent reads further back with `History`. The date in `context=` restarts the model's cache once a day.

## Running

```bash
export ANTHROPIC_API_KEY=...
uv run src/main.py
```

## Example

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question": "Hi, where is my order A-1001?", "session": "dana-1"}'
```

```json
{"reply": "Hi Dana, your order is with Acme Freight, tracking number AF-5520-1187, currently in transit. I'm sorry it's past its promised delivery date of 9/28 — as an apology, we've added a $10 credit to your account for your next order. Thanks for your patience!", "run_id": "5bf046b4e6504986aa3b66f872aff896"}
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question": "Can I send the side table back if I do not like it?", "session": "dana-1"}'
```

```json
{"reply": "Since the order hasn't been delivered yet, it isn't eligible for a return quite yet — but once it arrives, you can return the oak side table within 30 days for a full refund back to your Visa ending 4242. Just reach out once you've received it and want to send it back.", "run_id": "459c1f538a20439bbc5439f97ab42768"}
```

```bash
curl -sN localhost:8000/ask/stream -H 'content-type: application/json' -d '{"question": "I was charged twice for order A-1002. Can I get one back?"}'
```

```text
data: {"text": "Hi S"}

data: {"text": "am, I"}

data: {"text": " can confirm your"}

…

data: {"text": " within five business days."}

data: {"done": true, "run_id": "2b1928811ccd4de89efc9f8a8119ee2e"}
```
