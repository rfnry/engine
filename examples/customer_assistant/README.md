## Running

```bash
export ANTHROPIC_API_KEY=...
uv run main.py
```

## Example

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question": "Hi, where is my order A-1001?", "session": "dana-1"}'
```

```json
{"reply": "Hi Dana, I'm sorry for the delay — your order is in transit with Acme Freight, tracking number AF-5520-1187, and it's running past its promised delivery date of 9/28. As an apology, we'll add a $10 credit to your account for your next order. Thanks for your patience!", "run_id": "d4cb96cd3aa74490892e8511cd05164a"}
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question": "Can I send the side table back if I do not like it?", "session": "dana-1"}'
```

```json
{"reply": "Yes, Dana — once the side table is delivered, you can return it within 30 days for a full refund back to your Visa ending 4242, paid once we receive it back. Since it hasn't been delivered yet, just reach out again once it arrives to start the return.", "run_id": "1fa02986273f4dcf976fa1ea1a71d841"}
```

```bash
curl -sN localhost:8000/ask/stream -H 'content-type: application/json' -d '{"question": "I was charged twice for order A-1002. Can I get one back?"}'
```

```text
data: {"text": "H"}

data: {"text": "i Sam, I"}

data: {"text": " can confirm your"}

…

data: {"text": " needed on your end."}

data: {"done": true, "run_id": "757607fbf3544dfcb7dff91f2dcecb70"}
```
