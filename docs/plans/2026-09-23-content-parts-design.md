# Content parts — design (step 0 of the communication cycle)

Written 2026-09-23. Approved in conversation the same day. Touches rfnry-engine (this repository)
and rfnry-memory (the copied contracts). Lands before the thread / speech / vision rewrites, which
depend on it: vision's `look` returns an image to a model, thread messages carry parts, and every
library in the family speaks one content vocabulary.

## 1. Problem

The engine is text-only by contract and multimodal only by accident:

- `run(input=)` takes `str | dict | list`; anything not a string is JSON-stringified into one text
  message. An image cannot be an input.
- `Message.content` is `str | list[dict]` — untyped, vendor-shaped blocks. An image block slipped
  into `history` works only with an adapter that happens to understand that shape.
- Tool bodies return `str | dict`, stringified. A tool cannot hand the model an image, so a model
  driving vision through tools would be blind.
- Assistant tool calls live as `{"type": "tool_use", ...}` dicts inside `content` — a vendor's shape
  inside a vendor-neutral record.

rfnry-thread has its own content parts (`text`, `image`, `audio`, `document`, `form`); rfnry-memory
has none. Three vocabularies for one idea.

## 2. The parts

One vocabulary, defined in each library's `models.py`, byte-identical across the family (like the
`Tool`/`Tools` block):

```python
class Text(BaseModel):
    type: Literal["text"] = "text"
    text: str

class _Media(BaseModel):
    model_config = ConfigDict(val_json_bytes="base64")
    type: str
    media_type: str
    data: bytes | None = None
    url: str | None = None
    # after-validator: data xor url; json serializer: data as standard base64

class Image(_Media):
    type: Literal["image"] = "image"

class Audio(_Media):
    type: Literal["audio"] = "audio"

class Document(_Media):
    type: Literal["document"] = "document"
    name: str | None = None

Part = Annotated[Text | Image | Audio | Document, Field(discriminator="type")]
Content = str | list[Part]
```

- A media part carries exactly one of `data` or `url`; both or neither is an `InputError`
  (`models: image carries data or url, not both` — the part's `type` names it). One private base
  holds the check, so it is written once.
- `bytes` serialize to standard base64 in JSON (a field serializer: pydantic's
  `ser_json_bytes="base64"` writes the URL-safe alphabet, which vendors and dataset consumers do
  not expect) and parse back from base64 (`val_json_bytes="base64"`, which reads both alphabets,
  in JSON and from a dumped dict alike).
- No `Video` part: frames travel as `Image`, which is how every vendor accepts vision input.
- The library never fetches a `url`; it is carried to the adapter, which decides.

## 3. The engine

**Input.** `run(agent, scope, input, ...)` accepts `str`, a `dict` (JSON state, as today) or
`list[Part]`. A generation run turns the input into the first user message: a string or dict as
today; a list of parts as `Message(role="user", content=parts)`. `State` itself is unchanged
(`str | dict | list`), so a decision model's contract and a list of plain JSON values keep working;
a list is taken as parts when its items are parts, and a list that mixes parts with other values
is an `InputError` (`engine: input mixes parts with other values`). Parts are matched by shape,
like replies: another family library's copy of `Text` or `Image` is read by its fields, so a
vision or thread part reaches the model without either library importing the other.

**Messages.**

```python
class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: Content = ""
    tool_calls: list[ToolCall] = []
    tool_call_id: str | None = None
    reasoning: Any = None
    is_error: bool = False
```

- An assistant turn carries its text in `content` and its calls in `tool_calls` — no dict blocks.
- A tool result is `role="tool"`, `tool_call_id`, `content` (a string or parts), `is_error`.
- `Generation.conversation` and `Generation.messages` use this shape; `history=run.conversation`
  keeps working.

**Tools.** A tool body returns `str | dict | list[Part]`. The shared `Tools` block gains
`Result = str | dict[str, Any] | list[Part]` and its `Body`, `Dispatcher`, `Tool.call` and
`Tools.dispatch` return it (byte-identical change in engine and memory). `ToolsLike.dispatch`
returns `str | dict[str, Any] | list[Any]`: each library's `Part` is its own class, and a list is
invariant, so a narrower protocol would make memory's `MemoryTools` stop type-checking as the
engine's `ToolsLike`. The loop puts a parts result into the tool message's `content` unchanged
and into `Step.result` (now `Content`); a `str` or `dict` as today.

**Decision agents** take text only. A `list[Part]` made only of `Text` is joined (newline) into the
state; any media part is an `InputError` before the model is called
(`engine: decision agent 'triage' takes text only; got image`). A route forwards the original
input to its target, so a generation target still receives the parts.

**Builtins** stay text-only (`Read` refuses binary). A declared tool may return a `Document` or
`Image` part — the docs' `read_pdf` example becomes:

```python
async def read_pdf(args):
    path = Jail("agents/support").resolve(args["path"])
    return [Document(media_type="application/pdf", name=path.name, data=path.read_bytes())]
```

**Refinery.** Runs are recorded with their parts as they are. Export writes parts inline as base64
(the models' JSON form): rows stay self-contained; a consumer strips media if it does not want it.
No reference or blob store is introduced.

**Public names added:** `Text`, `Image`, `Audio`, `Document`, `Part`, `Content`.

## 4. rfnry-memory

- `models.py` gains the same parts block, byte-identical, and the same `Message` shape
  (`content: Content`, `tool_calls`).
- `structured.py` builds its retry turn with `tool_calls` instead of dict blocks.
- The shared `Tools` block widens exactly as in the engine.
- Ingestion stays text-in; no memory verb takes media. `MemoryTools` results stay `str | dict`.
- Public names added: the same six, for parity.

## 5. Adapters (consumer side — examples)

- Every example `AnthropicProvider` and `GoogleProvider` translates `Text`/`Image`/`Document`
  (and `Audio` where the vendor supports it) to vendor blocks, `Message.tool_calls` to tool-use
  blocks, and a tool message's parts to the vendor's tool-result content. The canonical classes
  stay byte-identical across folders.
- New engine example `image_question`: an agent answers a question about a photo passed as
  `input=[Text(...), Image(...)]`, run on both vendors; the same image given to a decision agent
  shows the refusal.
- `agent_knows_its_folder` (or the folder that carries `read_pdf`) returns a `Document`.

## 6. Tests

- Parts: discriminated parsing; data-xor-url; base64 JSON round trip.
- Engine: a scripted model receives parts from input, from history and from a tool result; an
  assistant turn records `tool_calls`; `history=run.conversation` continues with parts intact;
  a decision agent refuses media and joins text-only parts; a route forwards parts to its target.
- Refinery: a run with an image records and exports it as base64; storage suite round-trips parts
  on InMemory and Postgres.
- Memory: structured retry turn uses `tool_calls`; the parts block and `Tools` block are
  byte-identical with the engine (a test or a script diff).
- Docs examples test covers the new blocks; live: `image_question` on both vendors.

## 7. Breaking changes

- `Message.content` is typed (`str | list[Part]`); dict blocks are gone.
- Assistant tool calls move to `Message.tool_calls`.
- Tool bodies may return parts (widening, not breaking for existing tools).
- The `read_pdf` docs example returns a `Document`.

## 8. Pre-change checklist

1. Wanted two model generations ahead? Yes — models are multimodal; the record must carry what
   they see.
2. More useful as models improve? More: substrate, the content the record carries.
3. Not applicable.
4. Exists because a model is weak at X? No.
5. Generation only, decision only, both? Generation models take every part; decision models take
   text, and media is refused up front; a mixed tree routes parts to the generation target.
6. Documentable with a short example? Yes: one `input=[Text(...), Image(...)]` line and the
   three-line `read_pdf`.

## 9. Decisions

| decision | reason |
|---|---|
| One part vocabulary, copied across the family | thread, speech, vision, engine and memory speak one word for a piece of content |
| `data` xor `url` on media parts | inline bytes are what models accept; a url is carried, never fetched |
| No `Video` part | vendors take frames as images; vision sends frames |
| Tool calls get their own `Message` field | a vendor's block shape inside a neutral record was the only untyped seam left |
| Decision agents refuse media up front | decision models are text-only; a clear error beats a JSON dump of bytes |
| Export inlines base64 | rows stay self-contained; no blob store enters the library |
