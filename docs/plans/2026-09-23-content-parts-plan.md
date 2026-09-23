# Content parts — implementation plan

Executes `2026-09-23-content-parts-design.md` across rfnry-engine and rfnry-memory. Engine first;
memory copies its contracts. Tests first for every behaviour change.

Gate for every commit, from `packages/python`:
`uv run poe check:fix && uv run poe check && uv run poe typecheck && uv run poe test`, then the
Postgres tier (`RFNRY_TEST_POSTGRES_URL=postgresql://harness:harness@127.0.0.1:55435/harness`,
memory on `memory:memory@127.0.0.1:55434/memory`) `uv run poe test:integration`, no skips.
Examples: `uv run ruff format . && uv run ruff check .` per touched folder; each example whose
providers or main changed runs live once; both live e2e tiers run at the end. Before pushing,
`git grep` for the key prefixes finds nothing.

## Engine

**H1 — the parts and the message shape** (`models.py`, `records.py`, `engine.py`,
`tools/tools.py`, `tools/delegate.py`, `__init__.py`, `docs/engine.md`).
- `models.py`: the parts block (`Text`, `Image`, `Audio`, `Document`, `Part`, `Content`), media
  parts refuse both/neither of `data`/`url` with an `InputError`, base64 in JSON; `Message` gains
  `content: Content = ""` and `tool_calls: list[ToolCall]`.
- `engine.py`: a list of parts as input becomes `Message(role="user", content=parts)`; the
  assistant turn carries `tool_calls`; a tool body's parts go into the tool message unchanged and
  into `Step.result` (now `Content`); a decision agent joins Text-only parts into its state and
  refuses media before the model is called; a route forwards the original input.
- `tools/tools.py`: the shared block gains `Result = str | dict[str, Any] | list[Part]` and every
  `str | dict` in it becomes `Result`.
- Tests: `test_models.py` (discriminated parsing, data xor url, base64 JSON round trip);
  `test_engine_generation.py` (parts from input, history and a tool result reach the model;
  assistant `tool_calls`; `history=run.conversation` keeps parts); `test_engine_decision.py`
  (media refused with the rule-9 message and no model call, Text-only parts joined, a route
  forwards parts to its generation target); the existing dict-block assertions rewritten;
  `test_public_api.py` gains the six names; `lookalike.py` mirrors the new `Message`.
- Docs: Models (the parts vocabulary), Running an agent (parts input, `tool_calls`), Tools (parts
  results; `read_pdf` returns a `Document`), Decision agents (text only). `test_docs_examples.py`
  runs the new blocks.

**H2 — the Refinery carries parts** (`tests/unit/test_export.py`, `tests/integration/test_storage.py`).
- Recording already dumps runs in JSON mode; the tests pin it: a run with an image records and
  exports it as base64 in `messages` and `steps`, and the storage suite round-trips a body with
  parts on InMemory and Postgres, validating back to the same `Message`s.

## Memory

**M1 — the same contracts** (`models.py`, `structured.py`, `tools.py`, `__init__.py`,
`docs/memory.md`).
- The parts block byte-identical; the same `Message`; `structured.py` builds its retry turn with
  `tool_calls`; the shared `Tool`/`Tools` block byte-identical (with `Result`).
- Tests: `test_models.py` parts tests; `test_structured.py` asserts the retry turn's `tool_calls`;
  `test_public_api.py` gains the six names; `lookalike.py` mirrors `Message`.
- Docs: one paragraph — parts exist for parity; ingestion stays text-in.

## Byte-identity across the family

**X1 — a unit test in each repository** (`tests/unit/test_family.py`). It cuts the parts block
(`class Text` through `Content =`) out of `models.py` and the tools block (`Result =` through
`Tools.without`) out of the tool module, in this repository and in the sibling checkout
(`../memory` or `../engine` next to the repository root), and asserts they are equal. It skips
when the sibling checkout is absent, so a lone clone still passes. Lands in both repositories after
M1, when the two agree.

## Examples

**E1 — adapters speak parts** (every `providers.py` in both repositories).
- `AnthropicProvider`: `Text`/`Image`/`Document` to `text`/`image`/`document` blocks (base64 or
  url source), assistant `tool_calls` to `tool_use` blocks, a tool message's parts to
  `tool_result` content; `Audio` raises (no vendor support).
- `GoogleProvider`: `Text` to text parts, media to `inline_data` or `file_data`, `tool_calls` to
  function calls, a tool message's media beside its function response.
- The class-hash check (each class hashed across every folder of both repositories) reports one
  hash per class.

**E2 — `image_question`** (new engine example: `pyproject.toml`, `providers.py`, `main.py`,
`agents/`, a generated `photo.png`): an agent answers a question about the image passed as
`input=[Text(...), Image(...)]` on Claude and on Gemini; the same parts given to a decision agent
print the refusal.

**E3 — `read_pdf` returns a `Document`** (`agent_knows_its_folder`): the tool returns the PDF as a
part; a tiny PDF is built at runtime or committed.

Every changed example runs live once; then both live e2e tiers.

## Close

HANDOFF, where present, names the new contracts and the identity test; push both repositories to
`origin main`.
