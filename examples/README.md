# Examples

Each folder is a standalone project: a `main.py` that reads top to bottom, its own `agents/` tree,
a `providers.py` with the vendors that example uses, and a `pyproject.toml` that shows how few
dependencies it needs.

There is one provider per vendor — `AnthropicProvider`, `GoogleProvider`, `TypeSafeProvider`, and
`VoyageProvider` in rfnry-memory's examples — and every folder that uses a vendor carries the same
class, byte for byte, across the family's four libraries; only the line importing the library's
types differs. A folder may carry a method it does not call, such as `generate_stream`. Each creates
one client and retries rate limits and overloads; the generation providers report cache usage and
hand back the model's reasoning so a later turn can replay it. The generation providers translate
the parts — `Text`, `Image`, `Document`, and `Audio` where the vendor takes it — and
`Message.tool_calls` into the vendor's blocks, in user turns and in tool results alike. The decision
providers take a question's structure as it is written: `TypeSafeProvider` sends instructions,
options, levels, `yes` and `no` to the API unchanged, objects and lists included, and refuses more
options or levels than the API takes before sending; `LocalDecisionProvider` writes them into its
prompt, structure as compact JSON.

| example | what it shows |
| --- | --- |
| `question_answer` | one generation agent, `Engine.run`, the reply and usage, a second turn continuing from `history=run.conversation` |
| `decision_alone` | a decision agent with no generation model: a choice, a score and a check, each with written criteria |
| `decision_routes_to_llm` | a decision agent routing to generation agents with `min_confidence`; `routed_to` |
| `nested_agents` | an agent delegating to a generation agent and a decision agent; `Delegate` steps and `children` |
| `agent_knows_its_folder` | a support desk with a policy rule set, a flat skill and a folder skill, a task, a declared tool and an adopted one — `Load` hands the model a PDF price list and a PNG locker map from `knowledge/` as they are, a `Document` and an `Image`, with no parser in between |
| `structured_and_streaming` | `on_event=` printing text as it streams; `expect=` a pydantic model |
| `multi_tenancy` | two tenants on one tree: `context=`, per-tenant `run(tools=)`, rows per scope, `delete_scope` |
| `customer_support_refining` | runs tagged with an experiment arm (`tags=`, carried by every exported row), review, outcome, a judge with `yes`/`no` criteria and a teacher, conflicts, export of the three datasets as JSONL files into `datasets/` — one real row of each in [`datasets/sample/`](customer_support_refining/datasets/sample/) — then a few lines each turning them into TRL `messages`+`tools` (SFT) and `prompt`/`chosen`/`rejected` (DPO), loaded back with Hugging Face `datasets`, and an OpenAI-style chat fine-tuning JSONL |
| `workspace_python` | a coder agent with `workspace: true` writing and running Python in a temporary directory, every command bounded by `Workspace(timeout=60)` and the shell run with `unsandboxed=True`, as it is on a developer's machine; with `main.py ask`, each shell command waits for a y/n in the terminal through `approve=`, while `Write` and `Edit` pass |
| `postgres_refinery` | a refinery on `PostgresStorage` over an asyncpg pool |
| `same_agent_two_vendors` | the same agent run on two generation models, quoting from a CSV and a text file under `knowledge/` |
| `tools_without_a_model` | `EngineTools` driven by code: `dispatch`, `Delegate`, `Jail`, `+`, `only`, `without` |
| `tool_timeout` | a slow tool cut off by `Engine(tool_timeout=)`, reported to the model as an error |
| `decision_vs_generation` | the same 26 labelled routing questions — team and urgency — asked of a decision agent and of a generation agent with `expect=`; accuracy, latency, tokens, an estimated cost at assumed prices, accuracy per confidence bucket, and the cases where they disagree. A small illustrative sample, not a benchmark |
| `decision_on_open_weights` | a decision model on open weights, with no vendor and no key: `LocalDecisionProvider` speaks to any OpenAI-compatible server that returns `top_logprobs` — Ollama, vLLM, llama.cpp's server — asking each question alone and in parallel as lettered options with `max_tokens=1`, the probabilities a softmax over the letters' log-probabilities. The 26 cases of `decision_vs_generation` on its router, on an 8 GB GPU, beside that example's run of Jev and Claude Sonnet 5, with the urgency check's `yes:`/`no:` criteria reaching every model: team 14/26 on `llama3.1:8b` against 25/26 and 24/26, urgency 9/26 against 25/26 and 25/26, the same in three runs, 0.3–1.9 s a case on a shared GPU; `qwen3:8b`, measured before the criteria reached the model, gave 20–21/26 and 16/26. Then rfnry-memory's judge on the same adapter, where `llama3.1:8b` and `qwen3:8b` flag the forum post as an injection and the memo as contradicting the query. Raw log-probabilities from a general model are not calibrated — `llama3.1:8b` gave 0.9 or more to 23 or 24 of its 52 answers and was right on about half of them — and the refinery's `decision` dataset, each answer's probabilities beside what turned out true, is how to measure it |
| `image_question` | `input=[Text(...), Image(...)]` answered on Claude and on Gemini; the same parts refused by a decision agent |

## The agent folder, file by file

| file | example |
| --- | --- |
| `AGENT.md` frontmatter: `kind`, `description` | every example |
| `INDEX.md` — a map read at boot | `agent_knows_its_folder` |
| `rules/_index.md` at boot, `rules/*.md` read on demand | `agent_knows_its_folder`, `same_agent_two_vendors` |
| `skills/<name>.md` and `skills/<name>/SKILL.md` with files beside it, loaded with `Skill` | `agent_knows_its_folder` |
| `tasks/<name>.md`, chosen with `run(task=)` | `agent_knows_its_folder` |
| `tools/<name>.md` — a contract with `input:`, or a name-only adoption of a code tool | `agent_knows_its_folder`, `tool_timeout` |
| `knowledge/_index.md` at boot; any other file (`.csv`, `.txt`, `notes/`) read with `Read` and `Grep` | `same_agent_two_vendors`, `tools_without_a_model` |
| `knowledge/*.pdf` and `knowledge/*.png` — `Load` hands them over as parts the model reads natively; the example's `AnthropicProvider` declares the `media_types` it takes | `agent_knows_its_folder` |
| `questions/*.md` — `choice` with `options:`, `score` with `levels:`, `check` with `yes:`/`no:` | `decision_alone`, `customer_support_refining`, `decision_vs_generation` |
| `route:` with `min_confidence` | `decision_routes_to_llm` |
| `agents:` — delegation through `Delegate` | `nested_agents` |
| `workspace: true` | `workspace_python` |

## Running

The keys live outside the repository, in a file with `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` and
`DECISION_API_KEY`. Run an example from inside its folder:

```bash
cd examples/question_answer
uv run --env-file ../../../.env.examples main.py
```

`uv` builds each folder's environment on first run; the per-folder `uv.lock` files are ignored by git
(`examples/**/uv.lock`). Lint an example with its own ruff: `uv run ruff format . && uv run ruff check .`.

`postgres_refinery` reads `ENGINE_POSTGRES_URL`, defaulting to the development container,
`postgresql://engine:engine@127.0.0.1:55435/engine`.
`workspace_python` runs code the model writes; it does so in a fresh temporary directory.
`decision_on_open_weights` needs no key: it reads `LOCAL_DECISION_URL`, defaulting to Ollama on
port 11434, and runs the models named on its command line, `llama3.1:8b` and `qwen3:8b` when none
is — `ollama pull` them first; Qwen3 answers with its thinking off (`reasoning_effort: "none"`).

The live test tier uses `customer_support_refining/providers.py`. From `packages/python`:

```bash
RFNRY_LIVE_ADAPTERS=providers PYTHONPATH=../../examples/customer_support_refining \
  uv run --project ../../examples/customer_support_refining --with pytest --with pytest-asyncio \
  --env-file ../../../.env.examples pytest -m live -q
```
