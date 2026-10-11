# rfnry-engine

rfnry-engine is an agent harness with built-in dataset generation.

It is the scaffolding that turns a model into an agent — the loop, the tools, delegation between
agents — as a small library: an agent is a folder of markdown, running on a generation model, a
decision model, or both. Opt-in, it records every run, gathers labels from people, the world and
models, and exports datasets to evaluate, improve or train what comes next.

An agent is one of two kinds. A **generation agent** puts a language model in a loop with tools
until it has an answer. A **decision agent** asks a decision model a set of typed questions about a
state and gets typed, probabilistic answers back in one call. Agents nest in both directions: a
generation agent can delegate to any agent, a decision agent can route to any agent. Opt-in, a
refinery records what your agents did and exports it as datasets, with every label and its
source — from people, the world and models. What you do with them — evaluate, improve prompts and
rules, train or distill models, audit — is yours.

Every public name, with its signature, is in [the API reference](api.md).

## Philosophy

**The tree is the agent.** Everything a person needs to author, review or diff about an agent is a
markdown file under version control. The library reads the tree; it never writes it.

**Two kinds of model, one or both.** Open-ended work — conversation, tool use, writing — is
generation. Closed-set judgement — which of these, how much, is this true — is decision. A
generation model does the first well and the second expensively; a decision model does the second
in milliseconds with calibrated probabilities and cannot do the first at all. The library treats
them as two contracts and lets you build an agent from either or both.

**The library runs; it does not take over.** You bring the models, the conversation history, the
bodies behind tools, the storage connection, the bucket. There is no background loop, no plugin
registry, no configuration language. Every verb is a function you call.

**Nothing is hidden.** A run returns a record: what was sent, what came back, every tool call with
its timing, every model call with its usage, every child run. That record is what you feed your own
telemetry, and it is what the refinery stores and exports — everything that could be useful, for
any purpose, without assuming one.

**Tenancy by construction.** Every run names its scope. Every stored row carries it. There is no
default scope and nothing reads across scopes. Multi-tenancy is the default.

## Models

The library speaks to models through two protocols. Any object with the right method is a model;
adapters to vendors are yours to write and never ship here.

```python
class GenerationModel(Protocol):
    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ReplyLike: ...

class DecisionModel(Protocol):
    async def decide(self, state: State, questions: Mapping[str, QuestionLike]) -> Mapping[str, AnswerLike] | DecidedLike: ...
```

Models are matched by shape, statically and at runtime: `ReplyLike`, `DecidedLike` and `AnswerLike` are
protocols over the fields of `Reply`, `Decided` and the answers below, and a result is read by those
fields, not its class. `QuestionLike` is the same over the questions a decision model is handed; an
adapter branches on `question.type` and reads `options`, `levels` or `yes` and `no`. An adapter typed
against another library's copy of these contracts — rfnry-memory's — fits here unchanged, so one
adapter serves both libraries: annotated with `QuestionLike` from either library, it serves both,
statically and at runtime. A result whose fields do not fit is a `ModelError`.

`Message`, `ToolSpec`, `ToolCall` and `Reply` are the message shapes the loop exchanges with a
generation model. An assistant turn is a `Message(role="assistant", content=…, tool_calls=[…])`:
its text in `content`, its calls as `ToolCall`s. A tool result is a
`Message(role="tool", tool_call_id=…, content=…)` whose `is_error` says the call failed, for
adapters whose vendor marks failed tool results. `messages` may hold consecutive
`role="user"` messages — the boot ahead of the history, several messages from the inbox — so an
adapter whose vendor needs alternating roles merges them. A `Reply`
whose `stop_reason` is `"tool_use"` and whose `tool_calls` is non-empty means "run these and call
me again"; anything else is final:

```python
Reply(text="", tool_calls=[ToolCall(id="c1", name="Read", args={"path": "rules/refunds.md"})], stop_reason="tool_use")
Reply(text="Refunded the second charge.", stop_reason="end_turn", usage=Usage(input=812, output=41), model="…")
```

A message's `content` is a string or a list of parts — one vocabulary for everything a model can
see, shared by every library in the family:

```python
Text(text="Is this invoice charged twice?")
Image(media_type="image/png", data=png)
Document(media_type="application/pdf", name="invoice.pdf", url="https://example.com/invoice.pdf")
Audio(media_type="audio/wav", data=wav)
```

`Part` is any of the four and `Content` is `str | list[Part]`. A media part carries exactly one of
`data` (bytes) or `url`; both or neither is an `InputError`. Bytes travel as base64 in JSON. The
library never fetches a `url` — it carries it to your adapter, which decides; turning parts into a
vendor's blocks, and refusing what the vendor cannot take, is the adapter's job. There is no video
part: frames travel as images. Parts are matched by shape too, so another family library's copy of
them is accepted as it is.

Structured output is a tool call — the model is offered an `Output` tool — so an adapter needs
nothing beyond tool support. An adapter that also implements
`generate_stream(system, messages, tools, on_delta)` streams text when the caller asks for events. An adapter that has `media_types`, the file types its model takes —
`{"image/*", "application/pdf"}` — lets `Load` refuse the rest with an error the model can act on,
instead of the next model call failing; without it, `Load` hands the model any file. An adapter that
has `model`, the name of the model it calls, names the run's model before any reply does.

A decision model answers three kinds of question:

```python
Choice(instructions="Which department should handle this?", options={"billing": "…", "technical": "…"})
Score(instructions="How urgent is the request?", levels=["Can wait days", "Today", "Within the hour", "Outage"])
Check(instructions="Does the user ask to speak with a person?")
```

and returns, respectively, a `ChoiceAnswer` (`choice`, `probabilities`), a `ScoreAnswer` (`score`,
`probabilities`), or a `CheckAnswer` (`probability`). Each carries a `confidence` in `[0, 1]` that
the library computes from the distribution's peakedness, `(n·p_max − 1)/(n − 1)` — the same
formula for every adapter and every type, so a threshold means the same thing everywhere. A
generation model can stand in as a decision model through an adapter that asks it for
probabilities; the calibration is worse, the code is the same. `decide` may return the answers
alone, or a `Decided(answers, usage, model)` when the adapter can report what the call cost and
which model answered; the record carries both, and the adapter's `model` attribute when the call
reports no model. A decision adapter may carry `media_types` too, the media its model reads in a
state; without it, its model is handed text.

The instructions, what an option means, each level, and `yes` and `no` are strings, or JSON objects
or lists when a question has parts of its own — the question beside the data it names in
backticks, options told apart by what they cover and what they do not. An option may mean `None`
when its key says enough. The record keeps the structure as it was asked, and the adapter passes it
to its model as it is, or renders it where its model reads a prompt:

```python
Choice(
    instructions={"question": "Which team should handle this message?", "focus": "The main request, not every topic"},
    options={
        "billing": {"what": "Charges, invoices, refunds", "not_for": "Delivery", "examples": ["I was charged twice"]},
        "orders": {"what": "Delivery, cancellation, returns", "examples": ["Where is my package?"]},
        "other": None,
    },
)
```

Fakes are enough to run everything in tests:

```python
class Scripted:
    def __init__(self, *replies): self.replies = list(replies)
    async def generate(self, system, messages, tools): return self.replies.pop(0)

class Fixed:
    async def decide(self, state, questions):
        return {"intent": ChoiceAnswer(choice="billing", probabilities={"billing": 0.8, "technical": 0.15, "other": 0.05}),
                "urgent": CheckAnswer(probability=0.91)}
```

## Getting started

```python
engine = Engine(root=Path("agents"), generation=llm, tools=tools)

run = await engine.run("support", scope="acme", input="My invoice was charged twice.")
print(run.reply)
```

A root of agent directories, a generation model, and `tools` — the bodies behind the tools the
agents declare, one list for the whole tree ([Tools](#tools) builds it). One call runs an agent for
a tenant and returns its record.

## Core features

### An agent is a directory

```
agents/
  support/
    AGENT.md              who the agent is
    INDEX.md              a map of the directory, read at boot
    rules/_index.md       what the agent must do — the index is read at boot, the files on demand
    rules/refunds.md
    skills/_index.md      what the agent can do — the index at boot, a skill's body when loaded
    skills/escalate.md
    knowledge/_index.md   what the agent knows — the index at boot, the files on demand
    knowledge/rates.pdf   knowledge is text, images or PDFs
    tools/lookup_order.md
    tools/fetch_invoice.md
    tasks/triage.md       a named task, chosen per run
```

`AGENT.md` makes a directory an agent. Its frontmatter names the kind and what the agent may call;
its body is the persona:

```markdown
---
name: support
kind: generation
description: Handles technical support conversations
agents: [triage]
---
You are the support desk for Acme. Be brief, be exact, cite the rule you are applying.
```

At boot the model receives `AGENT.md`, `INDEX.md`, the three `_index.md` files, the list of tools it
has, the descriptions of the agents it may call, the task if one was named, and any context the
caller passes. Everything else it reads when it needs it, with the `Read`, `Grep`, `Glob`, `LS`,
`Skill` and `Load` tools, and it can read only inside its own directory. `Read` returns a text file
as text. `Load` hands the model any other file as it is — an image, a PDF, a recording, a
spreadsheet — so a price list, a diagram or a scanned form sits in `knowledge/` as it came, the
model reads it with its own understanding of images and documents, there is no parser in between,
and nothing is loaded until the model asks — unless the agent asks for its folder at boot, below.
Write the maps with full paths —
`rules/refunds.md`, not `refunds.md` — because a model opens exactly what the map names.
`engine.agents` lists the tree as the engine loaded it: each name to an `AgentInfo` with its
`kind`, `description` and the `agents` it may call.

An agent that needs its whole folder on every run — a few house rules applied to thousands of
documents — asks for it at boot:

```markdown
---
name: tagger
kind: generation
description: Tags each incoming document by the house rules
boot: all
---
Tag the document. Every rule applies to every document.
```

With `boot: all` the model also receives every other file under `rules/`, `skills/` and
`knowledge/`, sorted by relative path character by character — `knowledge/` before `rules/` before
`skills/` — each under its path: what `Read` would return for it, and for a file `Read` refuses —
over 5 MB, or not text — the part `Load` would hand over. The maps stay maps and the folder stays
as it is; only when the files are read changes, so no turn goes to opening them. They travel as one
user message ahead of the history, sent with every model call, and are read when the engine loads
the tree, as the fingerprint is taken, so an edit shows in the next `Engine`. `run.boot` holds that
message's parts; `run.messages` and `run.conversation` leave it out, so a conversation passed back
does not send the folder twice and a replay boots from the candidate's folder. The tools stay as
they are — `Read`, `Load` and `Skill` hand over what the model already holds. Each agent boots its
own folder, a delegated or routed one included. A decision agent reads no files, so `boot:` on one
is a `ConfigurationError` when the tree loads; so is a file neither `Read` nor `Load` would hand
over — over 20 MB, of a type the generation model's `media_types` does not name, or not the type its
name says — with `Load`'s own message.

### Running an agent

`engine.run(agent, scope, input)` names the agent, the tenant and the input, and returns a record.
For a generation agent that record is a `Generation`: the reply, the system prompt as sent, the files `boot: all` sent (`run.boot`), the messages as sent to the first model call, the whole conversation, the tools the model was offered (`run.tools`, each a `ToolSpec`), every tool
call as a `Step` with its result and timing, the model calls and their summed usage, and any child
runs. It is the whole observability payload; hand it to your own telemetry:

```python
run.run_id, run.model, run.calls, run.usage.input, run.usage.output, run.duration_ms
for step in run.steps:
    step.tool, step.args, step.is_error, step.duration_ms, step.child_run_id
run.children                                         # runs made through Delegate, in step order
```

`run.model` is the model the replies reported, the resolved one — a dated snapshot, say. When no
reply reported one — a run stopped during its first model call — it is the adapter's `model`
attribute, if it has one that is a non-empty string, and `None` otherwise.

Every run says which agent and which tools produced it, beside the model that answered:

```python
run.fingerprint.agent                                # "3f9a1c0e5b7d2a41": the agent's folder
run.fingerprint.tools                                # "a02c77e1f4b9d305": the tools it was offered
```

`fingerprint.agent` is a short sha256 over every file in the agent's directory, by relative path and
content — the only directory an agent reads — leaving out hidden files and folders, as `LS`, `Glob`
and `Grep` do, and links that lead out of it. The engine takes it once, when it loads the tree, so
an edit shows in the next `Engine` built from the folder. `fingerprint.tools` is one over
`run.tools` — each tool's name, description and schema — so it covers the tools the folder
declares, the code tools it adopts, the `tools=` a run passes, `Output` when `expect=` asks for one,
and in `Delegate` the agents it may call — their names, descriptions, kinds and tasks. Neither sees
what varies per run and is not the agent — the input, the history, `context=`, `tags=` — so two
runs of an unchanged folder share it and any edit changes it. The model is `run.model`; the code
behind a tool is yours to version. A decision run carries one too, its `tools` the fingerprint of no
tools and its `questions` the questions it asked ([Decision agents](#decision-agents)), and each
delegated or routed run carries its own. A run recorded before fingerprints were kept reads back
with `None`.

Keep the conversation yourself and pass it back. `run.conversation` is the history the run was
sent, the input, every assistant turn — its text in `content`, its calls in `tool_calls`, its `reasoning` as
the model returned it — every tool result with its `is_error`, and the final assistant turn, so the
next run continues from what the agent did, not only what it said:

```python
run = await engine.run("support", scope="acme", input="My invoice was charged twice.")
next = await engine.run("support", scope="acme", input="And the second charge?", history=run.conversation)
```

A conversation that runs for months is passed whole with where each run starts, and the run sends
only what follows: [Long conversations](#long-conversations).

The input is a string, a JSON object, or a list of parts — what the user sent, as they sent it:

```python
parts = [Text(text="Is this invoice charged twice?"), Image(media_type="image/png", data=png)]
run = await engine.run("support", scope="acme", input=parts)
```

A list of parts becomes the first user message as it is; parts in `history` reach the model
unchanged, and a later run continues from them. History, like input, is read by shape: a message or
part from another library is read by its fields.

Pass `record=False` to run without the refinery recording it — a replay, an evaluation, a dry run.

A run that raises instead of returning — the adapter broke the contract (`ModelError`), the loop
passed `max_steps` (`RunawayError`) — returns no record, and nothing of it is recorded, child runs
included; the exception is what you have.

Pass `tags=` to mark a run with what you will group it by later — the commit of the tree, an
experiment arm, a release. Keys and values are strings — at most 32 tags, keys of 1 to 64
characters, values up to 256 — anything else is an `InputError`. The record and every exported row
carry them; the model never sees them. Delegated and routed runs inherit the tags of the run that
started them, so a version or an arm covers the whole tree of runs:

```python
run = await engine.run("support", scope="acme", input=message, tags={"commit": "4f2c9e1", "arm": "b"})
run.tags                                             # {"commit": "4f2c9e1", "arm": "b"}
```

Ask for a typed reply and the model is offered an `Output` tool shaped like it. When the model
calls it with input that fits, the run ends and `run.reply` is that input as JSON; input that does
not fit goes back to the model with what is wrong. A model that answers in text instead ends the
run with that text as `run.reply`, which does not parse:

```python
class Verdict(BaseModel):
    refund: bool
    reason: str

run = await engine.run("support", scope="acme", input=message, expect=Verdict)
verdict = Verdict.model_validate_json(run.reply)
```

An accepted `Output` ends the run. A call the model made after it in the same turn is not run; it is
answered with an error saying so, so every call in `run.conversation` has its result.

Name a task to put `tasks/<task>.md` in front of the model — a decision agent's task adds a set of
questions, [below](#decision-agents) — pass `context=` to add facts about this
tenant that do not belong in the tree — the record keeps it as `run.context` — and pass `on_event=`
to receive text deltas, tool starts and tool results as they happen:

```python
async def show(event):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)

run = await engine.run("support", scope="acme", input=message, task="triage",
                       context="Acme is on the enterprise plan. Refunds up to $500 need no approval.",
                       on_event=show)
```

A `ToolCallStarted` carries the call's `args`, and a `ToolResultReady` its `step`: the same `Step`
the run records, sent however the call ended — a result, an error, a refusal, a delegation, a stop —
so a step can be shown, in a conversation or anywhere, while the run goes on.

Every model call is a turn, and a `TextDelta` carries the `turn` that said it, from 0. Text of a
turn that ends in tool calls — "Let me look that up." — is said beside them; a run that ends in
text replies with its last turn's text, so `run.reply` is that text and nothing said before it. A
delegated or routed run sends its own `run_id` and counts its own turns.

A person can stop or steer a run while it works. Pass `stop=Stop()` and call `stop.now()` to cancel
the model call in flight and every tool still running, or `stop.after_step()` to let the current
tool calls finish and end before the next model call. Pass `inbox=asyncio.Queue()` and put a string
or a list of parts on it whenever the person says something: at each step boundary — after the
tool results, before the next model call — the run takes everything waiting and appends each as a
`role="user"` message, in order:

```python
stop, inbox = Stop(), asyncio.Queue()
work = asyncio.create_task(engine.run("support", scope="acme", input=message, stop=stop, inbox=inbox))
inbox.put_nowait("Also check the March invoice.")
stop.after_step()
run = await work
run.stopped, run.reply, run.steps
```

A stopped run returns, never raises: `run.stopped` is `"now"` or `"after_step"` (`None` for a run
that ended on its own), `run.steps` are the steps it took, `run.reply` is the last text the model
produced — text streamed before `now()` included — or `""`, and it is recorded like any run. A tool
call cut off by `now()` is answered with the error `stopped by the user` and its `Step` has
`stopped=True`, so every tool call in `run.conversation` has its result and the next run continues
from it. A stop set before the run starts ends it before the first model call; one set after it
ended does nothing. A stop reaches delegated and routed runs, which end the same way at their own
next boundary. With `after_step()` pending the inbox is not read. Whatever is still in the inbox
when the run ends stays there — the engine never starts a run; you do, with those messages as its
input. A decision run never reads the inbox: its questions are one call, and a stop or an inbox
rides to the run it routes to. "Pause, please" is a message in the inbox: the model reads it at the
next boundary and wraps up in its own words.

A run that ended continues from its record, on any process, after any restart. With a refinery
attached every run that returns is recorded with its conversation: keep the run id beside what you
showed — a message's metadata, say — read the run back and pass its conversation on:

```python
last = await refinery.run("acme", run.run_id)
next = await engine.run("support", scope="acme", input="And the second charge?", history=last.conversation)
```

`refinery.run(scope, run_id)` returns the run as `engine.run` returned it, its delegated and routed
runs back in place; a child's id returns that child and the runs under it. A run id the scope does
not have is a `NotFoundError`. A deploy is a stop: on SIGTERM, ask every run in flight to stop after its step.
Each ends at its next boundary and is recorded with `stopped="after_step"`; the next message, on
whichever task takes it, reads the run back and continues:

```python
import signal

running = set()

async def answer(message):
    stop = Stop()
    running.add(stop)
    try:
        return await engine.run("support", scope="acme", input=message, stop=stop)
    finally:
        running.discard(stop)

def shutdown():
    for stop in running:
        stop.after_step()

asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, shutdown)
```

Rewinding is slicing. Cut `run.conversation` just before a `role="user"` message and pass it as the
next run's `history`; files the agent changed in a `Workspace` are yours to restore — a snapshot per
turn, say. The abandoned run stays recorded, and a review of it turns the rewind into preference
data:

```python
turns = [i for i, m in enumerate(run.conversation) if m.role == "user"]
retry = await engine.run("support", scope="acme", input="Refund only the second charge.",
                         history=run.conversation[: turns[-1]])
await refinery.review("acme", run.run_id, Review(stance="objection", correction=retry.reply))
```

### Long conversations

A model keeps nothing between calls: each one is sent the history a run is given, and a
conversation passed back whole grows with every turn — months of it fit no model's context, and
every token of it is paid for again on every turn. The engine does not keep a conversation or
choose what of it to send; you keep it, as you keep any conversation, and tell each run where to
start:

```python
conversation, start = [], 0


def window(conversation, start, turns=20, budget=150_000):
    starts = [i for i, m in enumerate(conversation) if m.role == "user" and i >= start]
    used = next((m.usage for m in reversed(conversation) if m.usage), None)
    if used and used.input + used.cache_read + used.cache_creation > budget and len(starts) > 1:
        return starts[len(starts) // 2]
    return starts[-turns] if len(starts) >= 2 * turns else start


for message in ["My invoice was charged twice.", "And the second charge?"]:
    start = window(conversation, start)
    run = await engine.run("support", scope="acme", input=message, history=conversation, start=start)
    conversation += run.added
```

`start=` is the position in `history` the run begins at; the run sends the messages from there,
and `run.added` — the input and every message after it — is what to keep. `start=0`, the default,
sends the whole history. `run.start` records where the run began, and `run.conversation` is what
the model saw: the history from `start`, then `run.added`. A routed run is given the history and
`start` its router was; a delegated run starts with no history, as always.

`window` is yours, one policy among many. It keeps 20 to 39 turns — a turn is a message the person
sent and everything that answered it — and moves its start 20 turns at a time, so between moves
the history only grows at its end. It starts at a message the person sent, so no tool call is cut
from its result; a history whose sent part holds a result without its call, or a call without its
result, is an `InputError` before any model is called. Each assistant message keeps the `usage`
its call reported, so `input + cache_read + cache_creation` is how many tokens the conversation
had reached there, and the window moves early, keeping the newer half, once the last call passed
`budget`. A window per sitting, per day, by your own count, from a summary you wrote: each is a
function of yours.

A conversation in a database is passed as itself. Anything with `read(start, stop)` that returns
the messages at those positions — every one from `start` when `stop` is `None` — is a
`HistoryLike`, and the run reads only what it sends:

```python
class Conversation:
    def __init__(self, pool, conversation_id):
        self.pool, self.id = pool, conversation_id

    async def read(self, start, stop=None):
        rows = await self.pool.fetch(
            "select body from messages where conversation = $1 and position >= $2"
            " and ($3::int is null or position < $3) order by position",
            self.id, start, stop)
        return [Message.model_validate_json(row["body"]) for row in rows]
```

A position is a message's index in the conversation from 0, and always holds the same message.

Reasoning goes back only where it was given. Some vendors bind a model's reasoning to the exact
conversation it was given in and refuse it anywhere else. Every assistant message the engine makes
from a reply carries `prefix`, a short hash of everything sent before it — the model, as its adapter
names it, the system prompt, the tools, the boot files and the messages — and `usage`, what the call
that wrote it reported. A message's `reasoning` is sent again only when what comes before it is
still the same; otherwise the message goes without it, its text and calls as they were, and
`run.messages` shows it as sent. So a history you cut or edited, a changed `context=`, `task=` or
`tools=`, another agent, another model and a replay never send reasoning a vendor would refuse.
`prefix` and `usage` are the engine's record; no model sees them.

### Decision agents

A decision model is a discriminative model: given a state and questions whose answers you define,
it returns the probability of each answer — as a generation model is generative, producing text.
What you use the answers for is yours: classify, score, check, extract among candidates, annotate,
grade, route. A decision agent is a directory of questions, asked together in one call:

```
agents/
  triage/
    AGENT.md
    questions/urgent.md
    questions/severity.md
```

```markdown
---
type: check
yes: The service is down or data is at risk
no: A question, a request, or a problem with a workaround
---
Does the message need a person right now?
```

```markdown
---
type: score
levels: [Can wait days, Today, Within the hour, Outage]
---
How severe is the problem?
```

A `check` may say with `yes:` and `no:` what each answer means; a `score` lists `levels:` in order;
a `choice` lists `options:`, each key with what it means. In frontmatter only `true` and `false` are
booleans; `yes`, `no`, `on` and `off` are words, as a key or as a value.

What an option means, a level, `yes` and `no` may each be a YAML object or list instead of a string,
and an option left empty means `None`. Instructions with structure go in the frontmatter as
`instructions:`, an object or a list, and replace the body; a question with instructions in both, or
in neither, is a `ConfigurationError`:

```markdown
---
type: choice
instructions:
  question: Which team should handle this message?
  focus: The main request, not every topic the message mentions
options:
  billing:
    what: Charges, invoices, refunds
    not_for: Delivery or account access
    examples: [I was charged twice, My refund never came]
  orders:
    what: Delivery, cancellation, returns
    examples: [My package never arrived]
  other:
---
```

Run it with a state: a string, an object or a list.

```python
engine = Engine(root=Path("agents"), decision=dm)

run = await engine.run("triage", scope="acme", input={"message": text, "plan": "enterprise"})
run.answers["urgent"].probability     # 0.91
run.answers["severity"].score         # 3.0
run.answers["severity"].confidence    # 0.87
```

Every question is answered in the same call, in parallel and in isolation: adding a question
changes neither the others' answers nor, noticeably, the latency. An engine with only a decision
model needs no tool bodies; the tree's generation agents are the ones it cannot run.

Some answer spaces are known only when the run starts: the children of the node a walk chose last,
the candidates a search returned, the skills an agent has today. Pass those questions as
`questions=`, ids to `Choice`, `Score` or `Check` — or anything with their fields — and they are
asked with the folder's, in the same call. The record keeps every question asked, and
`run.runtime` names the ids that came at run time:

```python
products = Choice(
    instructions="Which product is the message about?",
    options={"desk-120": {"what": "Standing desk, 120 cm"}, "desk-160": {"what": "Standing desk, 160 cm"}},
)
run = await engine.run("triage", scope="acme", input=text, questions={"product": products})
run.answers["product"].choice, run.answers["urgent"].probability
run.runtime                           # ["product"]
```

A runtime question holds what a question file holds — a choice needs two options, a score two
levels, and structure is JSON — and one that does not, an id the run already asks, or
`questions=` given to a generation agent is an `InputError`. A decision agent may have no
`questions/` at all and ask only what each run brings; a run with nothing to ask is an
`InputError`.

A decision agent's task adds questions, as a generation agent's task adds instructions:
`tasks/<task>/` holds question files as `questions/` does, and `task=` asks them beside
`questions/`, which every run asks, and beside the runtime ones. One call asks an id once, so a
task repeats no id of `questions/` — a `ConfigurationError` when the tree loads — while two tasks,
never asked together, may share one. An agent may keep all its questions in tasks, and a run of it
that names none and brings none has nothing to ask. A task the agent does not have is a
`NotFoundError`, as an agent the tree does not have is, for either kind of agent:

```
agents/
  grader/
    AGENT.md
    questions/grounded.md
    tasks/tone/polite.md
    tasks/tone/register.md
```

```python
run = await engine.run("grader", scope="acme", input={"reply": "Thanks, all sorted!"}, task="tone")
run.task, sorted(run.questions)       # "tone", ["grounded", "polite", "register"]
```

`run.fingerprint.questions` is a short sha256 over the questions the run asked — the folder's, the
task's and the runtime ones alike — as JSON ordered by question id, each question's fields as they
were asked and its options and levels in their order, since the order can change which answer a
model favours. Two runs that asked the same questions share it; a changed word, option or order
changes it. A generation run's is `None`. With `fingerprint.agent` for the folder, it tells two
versions of a decision apart even when the questions came at run time.

The state is text unless the decision model reads more. An adapter that carries `media_types` —
patterns such as `{"image/*", "application/pdf"}`, as a generation adapter's — takes a list of parts
with that media as the state, parts as they are, and the record keeps them. A list of parts made
only of `Text` is joined with newlines into the state, as it is for an adapter without
`media_types`, which reads text only. A part the adapter does not declare is an `InputError` before
the model is called, naming what it takes (`engine: decision agent 'triage' takes text only; got
image/png`).

Everything beyond the answers — weighing several, confirming with a person below a threshold,
acting — is your code over the returned `Decision`. The library gives you the answers; it does
not make the call.

### Nesting

`agents:` in a generation agent's frontmatter gives it a `Delegate` tool listing those agents by
name, description, kind and tasks. The model calls it with an agent and an input; the child runs
under the same scope and its reply — text for a generation agent; for a decision agent its answers,
and the routed run's reply when it routed, as `engine.tool` returns them — comes back as the tool
result. The child's record is attached to the parent's `children`, and the
step that called it carries the child's `run_id`. A decision agent used this way is how a
generation agent asks a fast, calibrated question mid-task: is this destructive, which skill
applies, is the user done.

The model can also name one of the target's `task`s, of either kind, and give a decision agent
`questions` it composes itself — ids to questions of the three types, written as JSON, structure
included — asked beside the agent's own in the same call. A question that occurs to the model
mid-task is answered with probabilities instead of guessed at, and a decision agent with no
questions of its own is worth delegating to. The model sends:

```json
{"agent": "triage", "input": "The March invoice shows the same $49 charge twice.",
 "questions": {"refund": {"type": "check", "instructions": "Does the customer ask for money back?"},
               "cause": {"type": "choice", "instructions": "What caused the second charge?",
                         "options": {"duplicate": "The same payment taken twice",
                                     "plan_change": "A prorated change of plan", "other": null}}}}
```

```python
run = await engine.run("support", scope="acme", input="I was charged twice in March. Refund one.")
child = run.children[0]
child.runtime                             # ["refund", "cause"]
child.answers["refund"].probability       # 0.91
child.answers["cause"].choice             # "duplicate"
```

The child records its task and its runtime questions as any run does, and the step keeps what the
model sent. `Delegate` offers `task` only when a target has tasks and `questions` only when one is
a decision agent, and its schema describes the question object once, for all three types. A task
the target does not have, questions for a generation agent, a question that does not fit, an id the
target already asks, or a decision run with nothing to ask reaches the model as a tool error it can
read and act on, and the run goes on.

Depth is capped (`max_depth`, default 5) and a cycle is refused; both are reported to the model as
tool errors, and the run continues.

### Both kinds together

The two kinds call each other. A generation agent asks a decision agent through `Delegate`, as in
[Nesting](#nesting), and a decision agent can hand its run to another agent by one of its answers:
a route. One choice question in `questions/`, one table from its options to agents, an optional
confidence floor:

```
agents/
  frontdesk/    kind: decision   route: {by: intent, to: {billing: billing, technical: support}}
  support/      kind: generation agents: [triage]
  billing/      kind: generation
  triage/       kind: decision   questions: urgent (check), severity (score)
```

```markdown
---
name: frontdesk
kind: decision
description: Routes an incoming message
route:
  by: intent
  to: {billing: billing, technical: support}
  min_confidence: 0.6
---
```

```markdown
---
type: choice
options:
  billing: Payments, refunds, invoices
  technical: The product misbehaves
  other: Anything else
---
Which department should handle this message?
```

When the chosen option maps to an agent and confidence clears the floor, that agent runs with
everything the run was given — the same input, parts included, `history`, `context`, `expect`,
`tools`, `workspace`, `on_event`, `stop`, `inbox` and `tags` — and the routed run is attached:

```python
engine = Engine(root, generation=llm, decision=dm, tools=tools)

run = await engine.run("frontdesk", scope="acme", input=message, history=history)
run.answers["intent"].choice            # "technical"
run.routed_to.agent                     # "support" — the generation agent that answered
run.routed_to.children[0].agent         # "triage" — the decision agent support consulted mid-task
```

A route is by a question of the router's `questions/`, which every run asks, so neither a task
nor a runtime question can drive it; a router's task and runtime questions are its own, asked
beside `intent` and not passed on. A route entry names an agent, or an agent and one of its tasks,
which the routed run is given — a generation agent's instructions or a decision agent's questions:

```yaml
route:
  by: intent
  to:
    billing: billing
    technical: {agent: support, task: triage}
```

The task must be one its agent has, and a decision agent a route reaches must have something to
ask — `questions/`, or questions in the task the entry names; either is a `ConfigurationError` when
the tree loads, since that tree cannot work. A decision agent reached through `Delegate` needs
neither, since the model can name a task or bring the questions.

One message, four model calls, one record. With only `generation=` the same tree runs its
generation agents and refuses the decision ones; with only `decision=` the reverse, and a decision
agent whose route reaches a generation agent is refused before its questions are asked. Nothing
about an agent changes when the other kind of model is absent — only which agents you can run.

### Workspace

> `Workspace` gives the model a real shell, and **a shell is not confined by its directory**. Without
> a sandbox it can do anything the user running the service can do: read every other tenant's
> workspace, read the service's configuration and any secret on disk, reach the database and the
> network, delete things. A document an agent reads can tell it to try. Run every command inside a
> sandbox with `sandbox=` — a container per command, as below, is the default to reach for — and keep
> the agent directory outside the workspace.

A workspace is a directory where an agent writes code, runs it, reads what happened and tries again —
the loop a coding assistant runs in a developer's terminal. An agent opts in with one line in its
`AGENT.md`, and a run supplies the directory and the sandbox its commands run in:

```
agents/coder/AGENT.md
---
name: coder
kind: generation
description: Writes and runs code to finish a task.
workspace: true
---
```

```python
from rfnry_engine import Workspace

workspace = Workspace("/srv/tenants/acme/work", timeout=600,
    env={"PATH": "/usr/bin:/bin", "DOCKER_CONFIG": "/etc/acme/docker"},
    sandbox=["docker", "run", "--rm", "-i", "--init", "--network", "none",
             "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only", "--tmpfs", "/tmp",
             "--memory", "1g", "--cpus", "1", "--pids-limit", "256", "--user", "1000:1000",
             "-e", "HOME=/srv/tenants/acme/work",
             "-v", "/srv/tenants/acme/work:/srv/tenants/acme/work", "-w", "/srv/tenants/acme/work",
             "acme/workspace:1", "timeout", "-s", "KILL", "600"])

run = await engine.run("coder", scope="acme", input="make the failing test pass", workspace=workspace)
```

Each command gets a fresh container from an image you build — a world of its own: no network, no
other tenant, nothing of the service — and `--rm` removes it when the command ends. The model sees
none of this: it has the same tools, and its command runs inside.

Every workspace says whether its shell is sandboxed. `sandbox=` names the command each shell call
runs inside — this container, or bubblewrap below; `Workspace(root, unsandboxed=True)` says out loud
that this machine holds nothing the model must not reach — a developer's laptop. A workspace with
neither refuses to start.

The agent then has three more tools. `Shell(command)` runs a command and returns `exit_code`,
`stdout`, `stderr` and `timed_out`. `Write(path, content)` writes a file exactly, and
`Edit(path, old, new)` replaces one exact occurrence — helpers that send only what changes and leave
a clean step in the record; the shell can do both too. `Load` takes `where="workspace"` to hand
the model a file from the directory, so an agent looks at the chart it just plotted. Each command starts fresh in the directory,
so a `cd` or a variable does not carry over, and the tool's description tells the model so, along
with its shell and its limits.

`Workspace(root, sandbox= or unsandboxed=True, shell=, timeout=, output_bytes=, env=)`:

| argument | default | what it does |
| --- | --- | --- |
| `sandbox` | none | the command each `Shell` call runs inside — a container, bubblewrap — before the shell |
| `unsandboxed` | `False` | `True` runs the shell on the machine itself, reaching whatever the service's user can |
| `shell` | `["bash", "-c"]` | the shell the command is given to, last; the model is told its name. Without a sandbox it must be on `PATH` |
| `timeout` | `120` seconds | the limit on each `Shell`, `Write` and `Edit` call; a command still running is sent SIGTERM, then killed a second later with everything it started, and the call returns what it printed with `timed_out: true` |
| `output_bytes` | `30_000` | the cap on `stdout` and on `stderr`, each keeping its first and last half |
| `env` | `PATH`, `HOME` (the directory), `LANG` | the only environment the sandbox's command sees, so the service's own keys never reach it |

The workspace's `timeout` replaces `Engine(tool_timeout=)` for its three tools, so a ten-minute
build and a thirty-second API call live in the same engine. An unsandboxed workspace without bash on
`PATH` is a `ConfigurationError` that says so; pass `shell=["sh", "-c"]` on a system that has only a
POSIX shell, such as Alpine.

`Write`, `Edit` and `Load` stay inside the directory: the engine runs them itself and refuses a path
that leaves it. `Shell` reaches what the process running it can reach, so the real boundary is
`sandbox=` or the machine.

In the container above, the image is the toolchain: a Dockerfile of yours with what your agents
need — Python, Go, a C++ compiler. The directory is mounted at the same path, so `Write` and the
shell see the same files, and `--user` is the user that owns it, so a file written on either side
stays editable on both. `env=` is the environment of the `docker` client, not of the command. The
client reads its configuration from `$HOME/.docker`, and the default `HOME` is the workspace: a
command that wrote `.docker/config.json` there put `HTTP_PROXY` into the next command's container.
So `DOCKER_CONFIG` points outside the workspace, and the command's own variables go in with `-e`.
Podman reads `$HOME/.config/containers/containers.conf` the same way, and that file can add mounts.

The client passes the engine's SIGTERM on to the container, so the timeout and `stop.now()` stop
it. SIGKILL stops only the client, so a command that ignores SIGTERM would keep its container
running; `timeout -s KILL` inside ends it at the same limit. Each command takes about a quarter of
a second to start.

The caps bound one command; how many run at once is yours to cap — a semaphore around runs — and
`--cgroup-parent` can put every workspace container under one slice, where a limit on the slice
bounds them all. A container shares the host's kernel; for code from many tenants, the gVisor
runtime (`--runtime runsc`) is the stronger boundary. The Docker socket is root on the host, so a
service that can reach it is as good as root; rootless Docker, or Podman — `podman run` takes the
same flags — avoids that.

A container kept for a tenant or a conversation keeps what commands installed between them: start
it with `--init`, and enter it with `docker exec -i -w /srv/tenants/acme/work acme-work timeout -s KILL 600`.
The engine's signals do not reach inside `docker exec`, so that time limit is the only one that
stops a command, `stop.now()` leaves a command running until then, and starting and removing the
container is yours.

Where Docker is not an option, on Linux, bubblewrap builds each command a world of its own from the
host: the system's programs read-only, this tenant's directory writable, an empty `/tmp`, and
nothing else — no other tenant, no service configuration, no network — and it kills every process
when the command ends:

```python
sandboxed = Workspace("/srv/tenants/acme/work", sandbox=[
    "bwrap", "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
    "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev",
    "--tmpfs", "/tmp", "--bind", "/srv/tenants/acme/work", "/srv/tenants/acme/work",
    "--chdir", "/srv/tenants/acme/work", "--unshare-all", "--die-with-parent", "--new-session",
])
```

Binding the whole system read-only (`--ro-bind / /`) is not enough: a command could still read
every other tenant's directory and every secret on disk. Mount only what commands need — the
recipe above suits a system whose programs live under `/usr`; add a read-only `/etc` path when a tool
needs one — and one workspace per tenant, each with its own sandbox. bubblewrap limits nothing: one
command can take all the memory, CPU or processes of the machine from every other tenant. Where the
service's user has a systemd user session,
`systemd-run --user --scope -p MemoryMax=1G -p CPUQuota=100% -p TasksMax=256 --quiet` before `bwrap`
runs each command in a scope with those limits; `env=` then needs `XDG_RUNTIME_DIR` too, which
`systemd-run` uses to find the session.

Without a sandbox, the kill reaches every process the command started except one that leaves its
process group with `setsid`; the process namespace of a container or of bubblewrap closes that gap.

An agent without `workspace: true` never sees these tools, even when the run has a workspace; an
agent with it, run without one, is a `ConfigurationError`, and so is a decision agent that declares
it. Delegated and routed agents receive the run's workspace on the same terms. `stop.now()` stops a
running command the same way the timeout does. You own the directory and its lifetime: a temporary
one per run, or one kept for a project.

### Approving tool calls

By default an agent calls any tool it has. Pass `approve` and every tool call waits for your answer
first — the shell, a `Write`, a refund, a `Delegate`. It receives the agent's name and the call, and
returns `True` to run it or `False` to refuse:

```python
async def approve(agent, call):
    if call.name in ("Write", "Edit"):
        return True
    return await ask_a_person(agent, call)

run = await engine.run("coder", scope="acme", input="make the failing test pass",
                       workspace=workspace, approve=approve)
```

Where the person is — a thread, a dashboard, a chat app — is yours; the engine only waits. "Allow
everything" is no `approve`; "ask every time" is one that always asks; "allow edits, ask for the
rest" is the function above; "always allow this command" is a set the function keeps. A refused
call does not run: the model reads `NotApproved: Shell was not approved` and carries on, and a
reason reaches it through the `inbox`. Each step records `approved` — `True`, `False`, or `None`
when nobody was asked — and the exported rows keep it, so a person's refusals become labels.

Time spent waiting is not a tool's time: the deadline starts when the call runs. `stop.now()` ends
a wait. Delegated and routed runs are asked through the same function, with their own agent's name;
`engine.tool()` and `EngineTools` take `approve=` for the runs they start. An `approve` that returns
anything but `True` or `False` is a `ConfigurationError`. The run waits in memory, so an approval
that takes hours is better refused now and continued later from `run.conversation`.

### Refining

The refinery records and exports what your agents did, with every label and its source. What you do
with it — evaluate, improve prompts and rules, train or distill models, audit — is yours.
[`evaluation.md`](evaluation.md) is a guide to doing it: capturing the right answer, the correct
label and what really happened, and the three levels of improvement they feed.

Attach a refinery and every run that returns is recorded; a run that raises is not. A recording
that fails is logged as a warning and never fails the run. `refinery.run(scope, run_id)` reads a
recorded run back, its delegated and routed runs in place, to continue it anywhere — see
[Running an agent](#running-an-agent).

```python
refinery = Refinery(PostgresStorage(pool))
await refinery.initialize()
engine = Engine(root, generation=llm, decision=dm, tools=tools, refinery=refinery)
```

Labels come from three places. A person reviews a run — the button under a reply, the correction a
supervisor types, the right answer to a question that was decided wrong:

```python
await refinery.review("acme", run.run_id, Review(stance="objection", correction="Refund the second charge in full."))
await refinery.review("acme", decision.run_id, Review(correction={"intent": "technical"}))
```

The world reports whether the action worked:

```python
await refinery.outcome("acme", run.run_id, Outcome(status="failed", detail="Customer escalated."))
```

Each call stores a new label. An app that labels from events more than one process hears — a staff
reply every replica of a backend receives — or that retries after a crash passes `label_id=`, an id
derived from the event such as the message id:
`refinery.review("acme", run.run_id, review, label_id=message.id)`. The first call stores the label
under it; every later call with that id on the same run, racing or not, returns that label and
writes nothing, and one on another run is an `InputError`.

And models label what people did not. A **judge** is a decision agent in your tree whose questions
are about an episode; it sees the input, the reply and every tool call with its result, with each image,
audio or document replaced by a placeholder naming it — `[image: image/jpeg]`, `[document: application/pdf, "policy.pdf"]` —
while the recorded run and the exported datasets keep the media. A judge sees neither the system
prompt nor `run.boot`, so a rule a booted agent never had to `Read` is not in front of it; ask about
them in the question when they matter:

```
agents/judge/
  AGENT.md                 kind: decision
  questions/grounded.md    type: check   Is every claim in the reply supported by a tool result?
  questions/rule.md        type: check   Does the reply apply the refund rule as written?
  questions/done.md        type: score   levels: [Not addressed, Partly, Fully, Fully and confirmed]
```

A **teacher** is a generation agent that answers a decision agent's questions with a rationale, so
a decision made without ground truth still gets one; it reads each question as the run asked it,
structure written as compact JSON. Both run when you say so, over runs they have not labeled yet:

```python
labeled = await refinery.label("acme", engine, judge="judge", teacher="teacher")
labeled.judged, labeled.taught, labeled.errors
```

A judge or teacher the tree does not have is a `NotFoundError`, and one of the wrong kind a
`ConfigurationError`, before any run is labeled. A run that could not be labeled — the model
failed, or the teacher answered a choice with no option, a score with no level index, a check with
anything but true or false — is named in `labeled.errors` with why, and stays unlabeled for the
next call. Every label carries its source. A
judge's score is exported as a judge's score, never as a person's.

Export writes datasets to any object with a `put(key, bytes)`:

```python
class Folder:
    def __init__(self, root): self.root = Path(root)
    async def put(self, key, data):
        (self.root / key).parent.mkdir(parents=True, exist_ok=True)
        (self.root / key).write_bytes(data)

exported = await refinery.export("acme", Folder("datasets"), since=last_watermark)
exported.keys, exported.rows, exported.watermark
```

Keys are `{scope}/{agent}/{dataset}/{timestamp}.jsonl`, so repeated exports add
files and never overwrite. `since` selects every run recorded or labeled after it, written in full
with all its labels. A run is recorded when it ends, so a long run still working at one export is in
the next, whenever it started; the watermark covers both, so a review of a run you already exported
reaches the next export. Rows are keyed by `run_id`: when a run appears again, its latest row
supersedes the earlier ones — dedupe by `run_id`, keep the latest. Because of that you may pass a
`since` a little before the last watermark: a run is stamped just before it is written, and
services recording with their own clocks can disagree by a little.

Every row carries `"holdout": true|false` — the rows kept aside so any measurement on them is
honest. It is derived from a hash of the run id (`holdout_percent=20` by default), so a run's flag
never changes between exports; `is_holdout(run_id)` gives the same answer in your own code. Three
datasets:

| dataset | one row per | what it holds |
| --- | --- | --- |
| `generation` | generation run | task, context, tags, model, fingerprint, system, boot, start, messages, tools, conversation, steps, reply, stopped, usage, every label |
| `preference` | human correction | tags, fingerprint, prompt (system, boot, messages), chosen (the correction), rejected (the reply) |
| `decision` | question of a decision run | task, tags, model, fingerprint, state, question, answer, probabilities, confidence, truth, outcome |

Every row carries its run's `tags`, `{}` for a run that had none or was recorded before tags were
kept, and its `fingerprint`, `{"agent": …, "tools": …, "questions": …}` or `null` for a run recorded
before fingerprints were kept — `questions` is `null` for a generation run and for a decision run
recorded before it was kept; group or filter the rows by them in your own code.

The first two are a generation agent's episodes and the corrections people made to them. A
generation row's `tools` is what the model was offered — `{name, description, input_schema}` each —
`boot` the parts `boot: all` sent ahead of `messages`, `[]` without it, and `conversation` is
`run.conversation` in order: history, input, assistant turns with their `tool_calls`
(`{id, name, args}`), tool results with `is_error`, the final reply. Together they fit
standard fine-tuning and eval formats with a few lines of mapping; the library writes no vendor's
format. A run recorded before `tools` or `boot` was kept exports it as `[]`. Parts
are written inline, their bytes as base64, so a row is self-contained; strip the media from a row
you do not want it in. The third
is a decision agent's answers beside what turned out true: a calibrated model is one whose 0.8 was
right 80% of the time, and this is the table that says whether it was:

```json
{"run_id": "…", "holdout": false, "agent": "frontdesk", "tags": {"arm": "b"}, "question_id": "intent", "answer": "billing",
 "probabilities": {"billing": 0.8, "technical": 0.15, "other": 0.05}, "confidence": 0.7,
 "truth": "technical", "truth_source": "human", "outcome": "failed"}
```

`refinery.conflicts(scope)` lists runs that carry two different human corrections — a dataset can
hold either, not both, and the library reports rather than guesses.

Replay measures a candidate — a new folder, a new model, any `Engine` you built — on what your
recorded runs were given. Each pair holds the recorded run, its labels and the candidate's run:

```python
candidate = Engine(root, generation=llm, decision=dm, tools=tools)

async def read_only(agent, call):
    return call.name in {"Read", "Grep", "Glob", "LS", "Skill", "Load", "Delegate", "lookup_order"}

async for pair in refinery.replay("acme", candidate, agent="support", holdout=True, labeled=True,
                                  approve=read_only):
    pair.recorded.reply, pair.labels, pair.candidate.reply if pair.candidate else pair.error
```

> **Replay runs the candidate's tools for real.** A tool that refunds refunds again, a tool that
> sends a message sends it, a `Shell` runs. Pass an `approve` that refuses every tool that changes
> anything — `read_only` above lets through the builtins that read the agent's folder, `Delegate`,
> whose runs it is asked about too, and a lookup — or build the candidate with bodies that only
> read. A refused call reaches the model as `NotApproved`, as in any run.

It replays the top-level runs of `agent` in the scope, oldest first. `holdout=True` keeps the
held-out ones, by the same hash and `holdout_percent` as `export`; `labeled=True` those a person
reviewed or the world reported on; `tags=` those whose tags contain these, as `delete_runs`
matches; `fingerprint=` the runs one version produced — the `run.fingerprint` of any of them, its
`agent`, `tools` and `questions` all equal; `limit=` the first so many. Each is
`candidate.run(agent, scope, …, record=False)` with the recorded input — parts as parts — the
history sent before it, the task and the `context`, so the candidate sees what the recorded run
saw. A decision run replays its state and its task, and asks its runtime questions again as they
were recorded while the questions of its folder come from the candidate's folder — a runtime id the
candidate's folder now asks fails that pair; when its route reached a generation run, that run's
input, history and context go too. What a record does not keep is yours to pass
again — `expect=`, `approve=`, and the `tools=` and `workspace=` the runs were given. The candidate
delegates and routes as it does, so a run with children replays as one run. A run recorded before
`context` was kept replays without it.

`repeat=3` replays each run three times: each attempt is a run of its own and a pair of its own,
`pair.attempt` counts them from 0, and the three come one after another; `limit=` still counts
recorded runs. A model samples, so the same candidate on the same input can answer differently, and
the attempts show how much. `concurrency=4` keeps up to four replays going, attempts included, and
while you read a pair at most three more run; with the default of one, a replay starts when you ask
for its pair. Pairs come in recorded order either way. A candidate that fails yields its pair with
`error` — `"ModelError: …"` — and `candidate=None`, and the replay goes on, the run's other
attempts included. Nothing is recorded, no label is written, and nothing is graded: the pair holds
both runs with their models and fingerprints, and what you compare is yours.
[`evaluation.md`](evaluation.md) grades the replies with a decision agent.

### Agents kept in a repository

The engine reads a folder; a new version of the agents is a new `Engine`. Keep the tree in a git
repository — people who do not write code edit it in the host's web UI, and every change is a
reviewed pull request — check each commit out into a folder of its own, build an engine from it,
and swap which engine new runs use. A run in flight finishes on the engine it started with. Rolling
back is loading an older commit, and since a commit's folder never changes, a half-written
checkout is never live. Agents read their files as they work, so remove an old folder only once
its runs have finished. The checkout is yours — `git worktree add`, a clone, an archive:

```python
def load(commit):
    return commit, Engine(checkout(commit), generation=llm, decision=dm, tools=tools, refinery=refinery)

live = load("4f2c9e1")
commit, engine = live
run = await engine.run("support", scope="acme", input=message, tags={"commit": commit})

live = load("a81d07b")

arms = {"a": load("4f2c9e1"), "b": load("a81d07b")}
arm = "ab"[hashlib.sha256(b"user-42").digest()[0] % 2]
commit, engine = arms[arm]
run = await engine.run("support", scope="acme", input=message, tags={"commit": commit, "arm": arm})
```

Tag each run with the commit it came from and the record, its delegated and routed runs, and every
exported row say which version of the tree answered. The tag names a version; the fingerprint proves
it. A tag says what you meant to load, and a checkout with an uncommitted edit carries the same tag;
`run.fingerprint.agent` is taken from the files the engine read, so two runs are the same agent when
their fingerprints match, whatever their tags say. The same tag is an A/B test: load two commits
side by side, send each user to one by a stable hash of their id, tag the run with its arm, and
group the exported rows by `tags["arm"]` to compare the labels and outcomes of whole versions of
the tree.

## Tools

One concept, one word. A **tool** is a name, a description, an input schema and a body. A
**`Tools`** is a set of them: what a model receives, and what your code can call the same way.

```python
class Tools:
    def definitions(self) -> list[dict]                 # [{name, description, input_schema}]
    async def dispatch(self, name: str, args: dict) -> str | dict | list[Part]
    def __add__(self, other) -> Tools                    # union; a duplicate name is refused
    def only(self, *names) -> Tools
    def without(self, *names) -> Tools
    names: list[str]                                     # every tool's name, in order
```

An agent's tools are declared in its directory. `Read Grep Glob LS Skill Load` come with every generation agent,
`Delegate` comes with `agents:`; everything else is a file under `tools/`:

```markdown
---
name: lookup_order
description: Fetch an order by id
input:
  order_id: {type: string, description: The order id, required: true}
---
```

```markdown
---
name: fetch_invoice
---
```

The first declares the contract the model sees and leaves the body to code. The second adopts a
code tool as it is — description and schema included. Bodies and code tools go on the engine, one
list for the whole tree. A body returns a string, a dict (the model reads it as JSON) or a list of
parts, which reach the model as the tool result — a PDF the model reads itself, an image it looks
at. `Load` already does that for files in the agent's directory; a tool does it for what lives
elsewhere, such as your own store:

```python
class InvoiceArgs(BaseModel):
    invoice_id: str

async def fetch_invoice(args):
    data = await invoices.get(args["invoice_id"])
    return [Document(media_type="application/pdf", name=f"{args['invoice_id']}.pdf", data=data)]

tools = [
    Tool("lookup_order", body=lookup_order),
    Tool("fetch_invoice", "Fetch an invoice PDF by id", body=fetch_invoice, args=InvoiceArgs),
]
engine = Engine(root, generation=llm, tools=tools)
```

A tool the engine was given that no agent declares is a `ConfigurationError`; so is a declaration
with no body, when the engine has a generation model to run it. So an agent's whole tool surface
is readable from its directory, and the engine cannot hand a model something the tree does not
show — except the `Tools` a caller passes to `run(tools=)`, which ride that one run.

Every tool call inside a run has a deadline, `Engine(tool_timeout=30.0)` by default and `None` to
turn it off. When it passes, the model gets the error `timed out after 30s` and the run continues.
Bodies are async and the deadline cancels them, so a body that runs blocking work in a thread must
put its own limit on that work: cancelling the await does not stop the thread. A workspace's
`Shell`, `Write` and `Edit` keep the workspace's own `timeout` instead. `Delegate` has no
deadline. The child run it starts is bounded by its own tool calls and `max_steps`.

The builtins are bounded whatever the deadline. `Read` returns UTF-8 text and refuses anything else
with a pointer to `Load`; `offset` and `limit` select lines. `Load` takes any file up to 20 MB and
hands it over as an image, audio or a document part, whichever its type says. The type comes from
the file's name, and for PNG, JPEG, GIF, WebP and PDF the first bytes must agree — a `.png` that is
not one is refused rather than sent — while a file with no known name is typed by its bytes or sent
as `application/octet-stream`. What the model can take is the adapter's `media_types`, and past
that the vendor's. `Grep` searches text only and skips the rest; `Glob` and `LS` list every file.
Neither `Read` nor `Grep` reads a file over 5 MB. `Grep` stops after scanning 50 MB and
notes the cut. A search that runs past 10 s in one call is an `InputError` the model can act on. `Glob` and
`Grep` walk at most 100,000 entries and note the cut. A tool that reads the
agent's directory itself resolves paths with `Jail`, which refuses one that leaves the directory,
as the builtins do.

Tools that belong to a tenant go on the call that names the tenant:

```python
run = await engine.run("support", scope="acme", input=text,
                       tools=MemoryTools(memory, corpus_id="acme"))
```

That is how rfnry-memory mounts: its `MemoryTools` is a `Tools`, and neither library imports the
other. The set an agent runs with is itself a first-class object, so you can drive it without a
model — from a test, or from a loop of your own:

```python
surface = EngineTools(engine, agent="support", scope="acme")
surface.definitions()                                        # what the model would see
await surface.dispatch("Read", {"path": "rules/refunds.md"})
await surface.dispatch("Delegate", {"agent": "triage", "input": "It is down"})

combined = MemoryTools(memory, corpus_id="acme") + surface.only("Read", "Grep")
```

A `Delegate` dispatched this way runs the child as its own top-level run, with the scope, `tools=`,
`workspace=` and `tags=` given to `EngineTools` and the `task` and `questions` it is called with —
what a `Delegate` inside a run would give it. A refusal raises here, where inside a run the model
reads it.

An agent is itself a tool. `engine.tool(agent, scope=...)` gives any runtime — a live voice model,
a loop of your own — one tool that runs the agent: named for the agent and described by its
`AGENT.md` unless you say otherwise, taking `{"input": str}` — and, as `Delegate` does, a `task`
when the agent has tasks and `questions` when it is a decision agent — returning a generation
agent's reply or a decision agent's answers as a dict:

```python
consult = engine.tool("billing", scope="acme", name="consult", description="Ask the billing desk.",
                      context="Acme is on the enterprise plan.")
reply = await consult.dispatch("consult", {"input": "Was I charged twice?"})
triaged = await engine.tool("triage", scope="acme").dispatch("triage", {"input": "It is down"})
triaged["answers"]["urgent"]["probability"]                # 0.91
surface = consult + MemoryTools(memory, corpus_id="acme")
```

A generation agent's tool returns its reply. A decision agent's returns `{"answers": {…}}`, and when it
routed, also `"routed_to": {"agent": …, "reply": …}` — the routed run's result, nested the same way
when the target is itself a decision agent — so a caller that consults a routing agent gets the
answer it routed to, not only the answers.

It is a `Tools` of one, so it adds to any other. Each call is a top-level run under the scope it
was made with, recorded like any run; `context=`, `tools=`, `workspace=` and `tags=` ride every call.
Mounted inside another run with `run(tools=)` it is a tool like any other, under `tool_timeout`,
and its run is not the caller's child — to nest with lineage, name the agent in `agents:`. A loop
through it is still refused.

## Storage

The refinery is the only part of the library that stores anything. Two storages ship: `InMemoryStorage()`
for tests and small things, `PostgresStorage(pool)` for everything else, taking an asyncpg pool you
built. `initialize()` creates two tables, `engine_runs` and `engine_labels`, and is safe to run
again. A run's row keeps `at`, when the run started, and `recorded`, when the refinery stored it —
one stamp for a run and every child run under it — and `runs(since=)` reads `recorded`. The tables
are prefixed `engine_` so one database can also hold rfnry-memory's and rfnry-thread's:

```python
pool = await asyncpg.create_pool(dsn)
memory = Memory(rfnry_memory.PostgresStorage(pool), embedding=emb)
refinery = Refinery(rfnry_engine.PostgresStorage(pool))
```

A storage is any object with the methods below. `scope` comes first in every one, and each reads
and writes that scope only. `refinery.storage` is the storage the refinery was given, to read its
rows back yourself, and `refinery.record(run)` stores a run and every child run under it — what an
engine with a refinery does after each run not made with `record=False`:

```python
class Storage(Protocol):
    async def initialize(self) -> None
    async def put_run(self, scope: str, row: RunRow) -> None
    async def get_run(self, scope: str, run_id: str) -> RunRow | None
    async def runs(self, scope: str, *, run_ids: list[str] | None = None, kind: str | None = None,
                   since: datetime | None = None, unlabeled_by: str | None = None,
                   limit: int | None = 1000) -> list[RunRow]       # newest first
    async def put_label(self, scope: str, row: LabelRow) -> LabelRow  # the row under its id; never replaces one
    async def labels(self, scope: str, *, run_ids: list[str] | None = None,
                     source: str | None = None, since: datetime | None = None) -> list[LabelRow]
    async def run_tree(self, scope: str, *, run_ids: list[str] | None = None,
                       tags: dict[str, str] | None = None) -> list[RunRow]  # oldest first
    async def delete_runs(self, scope: str, *, run_ids: list[str] | None = None,
                          tags: dict[str, str] | None = None) -> int        # the rows it removed
    async def delete_scope(self, scope: str) -> int                 # the rows it removed
    async def count_scope(self, scope: str) -> int

class Refinery:
    storage: Storage
    async def record(self, run: Run) -> None

class RunRow(BaseModel):   scope, run_id, parent_run_id, agent, kind, task, at, recorded, duration_ms, model, body
class LabelRow(BaseModel): scope, label_id, run_id, source, at, body
class Exported(BaseModel): keys, rows, watermark
class Labeled(BaseModel):  judged, taught, errors
class Conflict(BaseModel): run_id, corrections
```

A row's `body` is the run's record, or the label, as JSON, with a run's children and routed run
left out: each is a row of its own, joined by `parent_run_id`. `runs` returns the newest first and
`labels` the oldest first; `unlabeled_by="judge"` keeps the runs no label from that source names.
`run_tree` returns the runs a selector matches — `run_ids`, or `tags` by containment — and every run
under them, oldest first; `refinery.run` reads a run back from it. `delete_runs` removes the same
runs and every label on them in one transaction, for `refinery.delete_runs`, which checks the
selectors before it calls.
`Exported.rows` counts the rows written per dataset; `Conflict.corrections` lists each correction
as `(label_id, at, text)`, oldest first.

## Tenancy

`scope` is the tenant. It is required on every run and every refinery verb, it is the first column
of every table, and no read accepts its absence. Tools pinned to a tenant are passed on the call.
`refinery.delete_scope(scope)` removes everything; `refinery.verify_absent(scope)` proves it.

Inside a tenant, erase by tag or by run id. A support desk serves every customer in one scope; tag
each run with its customer, and when one asks to be forgotten:

```python
run = await engine.run("support", scope="acme", input=message, tags={"customer": "c-81"})
deleted = await refinery.delete_runs("acme", tags={"customer": "c-81"})
deleted = await refinery.delete_runs("acme", run_ids=[run.run_id])
await refinery.verify_absent("acme", tags={"customer": "c-81"})
```

`delete_runs` takes `run_ids` or `tags`, exactly one, and removes the runs they match, every run
delegated or routed from them, and every label on any of them, in one transaction; it returns the
number of rows removed. A tag matches by containment: `{"customer": "c-81"}` matches a run tagged
`{"customer": "c-81", "arm": "b"}`. `verify_absent` takes the same selectors; without them it checks
the whole scope. Datasets you exported before are files you hold — drop the customer's rows from
them yourself; every row carries its run's tags.

## At a glance

Everything the package exports, by role.

- **Run things:** `Engine`, `AgentInfo`, `Stop`, `HistoryLike`.
- **What a run returns:** `Generation`, `Decision`, `Run`, `Step`, `Fingerprint`; events `TextDelta`, `ToolCallStarted`, `ToolResultReady`, `Event`.
- **Speak to models:** `GenerationModel`, `StreamingGenerationModel`, `DecisionModel`; `Message`, `ToolSpec`, `ToolCall`, `Reply`, `Usage`, `State`; parts `Text`, `Image`, `Audio`, `Document`, `Part`, `Content`; `Choice`, `Score`, `Check`, `Question`; `ChoiceAnswer`, `ScoreAnswer`, `CheckAnswer`, `Answer`, `Decided`; the shapes a question and a result are read by, `QuestionLike`, `ReplyLike`, `DecidedLike`, `AnswerLike`; `confidence`.
- **Give models tools:** `Tool`, `Tools`, `EngineTools`, `Jail`, `Workspace`.
- **Refine:** `Refinery`, `Review`, `Outcome`, `Label`, `Labeled`, `Exported`, `Conflict`, `Replay`, `Bucket`, `is_holdout`.
- **Store:** `Storage`, `InMemoryStorage`, `PostgresStorage`, `RunRow`, `LabelRow`.
- **Fail loudly:** `EngineException`, `ConfigurationError`, `NotFoundError`, `InputError`, `ModelError`, `RunawayError`, `DelegationLoopError`, `DelegationDepthError`, `StorageError`.

## Errors

`ConfigurationError` for a tree, a model, a tool or a workspace the engine cannot use — a bad
`AGENT.md`, an unknown key in the frontmatter of an `AGENT.md`, a question or a tool file
(`frontmatter: unknown key 'route.min_confidance' in agents/frontdesk/AGENT.md`), a decision agent
with `boot: all`, a file `boot: all` would send that neither `Read` nor `Load` would hand over —
over 20 MB, of a type the model's `media_types` does not name, or not the type its name says — a
question with instructions both in its frontmatter and in its body or in neither, a decision agent
with a file straight in `tasks/` or a task that repeats an id of its `questions/`, a route entry
that names a task its agent does not have or gives a decision agent nothing to ask, a tool
declared without a body or given without a declaration, a run of an agent whose kind of model is
absent, a missing workspace, a `Workspace` root that is not a
directory, a `Workspace` with neither `sandbox=` nor `unsandboxed=True` or with both, an unsandboxed
shell not on `PATH`, a `sandbox` or `shell` that is not a list of strings, a `max_steps`,
`max_depth`, `tool_timeout`, `timeout` or `output_bytes` that is not positive, an `approve` that
returns anything but `True` or `False`, a judge that is not a decision agent or a teacher that is
not a generation agent. `NotFoundError` for an agent the tree does not have — a judge, a teacher
or a replay's agent in the candidate included — a task the agent does not have
(`engine: agent 'support' has no task 'ghost'`), found when the run starts, or a run the refinery
does not.
`InputError` for a call the library refuses: an empty scope, a history that breaks a tool call from
its result, a `start` outside the history or without one, tags
outside their limits, runtime questions that do not fit, a runtime id the run already asks,
`questions=` for a generation agent, a decision run with no question to ask, media the decision
model does not declare, a media part with both or neither of `data` and `url`, a
correction of the wrong shape or for a question the run was not asked, a `delete_runs` or
`verify_absent` given both `run_ids` and `tags` or an empty one, a `delete_runs` given neither, an
`export` or a `replay` given a `holdout_percent` outside 0 to 100, a `replay` given empty `tags` or
a `limit`, `concurrency` or `repeat` below 1, a tool called with bad arguments or a path outside its
directory. `ModelError` when an adapter breaks the contract — a
result whose fields do not fit, a missing answer, probabilities that do not cover the options, lie outside 0 and 1 or do not sum to 1, a score outside its levels. `RunawayError` when
a loop exceeds `max_steps` (default 200). `DelegationLoopError` and `DelegationDepthError` for
nesting: through `Delegate` they reach the model as tool errors, through a route they raise.
`StorageError` for the database. Inside a run, a tool's error is the model's to read, never raised.
All under `EngineException` — the root takes the `Exception` suffix, every child ends in `Error`,
the same shape as rfnry-memory's. Every message names the module that raised it, then says what went
wrong in lowercase: `engine: agent 'ghost' was not found`, `workspace: path outside the workspace: ../x`.
A tool's error reaches the model with its type in front: `InputError: tools: unknown tool 'Ghost'`.
