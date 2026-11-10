# rfnry-engine

rfnry-engine is an agent harness with built-in dataset generation.

It is the scaffolding that turns a model into an agent — the loop, the tools, delegation between
agents — as a small library: an agent is an entry in its folder's `rfnry.yaml`, pointing at its
markdown, its skills and the tools it calls, running on a generation model, a decision model, or
both. Opt-in, it records every run, gathers labels from people, the world and models, and exports
datasets to evaluate, improve or train what comes next.

An agent is one of three kinds. A **generation agent** puts a language model in a loop with tools
until it has an answer. A **decision agent** asks a decision model a set of typed questions about a
state and gets typed, probabilistic answers back in one call. A **voice agent** holds a spoken
conversation with a person on a live model, for as long as the person stays. Generation and
decision agents nest in both directions: a generation agent can delegate to either kind, a decision
agent can route to either, and a voice agent reaches them through its tools. Opt-in, a
refinery records what your agents did and exports it as datasets, with every label and its
source — from people, the world and models. What you do with them — evaluate, improve prompts and
rules, train or distill models, audit — is yours.

Every public name, with its signature, is in [the API reference](api.md).

## Philosophy

**The folder is the agent.** Everything a person needs to author, review or diff about an agent is
a file under version control: `rfnry.yaml`, which declares the agents, their models and their tools,
and the markdown, skills and scripts it points at. The library reads the folder; it never writes it.

**The model chooses.** A run gives the model who the agent is and the rules every run follows, a
map of the files it may open, the list of its skills and the tools it has; the model opens what a
question needs, follows the skill that fits, and calls the tool or the agent that can help. One
agent with a well-kept folder covers many jobs, and whether a second one is worth it is
[measured](#orchestrating), not assumed.

**Two kinds of model, one or both.** Open-ended work — conversation, tool use, writing — is
generation. Closed-set judgement — which of these, how much, is this true — is decision. A
generation model does the first well and the second expensively; a decision model does the second
in milliseconds with calibrated probabilities and cannot do the first at all. The library treats
them as two contracts and lets you build an agent from either or both. A live model, which hears and
speaks itself, is a third: the engine hands it to your voice session and records what happened.

**The library runs; it does not take over.** You bring the providers that build your models, the
conversation history, the storage connection, the bucket, and the tools that must live in your
process. The tools of the folder run in worker processes the engine starts beside yours, never
inside it, until you close the engine. There is no background loop and no plugin registry;
`rfnry.yaml` is plain data the engine reads once, and every verb is a function you call. It writes nothing of its own: the only writes are a model's, inside a `Workspace`
you pass a run, and the one file the engine itself creates there, the socket a command calls the
run's tools through, which lives as long as that command ([Workspace](#workspace)).

**Nothing is hidden.** A run returns a record: what was sent, what came back, every tool call with
its timing, every model call with its usage, every child run. That record is what you feed your own
telemetry, and it is what the refinery stores and exports — everything that could be useful, for
any purpose, without assuming one.

**Tenancy by construction.** Every run names its scope. Every stored row carries it. There is no
default scope and nothing reads across scopes. Multi-tenancy is the default.

## Getting started

```
agents/
  rfnry.yaml                  the agents, their models and their tools
  support/
    instructions.md           who the agent is, the rules every run follows
    knowledge/refunds.md      read when a question needs it
    skills/escalate/SKILL.md  how to do a thing: listed, and opened when it fits
    tools/lookup_order.json   a tool: what the model reads about it
    tools/lookup_order.py     what runs when the model calls it
```

```yaml
version: 1
models:
  strong:
    provider: anthropic
tools:
  lookup_order:
    module: support/tools/lookup_order.py
    schema: support/tools/lookup_order.json
agents:
  support:
    kind: generation
    model: strong
    instructions: support/instructions.md
    files:
      include: [support/knowledge]
    skills:
      include: [support/skills/*]
    tools:
      include: [lookup_order]
```

```markdown
You are the support desk for Acme. Be brief, be exact, cite the rule you are applying. Look an
order up with lookup_order before you answer about it.
```

```json
{
  "description": "Fetch an order by its id. Returns its items, status and charges.",
  "input_schema": {
    "type": "object",
    "properties": {"order_id": {"type": "string", "description": "The order id, such as A-1001"}},
    "required": ["order_id"]
  }
}
```

```python
import json
import os
from urllib.request import urlopen

ORDERS = os.environ["ORDERS_URL"]


def run(args, call):
    with urlopen(f"{ORDERS}/{call['scope']}/orders/{args['order_id']}", timeout=10) as reply:
        return json.load(reply)
```

```python
async with Engine(root=Path("agents"), providers=providers) as engine:
    run = await engine.run("support", scope="acme", input="My invoice was charged twice.")
    print(run.reply)
```

`providers` maps each provider the file's `models:` names to a function of `(model, effort)` that
builds your adapter to that model ([Models](#models)). `lookup_order` is the agent's tool, declared
once under `tools:` and picked by the agent's `tools.include`: the JSON is what the model reads about
it — what it does, what it returns and the arguments it takes — and the script is what runs,
`run(args, call)`, in a worker process the engine starts beside your server, so a tool that crashes
takes down its worker, never your server; `call["scope"]` is the tenant of the run
([Tools](#tools)). The instructions file holds the agent's persona and the rules every run follows —
who it is, what it does, when to use which tool. The model receives it, its tools, a map of what
`files.include` reaches and the list of its skills, opens `support/knowledge/refunds.md` when the
question needs it, and answers. One call runs an agent for a tenant and returns its record; leaving
`async with` closes the engine, which stops its workers.

An agent grows by its entry:

| to give an agent | in `rfnry.yaml` | in your code |
| --- | --- | --- |
| the persona and the rules every run follows | its [`instructions`](#what-a-run-is-given) file | |
| a reference, a document, anything else | a file its [`files.include`](#files) reaches, on the map with its `description:` | |
| a way of doing something | a [skill](#skills) in `skills.include` | |
| a system of yours | a [tool](#tools-of-the-folder) declared under `tools:`, in `tools.include` | |
| a shell to write and run code | [`workspace: true`](#workspace) | a `Workspace` |
| a fast, calibrated judgement | a [decision agent](#decision-agents), in `orchestration.delegate_to` | a decision model |
| another agent to hand work to | its name in [`orchestration.delegate_to`](#nesting) | |
| one agent for several channels | its [`modes`](#modes) | `mode=` |
| a spoken conversation | a [voice agent](#voice-agents) | a session and a live model |
| a memory, or a corpus too large for a folder | | rfnry-memory's `MemoryTools` |

A new agent is a new entry: the code you write grows with the models and systems you connect, not
with the number of agents.

rfnry-engine needs Python 3.14 or later.

## Models

The library speaks to models through two protocols. Any object with the right method is a model;
adapters to vendors are yours to write and never ship here. A third, `VoiceModel`, is the live model
a [voice agent](#voice-agents) runs on, which the engine hands to your session and never calls.

```python
class GenerationModel(Protocol):
    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ReplyLike: ...

class DecisionModel(Protocol):
    async def decide(self, state: State, questions: Mapping[str, QuestionLike]) -> Mapping[str, AnswerLike] | DecidedLike: ...
```

Models are matched by shape, statically and at runtime: `ReplyLike`, `DecidedLike` and `AnswerLike`
are protocols over the fields of `Reply`, `Decided` and the answers below, and a result is read by
those fields, not its class. `QuestionLike` is the same over the questions a decision model is
handed; an adapter branches on `question.type` and reads `options`, `levels` or `yes` and `no`. An
adapter typed against another library's copy of these contracts — rfnry-memory's — fits here
unchanged, so one adapter serves both libraries: annotated with `QuestionLike` from either library,
it serves both, statically and at runtime. A result whose fields do not fit is a `ModelError`.

`Message`, `ToolSpec`, `ToolCall` and `Reply` are the message shapes the loop exchanges with a
generation model. An assistant turn is a `Message(role="assistant", content=…, tool_calls=[…])`: its
text in `content`, its calls as `ToolCall`s. A tool result is a `Message(role="tool",
tool_call_id=…, content=…)` whose `is_error` says the call failed, for adapters whose vendor marks
failed tool results. `messages` may hold consecutive `role="user"` messages — the preloaded files
ahead of the history, several messages from the inbox — so an adapter whose vendor needs alternating
roles merges them. A `Reply` whose `stop_reason` is `"tool_use"` and whose `tool_calls` is non-empty
means "run these and call me again"; anything else is final:

```python
Reply(text="", tool_calls=[ToolCall(id="c1", name="Read", args={"path": "knowledge/refunds.md"})], stop_reason="tool_use")
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
nothing beyond tool support. An adapter that also implements `generate_stream(system, messages,
tools, on_delta)` streams when the caller asks for events: it awaits `on_delta(text)` with the
reply's text as it arrives and `on_delta(text, reasoning=True)` with the model's reasoning, when its
vendor streams it; one that never passes `reasoning=` streams text alone. An adapter that has
`media_types`, the file types its model takes — `{"image/*", "application/pdf"}` — lets `Load`
refuse the rest with an error the model can act on, instead of the next model call failing; without
it, `Load` hands the model any file. Each agent follows the `media_types` of its own model: `Load`
refuses a file for the agent whose model does not take it, and for that agent alone; nothing about a
file `files.preload` sends is refused when the folder loads. An adapter that has `model`, the name of
the model it calls, names the run's model before any reply does. An adapter that has `settings`, a
JSON object of what it sends its vendor besides the conversation — `{"effort": "high",
"max_tokens": 4096}` — has them recorded beside the model; a `Reply` or a `Decided` may carry its
own for one call, and they win, even when empty. An adapter's `settings` attribute that is not a
JSON object is ignored; a reply's or a decided's settings that are not JSON break the contract
(`ModelError`). They are stored and
exported with the run, so a credential or an endpoint never goes in them.

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
alone, or a `Decided(answers, usage, model, settings)` when the adapter can report what the call cost and
which model answered; the record carries the usage, model and settings the call reported, and the adapter's
`model` and `settings` attributes when the call reports none. A decision adapter may carry
`media_types` too, the media its model reads in a state; without it, its model is handed text.

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

### Models in the folder

A model is declared once, under `models:` in `rfnry.yaml`, and an agent or a mode picks it by name,
`model: strong`, the way tools and workers are picked:

```yaml
models:
  strong:                                 # the name agents pick
    provider: anthropic                   # required: a name you give in Engine(providers=)
    model: claude-opus-5-5                # optional: the default; a run can choose another
    effort: high                          # optional: the default; a run can choose another
  fast: {provider: anthropic, model: claude-sonnet-5-5}
  judge: {provider: classifier}
```

A provider is your function, a `Provider`. The engine calls it as `provider(model, effort)`,
positionally so `functools.cache` works on it, with the entry's `model` and `effort`, strings it does
not interpret, and it returns your adapter; the credentials, the endpoint and the vendor's client
stay in it:

```python
import os
from functools import cache


@cache
def anthropic(model, effort):
    return Messages(os.environ["ANTHROPIC_API_KEY"], model=model, effort=effort)


engine = Engine(root, providers={"anthropic": anthropic, "classifier": classifier, "live": live})
```

The engine calls a provider at the start of each run and of each run it delegates or routes to — and
at each `Load` on an `EngineTools` outside a run, to read the adapter's `media_types` — and keeps
nothing it built, so you cache and close what it builds. One vendor may have several providers, such
as `anthropic` and `anthropic-backup` for a second key, and two agents that must move together pick
one entry, so one change moves both. A lead on `strong` delegating to a writer on `fast` is two
entries of one file, each run on its own adapter, in one record that names the model of every run.
`engine.models` holds the file's entries, each a `ModelEntry`.

The file holds no key and no environment variable: the engine never reads a credential, and a push
to an agents repository must not choose which of the server's variables goes to a vendor. It does
name the models, so a push can change what a run costs; a provider that must hold to a list of ids
refuses one it does not offer. `models:` is outside every fingerprint, since a run's choice can
differ from the file's, and `run.model` records what ran.

A name no entry holds and an entry nothing picks are a `ConfigurationError` when the engine loads,
and so is an entry whose provider `providers` does not hold (`engine: model 'strong' names provider
'anthropic', which Engine(providers=) does not hold; it holds classifier`); `rfnry-engine folder
check` runs with no host and does not look at providers. What a provider builds is checked against
its agent's kind when that agent's run starts (`agent 'writer' is a generation agent, and model
'judge' has no generate()`), before any model is called. A provider's own exception reaches you
unchanged before that run's first model call; in a delegated run its text becomes the tool's result,
which the delegating model and the record hold, and in a routed run the router's record holds it, so
a provider keeps keys out of its errors.

A run changes what a name means, for itself and every run it delegates or routes to. A `ModelChoice`
replaces only what it says and the rest comes from the entry — `models={"strong":
ModelChoice(model="claude-sonnet-5-5")}` keeps the entry's provider and effort — and its provider
must be one you gave, or the run is refused before any model is called. An adapter you built takes
the entry's place the same way, for a model made for one run or a tenant's own key. A consumer that
lets people switch models then needs one engine and one worker pool, not one per model:

```python
engine = Engine(root, providers=providers)
run = await engine.run("drafter", "acme", "Draft it", models={"fast": fast_llm})
```

A key that names no agent's or mode's model is refused, and a name left out stays the file's. The
kind check runs when each run starts. Runs at once each keep their own adapters. `run.model` records
what the reply reports, else the adapter's `model` attribute. `engine.tool(models=)`,
`EngineTools(models=)`, `refinery.replay(models=)` and `refinery.label(models=)` take the same
mapping; a run does not lend its models to an `engine.tool` it calls.

## Core features

### An agent is an entry in `rfnry.yaml`

`Engine(root)` reads one file, `rfnry.yaml` at the root of the folder. It declares the agents, the
models they run on, the tools they call and the workers those tools run in; everything else in the
folder is ordinary markdown, folders holding a `SKILL.md`, and Python scripts, which the file points
at and the engine reads as they are. A folder written for another harness — `.claude/agents/*.md`
beside `.claude/skills/*` — loads by writing an `rfnry.yaml` that points at its files. A root
without one is a `ConfigurationError`: `manifest: agents holds no rfnry.yaml; write one there, or
point the engine at the folder that holds it`.

```yaml
version: 1

workers:                                  # optional; without it, one worker of one process
  live:
    processes: 2                          # the tool processes of this worker
    threads: 32                           # plain def run calls at once, in each process
    max_result_bytes: 33554432            # the largest result a tool may send back

models:                                   # each model once; agents pick them by name
  strong: {provider: anthropic, model: claude-opus-5-5, effort: high}
  judge: {provider: classifier}

tools:                                    # each tool once; agents pick them by name
  lookup_order: {module: support/tools/lookup_order.py, schema: support/tools/lookup_order.json}
  refund: {module: support/tools/refund.py, schema: support/tools/refund.json, timeout: 20}

agents:
  support:
    kind: generation                      # generation, decision or voice
    model: strong                         # a name under models:
    description: Handles support          # else the description: of its instructions file
    instructions: support/instructions.md # any markdown file; its frontmatter is left out
    max_model_calls: 24                   # per run; lowers Engine(max_model_calls=), never raises it
    files:
      include: [support/knowledge]        # what the model may open: folders, files or globs
      exclude: [support/knowledge/drafts/**]
      preload: [support/policy.md]        # sent with every model call
    tools:
      include: ["*", handoff]             # declared above, given by the engine, or by the run
      timeout: 30
    skills:
      include: [support/skills/*]         # folders holding SKILL.md, offered to the model
      public: [escalate]                  # people may type /escalate
    orchestration:
      delegate_to: [triage]               # the agents its Delegate tool may reach
      max_depth: 1
    workspace: true                       # a shell and files to write; the run passes a Workspace
    modes:
      chat:                               # engine.run("support", ..., mode="chat")
        worker: live
        tools:
          exclude: [refund]

  triage:
    kind: decision
    model: judge
    description: Routes a message to the desk that handles it
    instructions: triage/instructions.md
    questions: [triage/*.json]
    orchestration:
      route: {by: intent, to: {billing: support}, min_confidence: 0.7}
```

| section | declares | |
| --- | --- | --- |
| `version` | `1`, required | |
| `models:` | each model once, under the name agents pick it by; required | [Models in the folder](#models-in-the-folder) |
| `tools:` | each tool script once, under the name the model sees | [Tools of the folder](#tools-of-the-folder) |
| `workers:` | the processes the tools run in; without it, one `default` worker | [Workers](#workers) |
| `agents:` | each agent, under the name a run calls it by; required | below |

| an entry's key | holds | |
| --- | --- | --- |
| `kind` | `generation`, `decision` or `voice`; required | [Decision agents](#decision-agents), [Voice agents](#voice-agents) |
| `model` | a name under `models:`; required | [Models in the folder](#models-in-the-folder) |
| `description` | the line other agents and `engine.tool` show; else its instructions' `description:` | |
| `instructions` | a markdown file: who the agent is and the rules every run follows; on the entry or on every mode | [What a run is given](#what-a-run-is-given) |
| `max_model_calls` | the model calls of one run | [Limits](#limits) |
| `worker` | the worker its tools run in, `default` when left out | [Workers](#workers) |
| `files` | what its model may open | [`files`](#files) |
| `tools` | the tools it has | [Tools of the folder](#tools-of-the-folder) |
| `skills` | the skills it has | [Skills](#skills) |
| `orchestration` | the agents it may delegate or route to, how deep and how many at once | [Nesting](#nesting), [Both kinds together](#both-kinds-together) |
| `workspace` | `true` for a shell and files to write | [Workspace](#workspace) |
| `questions` | a decision agent's questions | [Decision agents](#decision-agents) |
| `modes` | named overlays of the entry | [Modes](#modes) |

The file is plain data, checked when the engine loads it, so a mistake fails at start rather than
in a run. A key a section does not take is a `ConfigurationError` naming its dotted path and the
keys taken there (`manifest: unknown key in /srv/agents/rfnry.yaml: agents.support.file (it takes
description, instructions, model, max_model_calls, worker, files, tools, skills, orchestration,
workspace, questions, kind, modes)`), and so is a key with no value, an empty list or an empty
mapping, which says nothing. An agent has exactly what its entry says: a missing section gives it
nothing, and there is no `all` or `none` for its tools or skills. A generation agent takes every key
but `questions` and `orchestration.route`; a decision agent takes `kind`, `model`, `description`,
`instructions`, `worker`, `questions`, `orchestration.route` and `modes`; a voice agent takes `kind`,
`model`, `description`, `instructions`, `worker`, `tools` and `modes`; any other key on its entry or
on a mode is a `ConfigurationError` naming it. A name under `agents:`, `models:`, `tools:`,
`workers:` or `modes:` is plain — not blank or padded, no leading `.`, no `/` — and so is a `model:`.
The YAML is read strictly: a key written twice is refused, only `true` and `false` are booleans, and
anchors, aliases and merge keys are refused, so nothing is copied between entries and a file from
another repository cannot expand as it loads.

Every path in the file is written from the folder root, with no leading `/`, no `..` and no
backslash. The file tools and the prompt use the same paths, so an instructions file that says
`support/knowledge/refunds.md` means that file in every call, and in every harness run from the
folder root.

#### What a run is given

At the start of a run the model receives the agent's instructions file, its frontmatter left out,
a map of the files it may open, the list of its [skills](#skills), the list of tools it has, the
descriptions of the agents it may call and any context the caller passes. The engine does not care
where rules live; put the ones every run follows in the instructions, which are in the prompt of
every run. A file the model has to open may not be opened, depending on the model, so one that must
reach every run is preloaded ([`files`](#files)).

The map is made when the engine loads: every folder and file the agent may open, under its path from
the root, less its own instructions, its tools' files and its offered skills, which the prompt
already holds, and beside a markdown file the `description:` of its frontmatter when it has one:

```markdown
---
description: When a refund is owed, and what needs a person
---
A duplicate charge is refunded in full without approval.
```

```
support/
  knowledge/
    rates.pdf
    refunds.md — When a refund is owed, and what needs a person
```

A file without a description is listed by its name, which often says enough; a description is one
line of what the file holds, so the model knows what to open without opening it. The frontmatter of
an instructions file, a `SKILL.md` or a markdown file on the map is read only for its
`description`, so a folder written for another harness loads unchanged: YAML that does not parse is
no error, and a top-level `description:` line is still read, its matching quotes stripped. The map
and the list of skills go only to an agent that can open them, one whose tools include `Read` or
`Load`; an agent without them still gets a skill a run invokes with `skill=` or `/name`, since its
text goes in the input.

`Read` returns a text file as text. `Load` hands the model any other file as it is — an image, a
PDF, a recording, a spreadsheet — so a price list, a diagram or a scanned form sits in the folder as
it came, the model reads it with its own understanding of images and documents, there is no parser
in between, and nothing is loaded until the model asks — unless the agent preloads it.

`engine.agents` lists the agents as the engine loaded them, each name to an `AgentInfo`: its `kind`,
`model` and `description`, the agents it may `delegate_to`, the `files` its model may open, the
`tools` it picks from `tools:` by name (the built-ins and the tools a run gives are not listed), the
`skills` it holds, each to its description, the `public` ones people may call, its `worker` and its
`modes`, each an `AgentInfo` of its own. When the folder has loaded, the engine logs the names of
its agents at INFO on the `rfnry_engine.engine` logger — `engine: agents loaded [lead, triage,
writer]` — which `logging.basicConfig(level=logging.INFO)` shows.

#### `files`

`files:` decides what a generation agent's model may open:

```yaml
support:
  kind: generation
  model: strong
  instructions: support/instructions.md
  files:
    include: [support/knowledge, shared/*.md]   # folders, files or globs
    exclude: [support/knowledge/drafts/**]      # taken back out of include and preload
    preload: [support/policy.md]                # sent with every model call
    gitignore: true                             # what a .gitignore ignores is not there
```

`include` lists folders, files or globs, `*` within a name and `**` across folders. `preload` is
sent with every model call and stays readable whether or not `include` reaches it. `exclude` takes
paths back out of both, so an excluded file is neither sent nor readable. What is out of reach reads
as missing to `Read`, `Load`, `LS`, `Grep` and `Glob`, which come with `files`, and none of them
lists it. Reach is fixed when the engine loads: a file added later reads as missing until a new
`Engine` loads the folder. A pattern that reaches nothing is a `ConfigurationError` (`compiled:
agent 'support' lists 'support/knowlege' in files.include, which reaches nothing; write a folder, a
file or a glob from the folder root that reaches a file, or remove it`), so a mistyped path cannot
quietly shrink what a model sees, and `rfnry.yaml` itself is never readable by a model.

A hidden name — one that starts with `.` — and `__pycache__` are reached only where a path in the
file names them, as `.claude/agents` does; hidden names below that path stay hidden, and a path
through one reads `tools/builtins: .env is hidden; names starting with . and __pycache__ are not
part of the folder of the agents`. A link that leads out of the folder is left out. One that stays
inside is read under its own name, a file link to a hidden file included, when what it leads to is
in reach too: `exclude`, a `.gitignore` and a skill's folder judge a link by where it leads as well
as by its own path, so `knowledge/alias.md` linking to an excluded `drafts/next.md`, or to the
`SKILL.md` of a skill the entry does not open, reads as missing and is not preloaded. A path that
leads outside the folder is the jail's `InputError`, `jail: path outside the folder: <path>`, and a
part written after a file names a folder that is not there, as POSIX has it: `Read` of
`notes.md/x` is `tools/builtins: not a folder: notes.md/x`.

`preload` sends its entries in the order of the list, each file under its path from the root, as
what `Read` would return for it. A file that is not text, or is over 5 MB, goes as a line naming its
type and size, such as `image/png, 15 bytes; Load opens it`, which `Load` opens when the model asks;
an agent without `Load` gets the line without `; Load opens it`. They travel as one user message
ahead of the history, sent with every model call, and are read when the engine loads, so an edit
shows in the next `Engine`. `run.preload` holds that message's parts; `run.messages` and
`run.conversation` leave it out, so a conversation passed back does not send the files twice and a
replay preloads from the candidate's folder. Each agent sends its own, a delegated or routed one
included.

`gitignore: true` leaves out what a `.gitignore` ignores, as Git would, so a folder kept in a
repository can hold its `build/` and its logs without the agent seeing them. The `.gitignore` files
at the folder root and below it decide; patterns are relative to their file, a deeper file overrides
a shallower one, `!` re-includes, and nothing under an ignored folder can be re-included. The engine
reads each one once, when it loads, and reads no Git data: a file a pattern matches is ignored even
if Git tracks it. A file so ignored is not there for that agent — not on its map, not preloaded, not
in its skills' folders or its fingerprint — while the files the agent is made of, its instructions,
its tools' files and its questions, load whatever a `.gitignore` says. A line that is no valid pattern is skipped,
as Git skips it, a `.gitignore` that is a link or not a plain file is not read, and one that cannot
be read or decoded is a `ConfigurationError` (`tree: cannot read agents/support/.gitignore:
Permission denied; make it readable to the engine`).

A folder or a content file the engine cannot read is left out, with one warning naming it each time
the folder loads, on the `rfnry_engine.folders` logger, and a file removed while its folder is listed
is left out as though it was never there. A file an agent is made of — its instructions, a skill's
`SKILL.md`, a question's `.json`, a tool's module or schema, a module beside it, a question's script
or helper — is never left out: one the engine cannot read is a `ConfigurationError` naming it, as is
a file or folder it reaches whose name is not UTF-8 (`tree: the name of
'/srv/agents/support/caf\udce9.md' is not UTF-8; rename it`).

`files` decides what the file tools and the prompt hold; it is not a boundary. A workspace's shell
sees the real files: with `Workspace(tree=)` a command reads the whole mount, `rfnry.yaml`, hidden
names and what `files` leaves out included. What must stay away from a model stays out of the folder.

#### Modes

A mode is a named overlay on one agent, written under its `modes:` and holding only the
differences: lists add to the entry's, single values replace them, and `exclude` wins over
`include` wherever either was written. A mode may set `description`, `instructions`, `model`,
`max_model_calls`, `worker`, `files`, `tools`, `skills`, `orchestration`, `workspace` and
`questions`, as far as its kind takes them, so one agent serves several channels, each with only
what it needs:

```yaml
support:
  kind: generation
  model: strong
  instructions: support/instructions.md
  skills:
    include: [support/skills/*]
  modes:
    email:
      skills:
        preload: [support/channels/email]
    chat:
      tools:
        include: [verify_customer]
      skills:
        preload: [support/channels/chat]
```

A run selects one with `mode=`, `engine.run("support", scope, message, mode="chat")`, and the
record's `mode` says which. Each mode compiles to a full agent when the engine loads, so a bad mode
fails at start, and each has its own fingerprint. An unknown mode is a `NotFoundError` before any
model is called, and an entry with no `instructions` needs a mode, so a run without one is an
`InputError` naming its modes. Delegated and routed runs, and a run through `engine.tool`, take no
mode: they run on the agent's entry, so `delegate_to` or `route` naming an agent that needs a mode
is a `ConfigurationError` when the engine loads. A replay runs on the mode the run recorded.

#### Limits

An entry's limits can lower what the `Engine` allows and never raise it, so a push to an agents
repository cannot raise a limit; the file names the models, so a push can change what a run costs.
`max_model_calls` bounds the model calls of one run, and a delegated run counts its own.
`orchestration.max_depth` counts levels of delegation below the agent: with `max_depth: 1` it may
delegate, and the agent it delegates to reads `DelegationDepthError` as a tool error if it delegates
further; a child's own `max_depth` cannot widen what an agent above it allows. An `EngineTools` is
no run, so the entry's `orchestration` limits do not apply to a `Delegate` dispatched through one.
`Engine(max_model_calls=)`, `Engine(max_depth=)` and `Engine(tool_timeout=)` are the ceilings: the
smaller value holds, `tool_timeout=None` leaves the entry's `tools.timeout`, and `Engine(max_depth=)`
counts the whole stack, the first agent included, so `Engine(max_depth=1)` allows no delegation
whatever an entry says. `tools.max_parallel` caps a run's tool calls at once and
`orchestration.max_parallel` its `Delegate` calls, which `tools.max_parallel` does not count;
without one, calls of that kind are not capped. Each run has its own slots. A call takes its slot
after `approve`, so a call waiting on a person holds none, and its deadline starts once it has one.
A call a command makes through the tool bridge takes a slot of `tools.max_parallel`, so a script's
fan-out stays under the cap, and a `Shell` on a run with a bridge takes none, so a command never
waits on a slot its own call holds.

### Skills

A skill is how to do something: a folder holding `SKILL.md`, named by the folder, in the shape of
the Agent Skills convention, so a skill folder written for another tool, `.claude/skills/*`, works as
it is. Its other files — references, templates, scripts — sit beside `SKILL.md`, which says when to
open them. An entry's `skills:` decides which skills a generation agent has:

```yaml
support:
  kind: generation
  model: strong
  instructions: support/instructions.md
  skills:
    include: [support/skills/*]          # offered: listed for the model to open
    exclude: [internal]                  # taken back out, by name, folder or glob
    preload: [support/skills/tone]       # its text sent with every model call
    public: [escalate]                   # people may type /escalate; or the word all
```

`include` and `preload` take the path of a skill's folder or a glob reaching some, from the folder
root; a glob's match that holds no `SKILL.md` is left out, and a path or a glob that reaches no
skill is a `ConfigurationError`, as are two folders that give one name, naming both. `Read`,
`Grep`, `Glob`, `LS` and `Load` come with `skills` as they do with `files`. The model is offered
`include` less `exclude`, listed in the prompt's `SKILLS` section, one line per skill in
name order, under the path `Read` takes and with the `description` its frontmatter gives:

```
<<<SKILLS>>>
support/skills/escalate/SKILL.md — Hand a conversation to a person
```

The model opens a skill with `Read` when it judges it fits, and finds the skill's other files
through it; there is no tool of its own for skills. An offered skill's folder leaves the map and
stays open to the file tools. `exclude` takes a skill's name, its folder or a glob, and takes the
skill out of `preload` too, so an excluded skill is neither listed nor sent; one that names none of
the agent's skills, or a `preload` it leaves with nothing, is a `ConfigurationError`. `preload`
sends a skill's text, its frontmatter left out, with every model call, in the shape `skill=` sends
it, whether or not the list offers it; a `files.preload` of the same `SKILL.md` keeps the file as
it is, and the file goes once. A skill's text is read when the engine loads, so editing it afterwards
changes no prompt until a new `Engine`.

A skill the model is not offered is reachable only where `files.include` covers it, and the folder of
the skill a run invokes is open for that run. `files.exclude` holds inside a skill's folder, for
everything but its `SKILL.md`, which `skills.exclude` decides, and a skill excluded from the list
stays closed through a skill it is nested in. This hides a skill from the file tools and the prompt;
it is not a boundary. A script a skill keeps is a file like any other: to run it, the agent needs a
[workspace](#workspace) whose shell can reach the folder, `Workspace(tree=)`, and the model runs
`python "$RFNRY_TREE/support/skills/plot/run.py"`.

A run invokes a skill in two ways: from the code with `skill=`, or from a person with an input that
starts with `/<name>`, when the entry makes that skill public:

```python
run = await engine.run("support", scope="acme", input="A third complaint this week.", skill="escalate")
run.skill                                            # "escalate"

run = await engine.run("support", scope="acme", input="/escalate A third complaint.")
run.skill                                            # "escalate"
run.input                                            # "A third complaint."
```

The skill goes in front of the input, in the same user message: a text part of `<<<SKILL
support/skills/escalate/SKILL.md>>>` on its own line and the file's body without its frontmatter,
then the input as it came. It is part of the conversation, so a run that continues from it with no
`skill=` still has the skill's text, but not its folder unless the entry offers it. `skill=` reaches
any skill the entry or one of its modes names, offered, preloaded or excluded alike, and
`engine.agents[name].skills` lists them, each to its description, `""` when it has none. A skill the
agent does not hold is a `NotFoundError` before any model call, naming the ones it holds —
`engine: agent 'support' has no skill 'refund'; its skills are escalate` — and `skill=` on a
decision agent is an `InputError`. `run.skill` records it, an exported generation row carries it,
and a replay asks the candidate for it again, so the candidate's own skill file is the one read.
Delegated and routed runs never take the caller's skill.

`public` names the skills people may call by typing `/name`, or is the word `all` for every skill
the model is offered; left out, typing `/name` invokes nothing. A name the entry does not offer, or
one that holds whitespace, which no one could type, is a `ConfigurationError`.
`engine.agents[name].public` is the menu a consumer shows people: the public skills in name order,
each to its description, `{}` when there are none.

`engine.run` reads an input for a command when it is a string that starts with `/`, at its very
first character, or a list whose first part is a `Text` that does. Its first word is the text up to
the first whitespace; when that word is `/` and the name of a public skill, the run takes that skill
and the input becomes the text after the word, its leading whitespace stripped — in a list the
first part becomes that text, or is dropped when nothing follows, and a list left with no part
becomes `""`. Any other input stays as it came, and is never an error: a path (`/etc/passwd`), an
unknown name, `/escalate/x`, `/escalatex`, ` /escalate`, a JSON object or a JSON list that is not
parts, and every input to a decision agent, which has no skills. A skill the agent holds but does
not make public is no command either, so typing its name gets the answer any other text would.

`public` decides what typing invokes, not what the model is offered: `/secret x` reaches the model
as text, and an offered skill stays in its `SKILLS` list whether or not it is public. A skill the
model should not be offered is named in `exclude`: the code still reaches it with `skill=`, and no
person can, since `public` names only skills the model is offered.

Giving both ways at once is an `InputError` before any model call — `engine: a run invokes one
skill, by skill= or by an input that starts with /<name>; got skill='escalate' and /close` — while
`skill="escalate"` with an input of `/ghost x`, which names no public skill, takes `escalate` and
keeps that input. The run records `skill` and the rest of the input as its `input`, and a replay
sends that pair and never reads `/name` again. `Delegate`, `engine.tool` and routes never read
`/name`: their input comes from a model or a router, and they never take a skill.

### Running an agent

`engine.run(agent, scope, input)` names the agent, the tenant and the input, and returns a record.
For a generation agent that record is a `Generation`: the reply, the system prompt as sent, the
files and skills its entry preloads (`run.preload`), the messages as sent to the first model call, the whole
conversation, the tools the model was offered (`run.tools`, each a `ToolSpec`), every tool call as a
`Step` with its result and timing, the model calls and their summed usage, and any child runs. It is
the whole observability payload; hand it to your own telemetry:

```python
run.run_id, run.model, run.calls, run.usage.input, run.usage.output, run.duration_ms
for step in run.steps:
    step.tool, step.args, step.is_error, step.duration_ms, step.child_run_id
run.children                                         # runs made through Delegate, in step order
```

`run.model` is the model the replies reported, the resolved one — a dated snapshot, say. When no
reply reported one — a run stopped during its first model call — it is the adapter's `model`
attribute, if it has one that is a non-empty string, and `None` otherwise. `run.settings` is the
settings of the call that wrote the run's last assistant message: that reply's `settings`, else the
adapter's `settings`.

Every run says which agent and which tools produced it, beside the model that answered:

```python
run.fingerprint.agent                                # "3f9a1c0e5b7d2a41": its entry and its files
run.fingerprint.tools                                # "a02c77e1f4b9d305": the tools it was offered
```

`fingerprint.agent` is a short sha256 over the agent's entry, resolved for the run's mode with the
entries of the tools it picks, and over every file it reaches, by its path from the root and its
content: its instructions, the files its `files` reaches and preloads, the folders of the skills it
holds, offered to the model or not, its tools' modules, schemas and the modules beside them that they
import, and its questions with their scripts — what `files` leaves out, a `.gitignore` under
`files.gitignore: true` included, it leaves out too. Editing another agent's entry changes nothing
here, and neither does `models:`, which is outside it. The engine takes it once, when it loads the
folder, so an edit shows in the next `Engine` built from the folder.
`fingerprint.tools` is one over `run.tools` — each tool's name, description and schema — so it
covers the tools of the folder it picks, the `tools=` a run passes, `Output` when `expect=`
asks for one, in `Delegate` the agents it may call — their names, descriptions and kinds — and a
workspace's tools, whose `Shell` description says whether the workspace has `tree=` and whether its
commands can call tools.
Neither sees what varies per run and is not the agent — the input, the history, `context=`, `tags=`
— so two runs of an unchanged folder share it and any edit changes it, a tool's script included. The
model is `run.model`; the code behind a tool a run passes is yours to version. A decision run
carries one too, its `tools` the fingerprint of no tools and its `questions` the questions it asked
([Decision agents](#decision-agents)), and each delegated or routed run carries its own.

Keep the conversation yourself and pass it back. `run.conversation` is the history the run was
sent, the input, every assistant turn — its text in `content`, its calls in `tool_calls`, its
`reasoning` as the model returned it — every tool result with its `is_error`, and the final
assistant turn, so the next run continues from what the agent did, not only what it said:

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

A list of parts becomes the user message after the history as it is; parts in `history` reach the
model unchanged, and a later run continues from them. History, like input, is read by shape: a
message or part from another library is read by its fields.

Pass `record=False` to run without the refinery recording it — a replay, an evaluation, a dry run.
A consult it makes through an `engine.tool`, or a `Delegate` dispatched through an `EngineTools`,
is not recorded either, so a replay whose candidate consults a production engine leaves the scope
as it found it. A run your own tool starts with `engine.run` is recorded unless it passes
`record=False`.

A run that raises instead of returning — the vendor refused the call, the adapter broke the
contract (`ModelError`), the loop passed `max_model_calls` (`RunawayError`) — returns no record; the
exception is what you have, the same one the adapter or the engine raised. With a refinery
attached, the run is recorded first, as far as it got: `error` says what ended it —
`"ModelError: …"` — `reply` is `""` and `stop_reason` is its last reply's. When `on_event` raises
on one call of a reply, the reply's other calls are still answered, run or cut by a stop, and their
events still reach the handler; the run then raises the first failure and logs each further one as
an error, `engine: a further call of the same batch failed: …`. Its conversation can hold a tool
call with no result, when the run failed between the call and its result, and `run()` refuses such
a history: go on from the history the run was given. A run refused before its first model call — a
bad input, an agent or skill the folder does not have, a route loop — has nothing to record,
and a run whose task you cancel is not recorded; `stop=` ends a run with its record. A generation
run whose `on_event` raises on its `RunStarted` had started: it is recorded with that `error` and
no model call. A delegated run that fails stays under its parent in `children`, its step the error
the parent's model was given, and the parent goes on; a routed run that fails fails its router,
which is recorded with its answers and the failed run in `routed_to`, or `None` when the routed run
was refused before its first model call. A run that failed keeps the delegated runs that had
returned in `children`: those its steps answered, in step order, then those it had not answered yet,
in the order they started, as `refinery.run` reads them back.

Pass `tags=` to mark a run with what you will group it by later — the commit of the folder, an
experiment arm, a release. Keys and values are strings — at most 32 tags, keys of 1 to 64
characters, values up to 256 — anything else is an `InputError`. The record and every exported row
carry them, and so does every tool call the run makes, in `call` ([Tools](#tools)); the model never
sees them. Delegated and routed runs inherit the tags of the run that started them, so a version or
an arm covers the whole tree of runs, and a ticket the tools are bound to binds the delegated run's
tools as well:

```python
run = await engine.run("support", scope="acme", input=message, tags={"commit": "4f2c9e1", "arm": "b"})
run.tags                                             # {"commit": "4f2c9e1", "arm": "b"}
```

Ask for a typed reply and the model is offered an `Output` tool shaped like it. The `expect=` model
is the output's contract, defined in code: its field descriptions and its docstring reach the model
in the tool's schema, and what the output means beyond that is said in the instructions or a skill. When
the model calls it with input that fits, the run ends and `run.reply` is that input as JSON; input
that does not fit goes back to the model with what is wrong. A model that answers in text instead
ends the run with that text as `run.reply`, which does not parse:

```python
class Verdict(BaseModel):
    refund: bool
    reason: str

run = await engine.run("support", scope="acme", input=message, expect=Verdict)
verdict = Verdict.model_validate_json(run.reply)
```

An accepted `Output` ends the run. A call the model made after it in the same turn is not run; it is
answered with an error saying so, so every call in `run.conversation` has its result. Of a reply
with more than one `Output`, the first that fits is taken: each before it is answered with what is
wrong, each after it is not run, and the reply's other calls before it run as they would without it.

Pass `context=` to add data about this run that does not belong in the folder — the contract under
review, today's date, never instructions — the record keeps it as `run.context` — and pass
`on_event=` to see the run as it happens: where it starts, its reasoning and its text, the messages
it takes from the inbox, its tool calls and their results:

```python
async def show(event):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)

run = await engine.run("support", scope="acme", input=message,
                       context="Acme is on the enterprise plan. Refunds up to $500 need no approval.",
                       on_event=show)
```

A `ToolCallStarted` carries the call's `args`, and a `ToolResultReady` its `step`: the same `Step`
the run records, sent once however the call ended — a result, an error, a refusal, a delegation, a
stop — so a step can be shown, in a conversation or anywhere, while the run goes on. A call
`stop.now()` cuts off sends its `ToolResultReady` whether or not it had started, an `Output`
included; any other call the engine answers without running it sends neither: an `Output` it accepts
or refuses, a call after an accepted `Output`, a call in a reply that was cut short. Its step is in
`run.steps`. Each handler, `approve`, tool and `expect=` model is handed its own copy of a call's
`args` — every list and object copied, any other value as it is — and a handler's `step`, like the
messages a `MessagesTaken` or the adapter is handed, holds copies of the parts, so one that changes
what it is handed changes neither the record nor what the model is sent next.

The same holds wherever the engine hands your code something. A generation model is handed, for each
call, copies of the messages — their parts, their tool calls' `args`, their `reasoning` — and of the
tool specs; a decision model, copies of the state and of the questions; a `Backend`, its own copy of
a command's variables; a storage, rows made for the call; each attempt of a replay, a recorded run
and labels of its own. And a record holds objects of its own: it copies what it is given — the
input, the history, what the inbox held, the questions asked at run time, the tools' specs and
results, each reply's tool calls and reasoning, each answer — and the preload and the questions it
takes from the folder, and a routed run's input is its own copy of its router's state. So no code of yours, and no tenant, reaches another run through what it was
handed or what it returned: changing a record changes nothing the folder shares with every tenant,
nothing of another record and nothing your code holds.

A foreground `Shell` command of a run with a `Workspace` also sends a `CommandOutput` for what
it writes, while it runs, and the calls it makes to the run's tools send their own
`ToolCallStarted` and `ToolResultReady`, with `by` the `Shell` call's id. See the
[Workspace](#workspace) section.

Every model call is a turn, and a `TextDelta` carries the `turn` that said it, from 0. Text of a
turn that ends in tool calls — "Let me look that up." — is said beside them; a run that ends in
text replies with its last turn's text, so `run.reply` is that text and nothing said before it. A
delegated or routed run sends its own `run_id` and counts its own turns.

Every generation run sends a `RunStarted` first, before anything else. Its `call` is the tool call
the run works for — the `Delegate` call that started a delegated run, the call an `engine.tool`
consult was made in, the call that started the decision run a run was routed by — or `None` for a
run started outside any tool call. A frontend places every later event of that `run_id` under that
call, so a sub-agent's text, reasoning and tool calls show nested under the call that asked for
them, at any depth. The event names the call even when it was made in another scope, so a consult
in a shared scope still nests where it was asked; the run's record keeps a `call` only from its own
scope, so no stored row holds another tenant's ids. A
decision run sends no events: the `ToolCallStarted` and `ToolResultReady` of the call that asked it
show it. A `ReasoningDelta` is the model's reasoning as the adapter streams it, with its turn, to show
apart from the reply: it never becomes part of the reply or the conversation, which keeps the
reasoning as the adapter returned it, and reasoning streamed before `stop.now()` is not kept. A model
that does not stream its reasoning sends none.

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

When the run takes messages it sends a `MessagesTaken` with them and the turn about to read them.
The inbox is first in, first out and the run takes everything waiting, so a frontend can show a
message as sent but not yet read, and mark the first `len(event.messages)` as read when the event
arrives. Messages still waiting when the run ends were never read and send nothing.

A stopped run returns, never raises: `run.stopped` is `"now"` or `"after_step"` (`None` for a run
that ended on its own), `run.steps` are the steps it took, `run.reply` is the last text the model
produced — text streamed before `now()` included — or `""`, and it is recorded like any run. A tool
call cut off by `now()` is answered with the error `stopped by the user` and its `Step` has
`stopped=True`, so every tool call in `run.conversation` has its result and the next run continues
from it. A call the stop finds not yet running never runs: a stop from a handler of a call's
`ToolCallStarted` comes before its tool runs, and one from a handler of a reply's `TextDelta` before
any of the reply's calls. A handler still running when a stop cancels its call, or a streamed
`TextDelta` or `ReasoningDelta` handler, is cancelled with that work, so a kill switch does its own
work, such as an audit, before it calls `stop.now()`. A call that already has its result keeps it,
and the stop waits for an `on_event` still handling its `ToolResultReady` to return: a handler can
stop the run on what a tool returned; a slow one delays the stop, and cancelling the run cancels it.
The stop waits the same way for a handler of a `RunStarted` or a `MessagesTaken`, which the run sends
between its steps, and one such handler sets ends the run before its next model call. Cancelling the run's
task cancels its model call and every tool call still running, delegations and their runs included,
and re-raises once all of them have ended, as an `asyncio.TaskGroup` does — a call whose tool
returns after the cancel ends with it, before its `ToolResultReady` — and a stop waits the same way
for the work it cancels, so a tool that ignores cancellation holds either one. The first stop is
graceful: a workspace command is sent SIGTERM and given a second to end, and one the workspace is
already stopping on its own — a `Shell` past its timeout, the kill after a `Shell` ended,
`Job(stop=true)` or `close()` — counts as stopped once, so the first stop lets its second run out. A
second stop — `stop.now()` again, or cancelling a run that is already stopping — ends what is still
running at once, with no grace period: each further `now()` cancels once more every tool call still
running but a call whose generation run — a delegated one, or a consult through `engine.tool` —
takes the same `now()`, and a call whose `ToolResultReady` handler the stop waits for; each further
cancel cancels everything still running once more; and a workspace kills its command outright. A
stop set before the run starts ends it before the first model call; one set after it ended does
nothing. A stop reaches delegated and routed runs, and the runs an `engine.tool` starts while the
run calls it, which end the same way at their own next boundary. With `after_step()` pending the
inbox is not read. Whatever is still in the inbox when the run ends stays there — the engine never
starts a run; you do, with those messages as its input. A decision run never reads the inbox: its
questions are one call, and a stop or an inbox rides to the run it routes to; a stop that finds a
delegated or consulted decision run still asking cancels its call, as it cancels a model call in
flight, and that run is not recorded. "Pause, please" is a message in the inbox: the model reads it
at the next boundary and wraps up in its own words.

A run that ended continues from its record, on any process, after any restart. With a refinery
attached every run that returns is recorded with its conversation: keep the run id beside what you
showed — a message's metadata, say — read the run back and pass its conversation on:

```python
last = await refinery.run("acme", run.run_id)
next = await engine.run("support", scope="acme", input="And the second charge?", history=last.conversation)
```

`refinery.run(scope, run_id)` returns the run as `engine.run` returned it, its delegated and routed
runs back in place; a child's id returns that child and the runs under it. A run id the scope does
not have is a `NotFoundError`. A deploy is a stop: on SIGTERM, ask every run in flight to stop after
its step. Each ends at its next boundary and is recorded with `stopped="after_step"`; the next
message, on whichever task takes it, reads the run back and continues:

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
and `run.added` — the input and every message after it — is what to keep; a router that routes
nowhere adds nothing, `[]`, so `conversation += run.added` keeps nothing of that turn. `start=0`,
the default, sends the whole history. `run.start` records where the run began, and
`run.conversation` is what the model saw: the history from `start`, then `run.added`. A routed run
is given the history and `start` its router was; a delegated run starts with no history, as always.

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
`HistoryLike`, and the run reads it only from `start` on:

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

When a run starts past the beginning, the model is offered `History`, which reads the messages
before the first one it was given, each under its position — a page at a time going back, each
page in the order it happened — or, with `pattern`, the ones whose text matches:

```json
{"before": 340, "limit": 20, "pattern": "invoice", "case_insensitive": true}
```

It never reads what the model was given or anything after it. A page is 20 messages, at most 100;
a search walks back at most `Engine(history_walk=10_000)` messages and a result holds at most
`Engine(history_chars=100_000)` characters of messages, ending with where to go on when there is
more; its deadline is `tool_timeout`, as every tool's, so a slow database gets the time you give
it. No line of the instructions asks for it: it comes with `start=`, as `tools=` comes with the call. To
send a window without it, pass the slice as the history and leave `start` at 0.

`History` never hands a file back. An image, audio or document shows as a line naming it, with its
URL when it has one — `[image: image/jpeg, https://files.acme.com/a1.jpg]`,
`[document: application/pdf, "rates.pdf"]` — and the model fetches what it wants to see again:
`Load` for a file of the folder or the workspace, whose call is in
the page above its result, or a tool of yours. A file is paid for on every call that holds it — an
image by its pixels, a PDF by its pages — so one the model has read need not stay in your
conversation either. Keep the line instead, and a tool that brings the file back:

```python
def forgotten(message):
    if isinstance(message.content, str):
        return message
    named = [p if isinstance(p, Text) or p.url is None
             else Text(text=f"[{p.type}: {p.media_type}, {p.url}]") for p in message.content]
    return message.model_copy(update={"content": named})


class SeeArgs(BaseModel):
    url: str
    media_type: str


async def see(args):
    return [Image(media_type=args["media_type"], url=args["url"])]


look = Tools([Tool("See", "Look again at an image by its URL", body=see, args=SeeArgs)])
conversation = [*history, *map(forgotten, run.added)]
```

Pass `tools=look` to each run. `See` hands the vendor the URL to download, so it must be one the
vendor can reach; a document comes back as a `Document` the same way, and a file in your own store
by its data, as `fetch_invoice` does under Tools. A file passed as data has nowhere to come back
from, so `forgotten` keeps it. The turns after a file you swapped send their reasoning no more —
what came before them changed — and the run goes on without it.

Reasoning goes back only where it was given. Some vendors bind a model's reasoning to the exact
conversation it was given in and refuse it anywhere else. Every assistant message the engine makes
from a reply carries `usage`, what the call that wrote it reported, `model`, the model that wrote
it, `settings`, the settings it reported or its adapter declared, and, when the reply carries
reasoning, `prefix`, a short hash of everything sent before it — the model, as its adapter names it,
the system prompt, the tools, the preloaded files and the messages. A message's `reasoning` is sent
again only when what comes before it is still the same; otherwise the message goes without it, its
text and calls as they were, and `run.messages` shows it as sent. So a history you cut or edited, a
changed `context=` or `tools=`, another agent, another model and a replay never send
reasoning a vendor would refuse. A run stopped after a step and continued after a change ahead of
its history — a new date in `context=`, a moved window — sends its last tool call without its
reasoning; a vendor that wants reasoning back on an open tool loop may refuse that, and handling it
is the adapter's job. `prefix`, `usage`, `model` and `settings` are the engine's record; no model
sees them, and `model` says which model wrote each reply when an adapter changes models mid-run,
where the run's `model` names the last. `settings` are not part of the hash: changing them never
takes reasoning away.

Caching is the adapter's. An adapter that marks a cache point has every call of a window read from
the cache what the last call sent; a new window restarts it, and so does any change ahead of the
history — `context=`, `tools=`, a new folder — so a date in `context=` restarts it once a
day.

A history the model cannot take is refused by its vendor: the adapter raises the vendor's error,
`engine.run` raises it as it is, and the run is recorded as far as it got, with its `error` — the
tool calls it already made included. A model that runs out of room while it answers ends the run
with the text it had, recorded like any other with `run.stop_reason` `"max_tokens"`; a tool call
it had begun is answered `NotRun` and never run, so the conversation still passes back. The
window's `budget` is what keeps either from happening.

To measure on a long conversation, tag its runs with it — `tags={"conversation": "dana-1"}` — and
`refinery.replay("acme", candidate, agent="support", tags={"conversation": "dana-1"})` replays them
oldest first, each on what it was sent, with `start=0` and so without `History`. Pass
`conversation=lambda run: Conversation(pool, run.tags["conversation"])` and each replays with the
window it was given and `History` over what came before it, never past its own input.

### Decision agents

A decision model is a discriminative model: given a state and questions whose answers you define,
it returns the probability of each answer — as a generation model is generative, producing text.
What you use the answers for is yours: classify, score, check, extract among candidates, annotate,
grade, route. A decision agent is a set of questions, asked together in one call:

```yaml
triage:
  kind: decision
  model: judge
  description: Says how urgent and how severe a message is
  instructions: triage/instructions.md
  questions: [triage/*.json]              # triage/urgent.json, triage/severity.json
```

`questions:` lists paths or globs from the folder root that reach its question files. A question is
`<id>.json`, its id the file's stem: an object with its `type` — `check`, `score` or `choice` — its
`instructions`, and what its answers mean. A `check` may say with `yes` and `no` what each answer
means; a `score` lists its `levels` in order, two or more; a `choice` maps its `options`, two or
more, each key to what it means:

```json
{
  "type": "check",
  "instructions": "Does the message need a person right now?",
  "yes": "The service is down or data is at risk",
  "no": "A question, a request, or a problem with a workaround"
}
```

```json
{
  "type": "score",
  "instructions": "How severe is the problem?",
  "levels": ["Can wait days", "Today", "Within the hour", "Outage"]
}
```

The instructions, what an option means, a level, `yes` and `no` may each be an object or a list
instead of a string, and an option given `null` means no more than its key:

```json
{
  "type": "choice",
  "instructions": {
    "question": "Which team should handle this message?",
    "focus": "The main request, not every topic the message mentions"
  },
  "options": {
    "billing": {
      "what": "Charges, invoices, refunds",
      "not_for": "Delivery or account access",
      "examples": ["I was charged twice", "My refund never came"]
    },
    "orders": {"what": "Delivery, cancellation, returns", "examples": ["My package never arrived"]},
    "other": null
  }
}
```

Each file is checked when the engine loads, so a mistake is caught before any run: a file that is
not a JSON object, a key a question does not take (`tree: agents/triage/urgent.json has the unknown
key 'option'; a question takes type, instructions, options, levels, yes and no`), a key of another
type, such as `levels` on a choice (`tree: agents/triage/team.json is a choice question holding
'levels'; a choice takes options, a score levels and a check yes and no, so drop 'levels' or change
its type`), instructions missing or empty, a choice with fewer than two options or a score with
fewer than two levels is a `ConfigurationError` naming the file. So are a pattern that reaches no
question file (`compiled: agent 'triage' lists the questions 'triage/*.jsn', which reach no question
file; write a path or a glob from the folder root, such as triage/*.json`), two questions with one
id, and a question in markdown, `<id>.md`, which says to write it as `<id>.json`. A decision model is
sent no prompt: the instructions file is where the agent's description can live, in its frontmatter,
and what the people who keep the agent need to know.

Run it with a state: a string, an object or a list.

```python
engine = Engine(root=Path("agents"), providers=providers)

run = await engine.run("triage", scope="acme", input={"message": text, "plan": "enterprise"})
run.answers["urgent"].probability     # 0.91
run.answers["severity"].score         # 3.0
run.answers["severity"].confidence    # 0.87
```

Every question is answered in the same call, in parallel and in isolation: adding a question
changes neither the others' answers nor, noticeably, the latency. An engine with only a decision
model needs no `Tool`s; the folder's generation agents are the ones it cannot run.

Some answer spaces live in your data: the orders of the customer who wrote, the plans of this
tenant. A script beside the question fills it: `<id>.py`, a function `run(input, call)`
that returns the fields it fills — `{"options": {...}}` for a choice, `{"levels": [...]}` for a
score — merged over `<id>.json`, which leaves them out. The JSON fixes the rest, its `type` always:
a script never changes what kind of question it fills, nor anything its JSON sets, so every run
asks the same question under one id and only what the script fills differs between tenants:

```yaml
matcher:
  kind: decision
  model: judge
  description: Says which of the customer's orders a message is about
  instructions: matcher/instructions.md
  questions: [matcher/order.json]         # matcher/order.py beside it fills it
```

```json
{"type": "choice", "instructions": "Which of the customer's orders is the message about?"}
```

```python
from json import load
from os import environ
from urllib.request import urlopen

ORDERS = environ["ORDERS_URL"]


def run(input, call):
    url = f"{ORDERS}/{call['scope']}/customers/{input['customer']}/orders"
    with urlopen(url, timeout=10) as reply:
        return {"options": {order["id"]: order["summary"] for order in load(reply)}}
```

```python
async with Engine(Path("agents"), providers=providers) as engine:
    run = await engine.run("matcher", scope="acme", input={"message": text, "customer": "c-17"})
    run.questions["order"].options      # {"A-1001": "Standing desk, shipped", ...}
    run.answers["order"].choice         # "A-1001"
```

The script runs in the decision agent's worker, as a tool's does ([Tools](#tools)), before the
decision model is called. `input` is the run's state as JSON — the text, the object or list as
given, or the parts as their JSON, media `data` in base64 — and `call` holds the run's `scope`,
`run_id`, `agent` and `tags` and the `question` it fills, so what it reads is the tenant's. The scripts of
a run's questions run at once, each under `tool_timeout`, and the question each fills is checked as
a file is. A script that raises (`WorkerError: engine: order.py of decision agent 'matcher' could
not fill question 'order': KeyError: 'customer'`), runs past its deadline (`TimeoutError: workers/pool:
order timed out after 30s`), returns anything but an object, returns a key its JSON sets
(`ModelError: engine: order.py of decision agent 'matcher' returned 'type', which question 'order'
sets in order.json; a script returns only what its question's JSON leaves out, such as {"options":
{...}}`) or one its type does not take, or fills a question that does not fit (`ModelError: engine:
question 'order' of decision agent 'matcher' does not fit once order.py fills it: the question:
models: a choice needs at least two options`) ends the run before the decision model is called,
recorded with its `error` as a failed decision call is, and with only the questions that need no
script. Otherwise the record keeps each question as it was asked, the filled options included, and
`fingerprint.questions` follows them: two tenants' runs of the matcher above ask the same question
with different options, under two question fingerprints and one agent fingerprint. A
question's script imports the standard library, installed packages and the helpers of its folder,
the `_name.py` files beside it, at the top of the script, and is read when the engine loads, as a
tool's is, so `fingerprint.agent` covers it. A `.py` without its `.json` in a folder `questions:`
reaches, one with no `run` at its top level, an import of a script beside it that is no helper, and
a helper named like a standard-library module, `_stat.py`, are a `ConfigurationError`, as for a tool
([Tools of the folder](#tools-of-the-folder)); a script or a helper may link anywhere inside the
folder root. A route goes by a question
whose options its JSON fixes, never by one a script fills
([Both kinds together](#both-kinds-together)).

Some answer spaces are known only when the run starts: the children of the node a walk chose last,
the candidates a search returned, the skills an agent has today. Those questions are composed when
the run starts, by a model or by a runtime of yours: a generation agent composes them
in a `Delegate` call ([Nesting](#nesting)), and a runtime of yours composes them in a
call to `engine.tool`. They are ids to `Choice`, `Score` or `Check` — or anything
with their fields — written as JSON, and they are asked with the folder's,
in the same call. A question known in advance is a file `questions:` reaches, and one whose
answers your data holds keeps its script beside it. The record keeps every question asked, and `run.runtime` names
the ids that came at run time:

```python
asking = engine.tool("triage", scope="acme")
sent = {
    "input": text,
    "questions": {
        "product": {
            "type": "choice",
            "instructions": "Which product is the message about?",
            "options": {
                "desk-120": {"what": "Standing desk, 120 cm"},
                "desk-160": {"what": "Standing desk, 160 cm"},
            },
        }
    },
}
result = await asking.dispatch("triage", sent)
result["answers"]["product"]["choice"], result["answers"]["urgent"]["probability"]
```

A runtime question holds what a question file holds — a choice needs two options, a score two
levels — and one that does not, an id the run already asks, or questions sent to a generation
agent is an `InputError`. A decision agent may have no `questions:` at all and ask only what each
run brings; a run with nothing to ask is an `InputError`.

Every run of a decision agent asks the questions of its entry, or of its mode, and beside them the
questions the run brings; another set of questions, asked on other runs, is another decision agent
or another mode, under whose name its records and its datasets stand apart.

`run.fingerprint.questions` is a short sha256 over the questions the run asked — the folder's, those
its scripts filled and the runtime ones alike — as JSON ordered by question id, each question's
fields as they were asked and its options and levels in their order, since the order can change
which answer a model favours. Two runs that asked the same questions share it; a changed word,
option or order changes it. A generation run's is `None`. With `fingerprint.agent` for the entry,
it tells two versions of a decision apart even when the questions came at run time.

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

`orchestration.delegate_to` on a generation agent's entry gives it a `Delegate` tool listing those
agents by name, description and kind:

```yaml
support:
  kind: generation
  model: strong
  instructions: support/instructions.md
  orchestration:
    delegate_to: [triage, billing]        # one way: neither reaches support back unless it says so
    max_depth: 2                          # levels of delegation below support
    max_parallel: 2                       # Delegate calls of one run at once
```

Each name is an agent of the file, generation or decision; a name the file does not hold, a voice
agent and an agent that needs a mode are a `ConfigurationError` when the engine loads. The model calls it with an agent and an input; the child runs
under the same scope and its reply — text for a generation agent; for a decision agent its answers,
and the routed run's reply when it routed, as `engine.tool` returns them — comes back as the tool
result. Its input comes from the model and is never read for `/name`, and the child takes no skill.
The child's record is attached to the parent's `children`, and the
step that called it carries the child's `run_id`. A decision agent used this way is how a
generation agent asks a fast, calibrated question mid-run: is this destructive, which skill
applies, is the user done.

The model can also give a decision agent `questions` it composes itself — ids to questions of the
three types, written as JSON, structure included — asked beside the agent's own in the same call. A
question that occurs to the model mid-run is answered with probabilities instead of guessed at, and
a decision agent with no questions of its own is worth delegating to. The model sends:

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

A decision child records its runtime questions, as any run does, and the step keeps what the model
sent. `Delegate` offers `questions` only when a target is a decision agent, and its schema describes
the question object once, for all three types. Questions for a generation agent, a question that
does not fit, an id the target already asks, a decision run with nothing to ask, or a question
script that fails reaches the model as a tool error it can read and act on, and the run goes on.

A run records the call it worked for in `call`, the same `CallInfo` its `RunStarted` carries, so a
run a tool of yours started links back to the run that called it.

Depth is capped (`Engine(max_depth=)`, default 5, and the entry's `orchestration.max_depth` under
it, as [Limits](#limits) says). Within it an agent may name itself in `delegate_to` and hand a
sub-problem to a fresh run of itself, and two agents may call each other. A route is different: it
forwards the same input, so a route into an agent that already decided it — in the same chain of
routes, since the last `Delegate` — is refused with `DelegationLoopError`; a `Delegate` starts a new
input, so the agents above it do not count. Through `Delegate` an error reaches the model as a tool
error and the run continues; through a route it raises.

### Both kinds together

The two kinds call each other. A generation agent asks a decision agent through `Delegate`, as in
[Nesting](#nesting), and a decision agent can hand its run to another agent by one of its answers:
a route. One choice question it asks, one table from its options to agents, an optional confidence
floor, under the router's `orchestration`:

```yaml
frontdesk:
  kind: decision
  model: judge
  description: Routes an incoming message
  instructions: frontdesk/instructions.md
  questions: [frontdesk/intent.json]
  orchestration:
    route: {by: intent, to: {billing: billing, technical: support}, min_confidence: 0.6}
support:
  kind: generation
  model: strong
  instructions: support/instructions.md
  orchestration:
    delegate_to: [triage]
billing:
  kind: generation
  model: strong
  instructions: billing/instructions.md
```

```json
{
  "type": "choice",
  "instructions": "Which department should handle this message?",
  "options": {
    "billing": "Payments, refunds, invoices",
    "technical": "The product misbehaves",
    "other": "Anything else"
  }
}
```

When the chosen option maps to an agent and confidence clears the floor, that agent runs with
everything the run was given — the same input, parts included, `history`, `context`, `expect`,
`tools`, `workspace`, `on_event`, `stop`, `inbox` and `tags` — and the routed run is attached:

```python
engine = Engine(Path("agents"), providers=providers)

run = await engine.run("frontdesk", scope="acme", input=message, history=history)
run.answers["intent"].choice            # "technical"
run.routed_to.agent                     # "support" — the generation agent that answered
run.routed_to.children[0].agent         # "triage" — the decision agent support consulted mid-run
```

A route is by a choice question the router asks from its `questions:`, whose options its `.json`
fixes, since the route maps them to agents: neither a runtime question nor one a script fills can
drive it, and `by` naming a scripted question is a `ConfigurationError` (`compiled: agent 'matcher'
routes by 'order', whose options come from matcher/order.py; route by a question whose options are
fixed`). A router's runtime questions are its own, asked beside `intent` and not passed on. A route
entry is an agent's name, as in `to` above, and a decision agent a route reaches must have
`questions:` of its own; an option the question does not have, an agent the file does not hold, a
voice agent and an agent that needs a mode are a `ConfigurationError` when the engine loads, since
that folder cannot work. A router with modes routes by the questions of each mode. A decision agent
reached through `Delegate` needs no questions of its own, since the model can bring them.

One message, four model calls, one record. Every agent runs on the model its entry names, built
when its run starts, so a folder of generation agents needs generation providers only, and one of
decision agents decision providers only; a provider that builds the wrong kind fails the runs of
the agents on it, each before its first model call, and leaves every other agent as it was.

### Voice agents

A live model hears and speaks itself: it takes a person's audio, answers in audio and runs its own
turns, for as long as the person stays. It is no model in a loop the engine drives — its prompt is
one string set when the session opens, it calls its tools as it talks, and it reports no tokens — so
a conversation on it is a third kind of agent, `kind: voice`. Voice models come in three types, live,
speech and transcription; a voice agent runs on a live model, while a conversation cascaded through a
transcription model, a text model and a speech model is a generation agent whose session holds the
other two.

```yaml
voice:
  kind: voice
  model: voice-live
  description: Talks with a person and consults another agent for facts
  instructions: voice/instructions.md
  tools:
    include: [consult]
  modes:
    duplex:
      instructions: voice/instructions_duplex.md
```

What it takes follows from what a live model is: `kind`, `model`, `description`, `instructions`,
`worker`, `tools` and `modes`. It has no loop that opens a file or counts calls, so `files`, `skills`,
`orchestration`, `workspace`, `questions` and `max_model_calls` are a `ConfigurationError` naming the
key, as is a tool its sections would grant, such as `Read`, in `tools.include`. `engine.talk` runs it
and refuses any other kind; `engine.run`, `engine.tool`, `delegate_to` and `route` refuse it. It
reaches other agents through its tools, such as a consult `engine.tool` builds.

The engine declares the agent, resolves its mode, gates its tools and records the run; the session
that moves the audio is yours — rfnry-senses' `Session`, or any other — and neither library imports
the other. One function of yours joins them:

```python
async def session(start: VoiceStart) -> Spoken:
    talking = Session(transport, live=start.model, instructions=start.instructions)
    await talking.start(start.tools)
    return Spoken(transcript=await until_it_ends(talking))


consult = engine.tool("support", scope="acme", name="consult")
run = await engine.talk("voice", "acme", session, mode="duplex", tools=consult,
                        models={"voice-live": live_model}, approve=gate)
```

`start.model` is the live model the entry names, here one built for this session and handed over by
`models=`, as any run takes one. `start.instructions` is the mode's instructions file, its
frontmatter left out, with `context` after it; a host that fills in per-session values renders them
from that text. `start.tools` is one `Tools`, and every call through it passes `approve` and the
deadline, becomes a `Step`, and is the same tool call a generation run makes; a bare name the run
does not give fails before the session opens. Cancelling a call, because the person changed what
they asked, cancels its body and the session goes on; a declared tool's plain `def run` goes on
holding its worker's thread until it returns, since a thread cannot be cancelled, and its step reads
`stopped` all the same. The record is a `Voice`: the transcript the session returns, the steps, the
mode, the model and the fingerprint. Its `usage` is `None` when the session has none, since live
models report no tokens, and that is not zero cost. A failed session is recorded with its `error`; a
cancelled one is not, as for any run.

`refinery.run` reads the record back equal to what `talk` returned, `usage` `None` included, and
`refinery.outcome` takes what happened. `refinery.review` takes a stance and a comment, and a
correction is an `InputError`, since a conversation has no one reply to replace and no question to
answer. `refinery.export` leaves voice runs out of every dataset, and `refinery.replay` refuses
one, as a pair with `error` set, since a person's side of a conversation cannot be sent again.

### Orchestrating

The library runs agents and records what they did. The order they run in is yours or the model's: your
code decides in a workflow, and the model decides when it delegates, routes or calls a tool of yours.
Start with a workflow in code, where you know the steps, and add autonomy only where a fixed step
breaks. There is no graph, planner or queue here to configure; what follows is what you write on
top of the public surface, and every block runs in the test suite.

A workflow is code. Tag every run with the workflow's id, so `run_tree(tags=)` and
`delete_runs(tags=)` reach all of it:

```python
workflow = {"workflow": "refund-17"}

billed = await engine.run("billing", scope="acme", input=message, tags=workflow)
answered = await engine.run("support", scope="acme", input=billed.reply, tags=workflow)

parts = await asyncio.gather(
    *(engine.run("support", scope="acme", input=part, tags=workflow) for part in (message, text))
)

routed = await engine.run("frontdesk", scope="acme", input=message, tags=workflow)
routed.routed_to.agent                  # the department the router chose
```

One run after another, runs side by side, and a route: the three patterns a workflow is made of.

A budget across a tree is an `on_event` that counts `RunStarted`, which every generation run in the
tree sends, and presses a `Stop` when the count runs out. Decision runs send no events, so they are
not counted:

```python
class Budget:
    def __init__(self, runs):
        self.left, self.stop = runs, Stop()

    async def __call__(self, event):
        if isinstance(event, RunStarted):
            self.left -= 1
            if self.left < 0:
                self.stop.after_step()

budget = Budget(50)
run = await engine.run("support", scope="acme", input=message, on_event=budget, stop=budget.stop)
```

Two agents that write `guide.md` into one workspace overwrite each other. Give each run a copy of
its own, have it claim work before it does it, and merge when a child comes back. A `Backend` is
yours, so the copy is a directory per run, named by `current_call().run_id`: inside a tool or a
backend, `current_call()` is the call the running run is making. `RunStarted.call` names the run that
started a child, and the parent's `ToolResultReady` for its `Delegate` step carries
`step.child_run_id`, which is where you merge. How a directory is copied and merged (worktrees, a
copy, a container per run) is yours too:

```python
from pathlib import Path

from rfnry_engine import Local, RunStarted, Tool, ToolResultReady, Tools, Workspace, current_call

class PerRun:
    def __init__(self, base):
        self.base, self.parent, self.merged = Path(base), {}, []

    def here(self):
        call = current_call()
        place = self.base / (call.run_id if call else "main")
        place.mkdir(parents=True, exist_ok=True)
        return Local(place)

    async def spawn(self, command, env): return await self.here().spawn(command, env)
    async def read(self, path, limit): return await self.here().read(path, limit)
    async def write(self, path, data): await self.here().write(path, data)

    async def on_event(self, event):
        if isinstance(event, RunStarted) and event.call:
            self.parent[event.run_id] = event.call.run_id
        if isinstance(event, ToolResultReady) and event.step.child_run_id:
            self.merged.append(event.step.child_run_id)

claims = {}

async def claim(args):
    me = current_call().run_id
    return {"claimed": claims.setdefault(args["item"], me) == me, "by": claims[args["item"]]}

backend = PerRun("work")
tools = Tools([Tool("claim", "Take an item of work for this run", body=claim, input_schema={
    "type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]})])
run = await engine.run("lead", scope="acme", input=message, tools=tools, on_event=backend.on_event,
                       workspace=Workspace(backend, scope="acme"))
```

Work in the background is two tools of yours: one starts `engine.run` in a task you keep, one reads
the task. You cancel these tasks and bound how many there are. The run they start records the call
that started it, so its record links back to the run that asked for it:

```python
jobs = {}

async def start(args):
    jobs[str(len(jobs))] = asyncio.create_task(
        engine.run("support", scope="acme", input=args["input"])
    )
    return {"job": str(len(jobs) - 1)}

async def check(args):
    job = jobs[args["job"]]
    return {"running": not job.done(), "reply": None if not job.done() else job.result().reply}

tools = Tools([
    Tool("start", "Start support on an input", body=start, input_schema={
        "type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}),
    Tool("check", "Read a job started earlier", body=check, input_schema={
        "type": "object", "properties": {"job": {"type": "string"}}, "required": ["job"]}),
])
outer = await engine.run("boss", scope="acme", input=message, tools=tools)
```

Whether a team is worth it is measured, not assumed. Run the team and a single agent on the same
input and compare what each spent over the whole tree, summed over `children` and `routed_to`. A
run a tool of yours started as a consult is not a child: it links back to its caller through its
`call`, and `spent` does not count it. Over the holdout, with the grades beside the cost, the same
comparison is a replay of each, as [`evaluation.md`](evaluation.md#replaying-the-holdout) shows:

```python
def spent(run):
    below = run.children if isinstance(run, Generation) else [run.routed_to] if run.routed_to else []
    usage, calls = run.usage, run.calls if isinstance(run, Generation) else 1
    for child in below:
        more, made = spent(child)
        usage, calls = usage + more, calls + made
    return usage, calls

team, team_calls = spent(await engine.run("support", scope="acme", input=message))
alone, alone_calls = spent(await engine.run("billing", scope="acme", input=message))
more = team_calls - alone_calls, team.output - alone.output    # calls and tokens the team adds
```

### Workspace

> `Workspace` gives the model a real shell, and **a shell is not confined by its directory**.
> `Local` runs each command as a process of your service: without a sandbox in front of its shell, a
> command can do anything the service's user can — read every other tenant's workspace, read the
> service's configuration and any secret on disk, reach the database and the network, delete things.
> A document an agent reads can tell it to try. Put a sandbox in front of the shell, as below, or
> pass a backend that runs commands somewhere else, and keep the agents folder outside the
> workspace.

A workspace is where an agent writes code, runs it, reads what happened and tries again — the loop
a coding assistant runs in a developer's terminal. An agent opts in with one line in its entry,
and a run supplies the workspace: the library's tools over a backend of yours, which runs the
commands and holds the files.

```yaml
coder:
  kind: generation
  model: strong
  description: Writes and runs code to finish the work.
  instructions: coder/instructions.md
  workspace: true
```

```python
from rfnry_engine import Local, Workspace

local = Local("/srv/tenants/acme/work", shell=[
    "bwrap", "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
    "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev",
    "--tmpfs", "/tmp", "--bind", "/srv/tenants/acme/work", "/srv/tenants/acme/work",
    "--chdir", "/srv/tenants/acme/work", "--unshare-all", "--die-with-parent", "--new-session",
    "bash", "-c",
])
workspace = Workspace(local, scope="acme", timeout=600)

run = await engine.run("coder", scope="acme", input="make the failing test pass", workspace=workspace)
```

`Local` runs each command on this machine, in the directory, given to the words of `shell` — here
bubblewrap, which builds each command a world of its own from the host: the system's programs
read-only, this tenant's directory writable, an empty `/tmp`, and nothing else — no other tenant,
no service configuration, no network — and kills every process when the command ends. The model
sees none of this: it has the same tools, and its command runs inside.

A sandbox per command, as here, gives each command its own network, so a server a job starts
answers only commands inside the same sandbox; to start a server in one message and reach it in
the next, keep one sandbox per workspace, such as a container kept for the workspace's life behind
a backend of yours, below.

The agent then has five more tools. `Shell(command, background=false)` runs a command and returns
`exit_code`, `stdout`, `stderr` and `timed_out`; when the command ends, anything it left running is
stopped. With `background=true` the command starts as a job — a server, a watcher, a long build —
and the call returns its id at once, `{"job": "j1"}`. `Job(id, wait=0, stop=false)` returns what the
job printed since the last read, stdout and stderr in the order they arrived, whether it is
`running`, and its `exit_code` once it has ended; `wait` is seconds to wait for the job to end
first, up to the workspace's `timeout`, returning as soon as it does — a model waits for a build
instead of asking again and again — and `stop=true` stops a job still running after the wait.
`Jobs()` lists the jobs: id, command, when it started, running or exit code. `Write(path, content)`
writes a file exactly, and `Edit(path, old, new, all=false)` replaces one exact occurrence, or with
`all=true` every one — helpers that send only what changes and leave a clean step in the record; the
shell can do both too. `Read` and `Load` take `where="workspace"`: `Read` for a text file of the
workspace, with `offset` and `limit` selecting lines, `Load` to hand the model a file as it is, so
an agent looks at the chart it just plotted. Both go through the backend. `LS`, `Grep` and `Glob`
stay on the folder; a path the folder does not hold, given to `LS`, `Read` or `Load` of a run with a
workspace, is answered with the way to the workspace: `tools/builtins: not a directory of your folder:
task_folder; Shell lists your workspace`. Each command starts fresh
in the directory, so a `cd` or a variable does not carry over, and the tool's description tells the
model so, along with its limits.

Jobs belong to the workspace, not to the run: they keep running after the run that started them,
after a run that failed and after `stop.now()`, until the model stops them or you close the
workspace — so an agent starts a server in one message and tests against it in the next. Every run
given the workspace, delegated and routed runs included, sees the same jobs, numbered `j1`, `j2`, …
for the workspace's life. A job that ended stays listed until its last output is read. Each job is
read in the background until it ends or the workspace is closed, so it never stalls on a full pipe;
what it printed waits for `Job`, kept to its first and last `output_bytes // 2` bytes. The list of
jobs is not in the system prompt, which would change with every job and cost the cached prefix of
every conversation on the workspace; the model asks `Jobs()`.

A frontend shows the same jobs with `workspace.jobs()`, a list of `JobInfo(job, command, started,
running, exit_code)`, and reads one with `workspace.output("j1")`, everything the job printed so far
in the same cap. Neither changes what the model's `Job` reads next, and a job that ended leaves the
list once the model reads its last output. An unknown id is an `InputError`.

Close the workspace when you are done with it — `async with Workspace(…) as workspace:` or `await
workspace.close()` — and every job is stopped; the workspace stays usable. A workspace never closed
leaves its jobs to the backend: on `Local` they may outlive your process, unless the sandbox ties
them to it, as bubblewrap's `--die-with-parent` does. A workspace and its jobs live in one event
loop, and a job whose loop ends is stopped: a service keeps a conversation's workspace in one of its
processes and on one loop.

`Workspace(backend, scope=, timeout=, output_bytes=, bridge=, tree=)`:

| argument | default | what it does |
| --- | --- | --- |
| `backend` | | where commands run and files live: `Local`, or one of yours |
| `scope` | | the tenant the workspace belongs to |
| `timeout` | `120` seconds | the limit on each `Shell`, `Job`, `Write` and `Edit` call; a `Shell` command still running is stopped — on `Local`, sent SIGTERM, then killed a second later with everything it started — and `Shell` returns what it printed with `timed_out: true` |
| `output_bytes` | `30_000` | the cap on `stdout` and on `stderr`, each keeping its first and last half |
| `bridge` | `True` | lets a command call the run's tools, below; `False` turns it off |
| `tree` | `None` | where the agent tree is mounted in the sandbox, as commands see it, below |

`Local(root, shell=, env=)`:

| argument | default | what it does |
| --- | --- | --- |
| `root` | | the directory commands start in and files live in |
| `shell` | `["bash", "-c"]` | the words each command is given to, last: a sandbox goes in front of the shell, and the first word must be on `PATH` |
| `env` | `PATH`, `HOME` (the directory), `LANG` | the only environment commands see, so the service's own keys never reach them |

The workspace's `timeout` replaces `Engine(tool_timeout=)` for `Shell`, `Job`, `Write` and `Edit`,
so a ten-minute build and a thirty-second API call live in the same engine. A `shell` whose first
word is not on `PATH` is a `ConfigurationError` that says so; pass `shell=["sh", "-c"]` on a system
that has only a POSIX shell, such as Alpine.

On `Local`, `Write`, `Edit`, `Read` and `Load` stay inside the directory: they refuse a path that
leaves it, symlinks included. What the system refuses reaches the model with its reason and the
model's own path, never the path on your machine: `tools/workspace: cannot write notes.txt/x: Not a
directory`. `Shell` reaches what the process running it can reach, so the real boundary is the
sandbox in front of the shell, or the machine. `Local` reads and writes files in a thread, which a
cancel cannot stop, so a `Write` or `Edit` cancelled while it writes returns once the file is
written, and one past the workspace's `timeout`, like a workspace `Read` or `Load` past
`tool_timeout`, returns its `TimeoutError` only once its thread has ended.

The workspace root and the folders above it must be out of the model's reach: `Local` checks every
call against the folder it was made with, but a `Local` made later on a path the model could rename
follows the rename. The bubblewrap recipe binds the root itself, so a sandboxed model never sees
its parent.

`Local` refuses a root that is no longer the folder it was made with, by its directory inode number.
On a filesystem whose directory inode numbers are not stable — overlayfs without `xino`, FUSE
without `use_ino` — that check refuses a valid root, so such a filesystem is not supported as a
workspace root.

Binding the whole system read-only (`--ro-bind / /`) is not enough: a command could still read
every other tenant's directory and every secret on disk. Mount only what commands need — the
recipe above suits a system whose programs live under `/usr`; add a read-only `/etc` path when a
tool needs one — and give each tenant its own workspace, each with its own sandbox. bubblewrap
limits nothing: one command can take all the memory, CPU or processes of the machine from every
other tenant. Where the service's user has a systemd user session,
`systemd-run --user --scope -p MemoryMax=1G -p CPUQuota=100% -p TasksMax=256 --quiet` before
`bwrap` runs each command in a scope with those limits; `env=` then needs `XDG_RUNTIME_DIR` too,
which `systemd-run` uses to find the session.

Docker can be the sandbox too. A container per command is a `shell=` of `docker run --rm` with the
directory mounted at its own path and `timeout -s KILL <seconds>` before the command inside: the
`docker` client passes SIGTERM on to the container, but SIGKILL stops only the client, so the limit
inside is what ends a container whose command ignores SIGTERM. `env=` is then the client's
environment, not the command's — the command's own variables go in with `-e`: a workspace with
`tree=` needs `-e RFNRY_TREE`, and one with the tool bridge, below,
`-e RFNRY_TOOLS -e RFNRY_TOOLS_KEY` — and the client reads
its configuration from `$HOME/.docker`, which `Local`'s default `HOME` puts in the workspace, where
a command could write a `.docker/config.json` the next command's client obeys; point
`DOCKER_CONFIG` outside the workspace. One container per workspace, entered with `docker exec`,
needs a backend of yours: signals do not pass through `docker exec`, so its `kill()` must reach the
command's processes inside the container itself — a tag in their environment that everything they
start inherits, say, whose processes it signals.

The folder in the shell. `Workspace(local, scope="acme", tree="/srv/agents")` says where the agents
folder is mounted in the sandbox, as commands see it. The commands of a run then get `RFNRY_TREE`,
the folder as `posixpath.normpath` writes it, so `/srv/agents/` is `/srv/agents`. The `Shell`
description tells the model that the folder is at `$RFNRY_TREE`, read-only, with its files, skills
and scripts to read and run, and that each folder its `files.include` names is at
`$RFNRY_TREE/<name>`. A skill's script is run as `python "$RFNRY_TREE/support/skills/plot/run.py"`.
The sandbox mounts the folder read-only at the same path:
`--ro-bind /srv/agents /srv/agents` for bubblewrap, `-v /srv/agents:/srv/agents:ro` for Docker, and
for an unsandboxed `Local` the folder's own path. Commands started with `background=true` get the
variables too; a `Shell` that code dispatches outside a run gets none. Cautions:

- the shell sees every file under the mount, `rfnry.yaml`, hidden ones, those `files.gitignore`
  leaves out and whatever `files` leaves out included, which the file tools refuse;
- the fingerprint covers what the agent reaches, not what a command reads;
- a mount of the whole folder shows every agent's files; mount less to hide them;
- `refinery.replay(candidate, workspace=)` needs a workspace whose sandbox mounts the candidate's
  folder.

Showing a command. With `on_event`, a foreground `Shell` command sends a `CommandOutput` for what it
writes, as it arrives (chunks that queued while the handler was busy arrive joined, one event per
stream, and one more at each gap, since an event never joins across one), with its `call_id`, the
`stream` (`"stdout"` or `"stderr"`) and the `text`, decoded as UTF-8 per stream so a character split
across two writes arrives whole. A frontend can show a long build as it runs instead of after it.
The model's view is unchanged: it reads the same capped `stdout` and `stderr` in the `Shell` result,
and a handler never slows the command. The events wait in a queue of 1 MiB per command; while a
handler is busy past that, chunks are dropped from the events, as is a single read of more than 1
MiB even while the handler is idle, and once a stream overflows its output is dropped until the
handler catches up and takes what is queued; the next event of that stream starts with `[... <n>
bytes not shown ...]`, counting all of it. Every event of a command comes before the
`ToolResultReady` of its call, in order per stream — `stdout` and `stderr` are two pipes, with no
order between them — and a handler that raises fails the run once the command has ended. A command
started with `background=true` sends none.

Calling tools from code. A command can call the run's tools itself, so a script that looks up fifty
orders, or checks every row of a table against a rule, makes its calls in a loop, and the model reads
only what the script printed instead of making one call a turn. For each foreground `Shell` command
of a run, the engine listens on a Unix socket in the workspace and gives the command its path and a
key of its own:

```
 model                    engine (your process)                           sandbox (this tenant's shell)
   │ Shell("python solve.py")    │                                                 │
   │────────────────────────────▶│ 1. listen on <workspace root>/.rfnry-<8 hex>.sock
   │                             │ 2. spawn the command with RFNRY_TOOLS=<that path> ──▶ python solve.py
   │                             │◀── {"key":…,"tool":"search","args":{"query":"heat equation"}}   (connect, one line)
   │                             │ 3. the run's call path: events, approve, deadline, dispatch, record
   │                             │─── {"result":[ … ]} ─────────────────────────────────────▶│ (one line, close)
   │◀── Shell result ────────────│ 4. command ended: socket closed, removed if its path still holds it
```

A client is a few lines in any language; in Python:

```python
import json, os, socket


def call(tool, **args):
    with socket.socket(socket.AF_UNIX) as s:
        s.connect(os.environ["RFNRY_TOOLS"])
        request = {"key": os.environ["RFNRY_TOOLS_KEY"], "tool": tool, "args": args}
        s.sendall(json.dumps(request).encode() + b"\n")
        return json.loads(s.makefile(encoding="utf-8").readline())


order = call("lookup_order", order_id="A-1001")
print(json.loads(order["result"])["status"])
```

A connection makes one call: one line of JSON in, `{"key": …, "tool": …, "args": {…}}`, and one
line back, `{"result": …}` or `{"error": …}`, then it closes. `result` is what the model would read
for the call: its text as the step records it — a tool that returns an object or a number gives its
JSON text, so a script runs `json.loads` on a result it expects to be JSON — or, for a tool that
returns parts, their JSON list, media `data` in base64. `error` is the text the model would read:
`NotApproved: lookup_order was not approved`, `TimeoutError: engine: lookup_order timed out after
30s`, a tool's own error. A line that is not one JSON object with a string `tool` and an object
`args`, a line over 1 MiB, and a request without the command's key are answered with an `error` that
starts with `tools/bridge:`, and run nothing. A command makes at most 10,000 calls and sends at most
2 MiB of request lines in all: every request that reaches the run's tools is a call, a name the
socket does not carry included, and every line read whole counts toward the 2 MiB as it is read,
before it is parsed, one the bridge refuses included. Past either, every further request is answered
`{"error": "tools/bridge: this command made its 10000 calls; no more are answered"}` or
`{"error": "tools/bridge: this command sent its 2097152 bytes of calls; no more are answered"}`, so
a script with more to do splits it over several commands, as the `Shell` description tells the
model. The budgets are each command's own: commands a run has going at once each get theirs, so
what a run's commands send in all grows with how many it runs at once. A call whose `args` nest
deeper than 128 levels of lists and objects, `args` itself the first, is answered
`{"error": "tools/bridge: a call's args nest deeper than 128 levels"}`. The depth is read from the
line before it is parsed, its strings aside and the request itself the first level, so any field
nested past 129 levels gets this error, not only `args`. A request is checked in this order and
answered with the error of the first check it fails: a line over 1 MiB, which is not read whole
and counts toward neither budget; then, the line counted toward the 2 MiB, the 10,000 calls and
then the 2 MiB; then the depth; then a line that is not one JSON object; then the command's key;
then a string `tool` and an object `args`. So a line over 1 MiB gets the size error whatever else
is wrong with it, a line past a budget gets the budget's error even when it is nested too deep or
malformed, and a line nested past 129 levels gets the depth error even when its key is not the
command's or it is no JSON at all. None of these runs anything or leaves a step or an event, and a
connection closed at the cap of 64 below, refused before its line is read, counts toward neither
budget.

The socket carries every tool the run offers the model — the builtins, `History`, the folder tools,
the run's `tools=` — except the workspace's own (`Shell`, `Job`, `Jobs`, `Write`, `Edit`) and any
tool named `Output` or `Delegate`, the engine's or one of the run's `tools=`, which the model still
calls. A name the socket does not carry gets the model's error for an unknown tool,
`InputError: tools: unknown tool 'Shell'`, and leaves no step and no event. Connections are answered
as they come, so calls a script makes at once — from threads, or with `asyncio` — run at once, as a
reply's calls do, up to 64 at once per command: a connection that arrives while 64 are in flight is
closed at once, unread, after an `{"error": "tools/bridge: at most 64 calls at once; wait for one to
finish"}` line. The script may read that line or see a broken pipe or a connection reset instead, and
should retry on any of them. A `Shell` command of a delegated or routed run calls that run's tools,
under its own `run_id`.

Each call runs as the model's calls do — `ToolCallStarted`, `approve`, a slot of
`tools.max_parallel`, the deadline (`tool_timeout`, or a folder tool's own), the stop,
`ToolResultReady` — and is recorded as a `Step` with `by`, the id of the `Shell` call whose command
made it; its own id is `<Shell call id>.<n>`, n counting from 1 for each command. It finished
first, so its step sits in `run.steps` before the `Shell` step that made it, and the conversation
holds no message for it: the model reads only the `Shell` result.
`ToolCallStarted`, `ToolResultReady` and `CallInfo` carry the same `by`, so `approve` and a tool tell
a call a script made from one the model made with `current_call().by`.

A call ends with its command. When the command ends, for any reason — it exits, a stop, the
workspace's `timeout` — a call still in flight is cut as a stopped call is: its step is recorded with
`stopped` and `stopped by the user` as its result, its `ToolResultReady` is sent, and a script still
connected reads `{"error": "stopped by the user"}`. A consult that a script's call started, an
`engine.tool` passed in `tools=`, ends with the command: one still running when the command ends is
cancelled and leaves no record of its own, as no cancelled run does. The exception is a command that
`stop.now()` ended: its process is killed at once, while the consult takes the stop, as one the
model's own call started does, and keeps its record, and the `Shell` call ends after it. A further
`stop.now()` does not cut it; a cancel of the run, or its call's deadline, does. The script's step
is recorded stopped either way. A reply to a process that outlived the command, one that left its
process group outside a sandbox, is written but not waited for, and may be cut off when the socket
closes. An `on_event` that raises on a script's call fails the run once the command has ended.

The socket is the one file the engine creates itself, `<root>/.rfnry-<8 hex>.sock`, inside the
workspace you passed, for the life of one command: made just before the command starts, closed once
it has ended and its output is read, and then removed if its path still holds it — a file the
command put in its place is kept, and a socket in a directory the command made read-only stays.
Under bubblewrap it is in the bound directory at the same path, and a socket bound to a path reaches
through `--unshare-all`; under Docker it is in the directory mounted at its own path, with
`-e RFNRY_TOOLS -e RFNRY_TOOLS_KEY`. The socket can be reached only from a sandbox on the same kernel
that sees the workspace directory at the same path: bubblewrap, or Docker on Linux. Docker Desktop's
file sharing on macOS and Windows does not carry Unix sockets, so use a backend that runs the engine
beside the commands there. A container user other than the service's needs write permission on the
socket file, which takes the service's umask and a new random name for each command, so it cannot be
granted ahead of time: run the container as the service's uid, or as a user of the service's group
with the service under a umask of 002. The `Shell` description tells the model of it in those
variables, never in a path. Whether the bridge is on, and whether `tree=` is set, change that
description and so `fingerprint.tools` — rightly, since the model was offered something different —
but where a deployment puts the workspace does not.

- The bridge gives code nothing more than the model has: the model's tools but those named above,
  through the same `approve`, scope and record. What is new is speed: a script can make many calls
  quickly; `approve` still sees each one.
- Only processes that see the socket's path can connect — with a sandbox per tenant, only this
  tenant's commands — and only a request with the command's key is answered.
- The key is as private as the sandbox keeps a command's environment: a process of the same uid in
  the same PID namespace reads it from `/proc/<pid>/environ`. A sandbox per command, as the
  bubblewrap recipe above runs each one under `--unshare-all`, keeps it to that command. No
  sandbox, one container per workspace entered with `docker exec`, or anything else where commands
  share a PID namespace and a uid does not: there another command, a background job included, can
  read a running command's key and call the run's tools through its socket. `by` names the command
  whose socket was used.
- Until its command ends, each open connection holds a file descriptor and up to 1 MiB in the
  engine's process, which a sandbox's limits don't cover. The workspace `timeout` bounds how long,
  and a command has at most 64 calls in flight at once, so it holds at most 64 file descriptors and
  64 requests of up to 1 MiB each in the engine's process. A script that opens more finds its
  connection closed, and should retry.
- A command makes at most 10,000 calls, sends at most 2 MiB of request lines and nests a call's
  `args` at most 128 levels deep. Past each, a request is answered with its error before it is
  parsed — `tools/bridge: this command made its 10000 calls; no more are answered`,
  `tools/bridge: this command sent its 2097152 bytes of calls; no more are answered`,
  `tools/bridge: a call's args nest deeper than 128 levels` — and leaves no step and no event. So
  one command adds at most 10,000 steps to the run's record, the engine parses at most 2 MiB of its
  requests however they are shaped, which bounds the arguments it holds and records, and every
  argument it records nests well inside the depth the record's serializer takes. A line counts
  toward the 2 MiB as it is read, and its key is checked only once it is parsed, so no line past
  the budget is parsed, one without the key included, while one without the key inside the budget
  is parsed and then refused. So a process that can reach the socket file can make the engine parse
  at most 2 MiB of lines per command — about a 350 ms stall of the event loop at worst, as
  measured — and with junk lines spend the rest of that command's budget, so the command's own
  calls are refused from then on, as it can hold its 64 slots. Keeping the socket to the command is
  the sandbox's job, as it is for the key.
- The socket lives only while the command runs; something the command leaves running cannot call
  afterwards.

A backend gets the bridge when it has a `root` attribute: a local directory its commands see at
the same path, as `Local`'s is, where each command's socket is created. A `root` that means
anything else gets a socket created at that path, so a backend without such a directory has no
`root`. `Workspace(..., bridge=False)` turns the bridge off, and so do a backend without a `root`
attribute and a run that offers no tool it could carry: the `Shell` description then does not
mention it. A command started with `background=true`, and a `Shell` that code dispatches outside a
run's own `Shell` call, get no socket either. A command without the bridge gets no `RFNRY_TOOLS`.
A root too deep for a socket's path — about
107 bytes on Linux, 103 on macOS — runs the command without it, and its `stderr` ends with the
reason: `tools/workspace: no tool bridge for this command: AF_UNIX path too long`.

Without a sandbox, the kill reaches every process the command started except one that leaves its
process group with `setsid`; the process namespace of a container or of bubblewrap closes that gap.
Such a process keeps running, but not on the command's pipes: once the command has exited and its
output is no longer read, `Local` closes them, so it holds none of the engine's files and what it
writes there after is lost.

A backend is three methods; the library does the rest — the tools, the time limits, the output
caps, stopping what a command leaves behind:

```python
class Backend(Protocol):
    async def spawn(self, command: str, env: Mapping[str, str]) -> Process: ...
    async def read(self, path: str, limit: int) -> bytes: ...
    async def write(self, path: str, data: bytes) -> None: ...
```

`spawn` starts a command in the workspace with no input, with `env` added to its environment, and
returns at once a `Process`: its `stdout` and `stderr`, read until they end, `wait()` for its exit
code, and `kill()` to end it and everything it started. Every kill the workspace starts runs to its
end, and the call that started it waits for it: a stop or a cancel of a `Shell` call, of
`Job(stop=true)` or of `close()` returns once the kill has returned — `close()` once the jobs it
stopped have ended too — however many cancels come meanwhile. A cancel that comes while `spawn` runs
waits for it to return, then for the kill of what it started — with `background=true`, for its job
to be listed, so `close()` stops it. So `spawn` and `kill()` must return promptly. A kill counts as
the first stop, so a second ends a command at once: the next cancel after the one that started a
kill — even one that comes while `spawn` still runs, or at the same moment — cancels that kill,
once, and so does the second cancel during a kill the workspace started on its own, for a command
past its timeout or that has ended, `Job(stop=true)` or `close()`. The call still waits for the
kill: a cancelled `kill()` must still end the command and everything it started, at once, before it
returns; `Local` sends SIGKILL in a `finally`, and a backend of yours must do the same. A `kill()`
that raises is the call's error; when the call is already leaving with a cancellation, or with the
error of a `wait()` that raised, it is logged as an error instead, `tools/workspace: could not stop
the command: …`, and the kill that follows a job's end puts its error in the job's output. `read`
refuses a file larger than `limit`; `write` creates parent directories. A path is the model's,
relative to the workspace, and what it can reach is the backend's to decide; an error a backend
raises reaches the model as the tool's error, a `TimeoutError` of its own as it is: only the
workspace's `timeout` makes a `Shell` return `timed_out` or a `Write` or `Edit` fail with
`tools/workspace: Write did not finish within 120s`. A backend's `description`, if it has one, is
ignored: the machine the model has is told in the agent's instructions, not by the workspace.

The library has no notion of a secret. A key a user types is the run's input; a key you put in
`Local(env=)` is the commands' environment, which a command can print. Either way the record keeps
what the model was given and what commands printed, exactly, in the run's scope and in what you
export from it. `record=False` keeps a run out of the record; `delete_runs` and `delete_scope` take
it out later.

An agent without `workspace: true` never sees these tools, even when the run has a workspace; an
agent with it, run without one, is a `ConfigurationError`, and so is a decision agent that declares
it. Delegated and routed agents receive the run's workspace on the same terms. `stop.now()` stops a
running command the same way the timeout does, and a second one kills it at once; either leaves jobs
running. You own the workspace and its lifetime: a temporary directory per run, or one kept for a
project. A workspace belongs to one tenant, the one its `scope` names: a run, a tool or a replay of
another scope refuses it before anything runs — `engine: workspace belongs to scope 'acme', not
'globex'` — so a workspace, its files and its jobs never cross tenants.

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
rest" is the function above; "always allow this command" is a set the function keeps. A refused call
does not run: the model reads `NotApproved: Shell was not approved` and carries on, and a reason
reaches it through the `inbox`. Each step records `approved` — `True`, `False`, or `None` when
nobody was asked — and the exported rows keep it, so a person's refusals become labels. A
workspace's calls carry what a person needs to decide: `background` on `Shell`, `wait` and `stop` on
`Job`.

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

Attach a refinery and every run is recorded when it ends — one that raises after its first model
call too, as far as it got, with its error. A recording that fails is logged as a warning and
never fails the run, as one does when a model's tool call nests its `args` past 254 levels of lists
and objects, `args` itself the first, or fewer in a delegated or routed run, deeper than the
record's serializer goes (`ValueError: Circular reference detected (depth exceeded)`).
`refinery.run(scope, run_id)` reads a recorded run back, its delegated and routed runs in place, to
continue it anywhere — see [Running an agent](#running-an-agent). The record keeps a run's input as
it was given, and reads it back so: a generation's list input is the parts carried by the user
message the input became, the skill's part aside, when their JSON equals the stored input, and JSON
otherwise — a part-shaped class of your own is stored as it dumps itself; a decision's list state
holding media reads back as parts only when every item is written exactly as a part, all its fields
and no other, and as JSON otherwise.

```python
refinery = Refinery(PostgresStorage(pool))
await refinery.check()
engine = Engine(root, providers=providers, refinery=refinery)
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

And models label what people did not. A **judge** is a decision agent of your folder whose questions
are about an episode; it sees the input, the reply, every tool call with its result — one a
command made through the [tool bridge](#workspace) with its `by` — and the error a failed run ended
with, with each image, audio or document replaced by a placeholder naming it —
`[image: image/jpeg]`, `[document: application/pdf, "policy.pdf"]` — while the recorded run and the
exported datasets keep the media. A judge sees neither the system prompt nor `run.preload`, so a
rule in an agent's instructions, or one a preloaded agent never had to `Read`, is not in front of
it; ask about them in the question when they matter:

```yaml
judge:
  kind: decision
  model: judge
  instructions: judge/instructions.md
  questions: [judge/*.json]
```

```
judge/grounded.json    check   Is every claim in the reply supported by a tool result?
judge/rule.json        check   Does the reply apply the refund rule as written?
judge/done.json        score   levels: Not addressed, Partly, Fully, Fully and confirmed
```

A **teacher** is a generation agent that answers a decision agent's questions with a rationale, so
a decision made without ground truth still gets one; it reads each question as the run asked it,
structure written as compact JSON. Both run when you say so, over runs they have not labeled yet:

```python
labeled = await refinery.label("acme", engine, judge="judge", teacher="teacher")
labeled.judged, labeled.taught, labeled.errors
```

A judge or teacher the folder does not have is a `NotFoundError`, and one of the wrong kind a
`ConfigurationError`, before any run is labeled. A run that could not be labeled — the model
failed, or the teacher answered a choice with no option, a score with no level index, a check with
anything but true or false — is named in `labeled.errors` with why, and stays unlabeled for the
next call. Every label carries its source. A
judge's score is exported as a judge's score, never as a person's. A judge's or a teacher's label also
names the `model` and `settings` of the run that made it, so a score can be traced to what produced
it; a person's and the world's carry `null` for both.

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

Keys are `{scope}/{agent}/{dataset}/{timestamp}.jsonl`, so repeated exports add files and never
overwrite. A run in a mode goes under `{scope}/{agent}.{mode}/{dataset}/{timestamp}.jsonl`, since two
modes may hold one question id with different wording; each row carries its `mode`, and
`mode="chat"` exports only that mode. Keep one watermark per `mode` value, and never pass a filtered
export's watermark to an unfiltered one: the runs it did not cover, recorded before it, are skipped
with no error. An agent named like a mode of another is refused at load, so a key still names one
agent. `since` selects every run recorded or labeled after it, written in full with all its
labels. A generation or decision run that raised — its `error` set — is in no dataset; replay and
label it like any other. A router whose routed run raised is the exception: it answered, so its
decision rows are written. A voice run is in no dataset, whether it raised or not, and the watermark
does not count it.
A run is recorded when it ends, so a long run still working at one export is in the next, whenever
it started; the watermark covers both, so a review of a run you already exported reaches the next
export. Rows are keyed by `run_id`: when a run appears again, its latest row supersedes the earlier
ones — dedupe by `run_id`, keep the latest. Because of that you may pass a `since` a little before
the last watermark: a run is stamped just before it is written, and services recording with their
own clocks can disagree by a little.

Every row carries `"holdout": true|false` — the rows kept aside so any measurement on them is
honest. It is derived from a hash of the run id (`holdout_percent=20` by default), so a run's flag
never changes between exports; `is_holdout(run_id)` gives the same answer in your own code. Three
datasets:

| dataset | one row per | what it holds |
| --- | --- | --- |
| `generation` | generation run | skill, context, tags, model, settings, fingerprint, system, preload, start, messages, tools, conversation, steps, reply, stopped, stop_reason, usage, every label |
| `preference` | human correction | tags, fingerprint, model, settings, prompt (system, preload, messages), chosen (the correction), rejected (the reply) |
| `decision` | question of a decision run | tags, model, settings, fingerprint, state, question, answer, probabilities, confidence, truth, truth_source, truth_model, truth_settings, outcome |

Every row carries its run's `tags`, `{}` for a run that had none, and its `fingerprint`,
`{"agent": …, "tools": …, "questions": …}` — `questions` is `null` for a generation run; group or
filter the rows by them in your own code. Every row also carries, right after `run_id`, the run's
`parent_run_id` — the run that delegated or routed to it, `null` for a top-level run — and its
`call` as `Run.call` holds it (`{run_id, agent, scope, tool, call_id}`); a dataset regroups its rows
by tree with them.

`model` and `settings` are the model that wrote the run's reply, or made its decision, and the
settings it reported or its adapter declared, `null` when neither said; a preference row's name the
ones that wrote `rejected`. Each exported label carries the `model` and `settings` of the judge or
teacher behind it, and a decision row's `truth_model` and `truth_settings` are the teacher's when
`truth_source` is `"teacher"`, `null` when it is `"human"`.

The first two are a generation agent's episodes and the corrections people made to them. A
generation row's `tools` is what the model was offered — `{name, description, input_schema}` each —
`preload` the parts `files.preload` and `skills.preload` sent ahead of `messages`, `[]` without it, and `conversation` is
`run.conversation` in order: history, input, assistant turns with their `tool_calls` (`{id, name,
args}`), tool results with `is_error`, the final reply. Together they fit standard fine-tuning and
eval formats with a few lines of mapping; the library writes no vendor's format. Parts are written
inline, their bytes as base64, so a row is self-contained; strip the media from a row you do not
want it in. The third is a decision agent's answers beside what turned out true: a calibrated model
is one whose 0.8 was right 80% of the time, and this is the table that says whether it was:

```json
{"run_id": "…", "parent_run_id": null, "call": null, "holdout": false, "agent": "frontdesk", "tags": {"arm": "b"}, "question_id": "intent", "answer": "billing",
 "probabilities": {"billing": 0.8, "technical": 0.15, "other": 0.05}, "confidence": 0.7,
 "truth": "technical", "truth_source": "human", "truth_model": null, "truth_settings": null,
 "outcome": "failed"}
```

`refinery.conflicts(scope)` lists runs that carry two different human corrections — a dataset can
hold either, not both, and the library reports rather than guesses.

Replay measures a candidate — a new folder, a new model, any `Engine` you built — on what your
recorded runs were given. Each pair holds the recorded run, its labels and the candidate's run:

```python
candidate = Engine(root, providers=providers)

async def read_only(agent, call):
    return call.name in {"Read", "Grep", "Glob", "LS", "Load", "Delegate", "lookup_order"}

async for pair in refinery.replay("acme", candidate, agent="support", holdout=True, labeled=True,
                                  approve=read_only):
    pair.recorded.reply, pair.labels, pair.candidate.reply if pair.candidate else pair.error
```

> **Replay runs the candidate's tools for real.** A tool that refunds refunds again, a tool that
> sends a message sends it, a `Shell` runs. Pass an `approve` that refuses every tool that changes
> anything — `read_only` above lets through the builtins that read the folder, `Delegate`,
> whose runs it is asked about too, and a lookup — or replay with `tools=` stubs that only read: a
> run's tool wins over the folder's tool of the same name. A refused call reaches the model as
> `NotApproved`, as in any run.

It replays the top-level runs of `agent` in the scope, oldest first. `holdout=True` keeps the
held-out ones, by the same hash and `holdout_percent` as `export`; `labeled=True` those a person
reviewed or the world reported on; `tags=` those whose tags contain these, as `delete_runs` matches;
`fingerprint=` the runs one version produced — the `run.fingerprint` of any of them, its `agent`,
`tools` and `questions` all equal; `limit=` the first so many. Each is a run of the candidate,
as `candidate.run(agent, scope, …, record=False)` would be but never reading `/name`, with the
recorded input — parts as parts — the history sent before it, the
skill and the `context`, so the candidate sees what the recorded run saw, its own skill file
read for a recorded skill. A run invoked by `/name` recorded that skill and the rest of its input,
and the replay sends that pair: `/name` is never read again, so a rest that itself starts with
`/<skill>` replays as text, as does a recorded input of `/close x` whose `close` has become public
on the candidate since. A decision run replays its state, and asks its runtime questions again
as they were recorded while the questions of its entry come from the candidate's folder, its
scripts filling theirs anew — a runtime id the candidate's folder now asks fails that pair; when its
route reached a generation run, that run's input, history and context go too. What a record does not
keep is yours to pass again — `expect=`, `approve=`, and the `tools=` and `workspace=` the runs were
given. Replays given one workspace share its files and its jobs, and a job outlives the replay that
started it: close the workspace when the replays are done, and the candidate, which stops its
workers. The candidate delegates and routes as it does, so a run with children replays as one run.

`repeat=3` replays each run three times: each attempt is a run of its own and a pair of its own,
with its own copy of the recorded run and its labels, so what you change in one pair reaches no
other attempt; `pair.attempt` counts them from 0, and the three come one after another; `limit=` still counts
recorded runs. A model samples, so the same candidate on the same input can answer differently, and
the attempts show how much. `concurrency=4` keeps up to four replays going, attempts included, and
while you read a pair at most three more run; with the default of one, a replay starts when you ask
for its pair. Pairs come in recorded order either way. A candidate that fails yields its pair with
`error` — `"ModelError: …"` — and `candidate=None`, and the replay goes on, the run's other
attempts included. Nothing is recorded, no label is written, and nothing is graded: the pair holds
both runs with their models and fingerprints, and what you compare is yours.
[`evaluation.md`](evaluation.md) grades the replies with a decision agent.

### Agents kept in a repository

The folder is plain files, so it can live in a repository of its own: people who do not write code
edit it in the host's web UI, every change is a reviewed pull request, and CI runs `rfnry-engine
folder check` on it ([Checking an agents folder](#checking-an-agents-folder)). Fetching it is a step
of your deployment, not of the engine, which takes a path.

Pinned at build time, a Dockerfile `ADD` fetches one commit into the image, and the commit in the
URL is the version, so building the same Dockerfile again gives the same folder:

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.14-slim
ADD https://github.com/acme/agents.git#4f2c9e1d8b7a6c5e4f3d2c1b0a9f8e7d6c5b4a39 /srv/agents
```

```sh
GIT_AUTH_TOKEN=$(cat agents-token) docker build --secret id=GIT_AUTH_TOKEN -t support .
```

BuildKit reads the `GIT_AUTH_TOKEN` secret to fetch a private repository and leaves it out of the
image, and `ADD` leaves out the repository's `.git`. Fetched at start instead, a `git clone` in the
entrypoint takes the commit the deployment names, removes what holds the token, and hands over to
the server without it:

```sh
#!/bin/sh
set -eu
git clone --quiet "https://token:${GIT_AUTH_TOKEN}@github.com/acme/agents.git" /srv/agents
git -C /srv/agents checkout --quiet --detach "$AGENTS_COMMIT"
rm -rf /srv/agents/.git
unset GIT_AUTH_TOKEN
exec "$@"
```

Push access to the agents repository is push access to the server. The folder holds the scripts the
engine's workers run with your server's environment, and it names the models every run calls, so
whoever can push to it runs code as your server and decides what a run costs; protect its main
branch as you protect your server's code, and deploy a pinned commit. A value in the file can lower
a limit of the `Engine` and never raise it ([Limits](#limits)).

The engine reads a folder once; a new version of the agents is a new `Engine`. Check each commit out
into a folder of its own, build an engine from it, and swap which engine new runs use. A run in
flight finishes on the engine it started with. Rolling back is loading an older commit, and since a
commit's folder never changes, a half-written checkout is never live. What an agent's prompt holds —
its instructions, the map, the list of its skills, the preload, an invoked skill, its tools'
definitions and scripts — is what the engine loaded at `Engine()`; what the model reads with a tool,
and the data a script opens, is read from disk when it asks, so remove an old folder only once its
runs have finished. Each engine starts the workers of its own folder's tools, which run the scripts
as it read them, so close the engine you swap out — `await engine.close()` — once its runs have
finished too. An agent's name is its key in `rfnry.yaml`, so every commit keeps its names whatever
path it lands at. The checkout is yours — `git worktree add`, a clone, an archive:

```python
def load(commit):
    return commit, Engine(checkout(commit), providers=providers, refinery=refinery)

live = load("4f2c9e1")
commit, engine = live
run = await engine.run("support", scope="acme", input=message, tags={"commit": commit})

_, retired = live
live = load("a81d07b")
await retired.close()

arms = {"a": load("4f2c9e1"), "b": load("a81d07b")}
arm = "ab"[hashlib.sha256(b"user-42").digest()[0] % 2]
commit, engine = arms[arm]
run = await engine.run("support", scope="acme", input=message, tags={"commit": commit, "arm": arm})
```

Tag each run with the commit it came from and the record, its delegated and routed runs, and every
exported row say which version of the folder answered. The tag names a version; the fingerprint proves
it. A tag says what you meant to load, and a checkout with an uncommitted edit carries the same tag;
`run.fingerprint.agent` is taken from the files the engine read, so two runs are the same agent when
their fingerprints match, whatever their tags say. The same tag is an A/B test: load two commits
side by side, send each user to one by a stable hash of their id, tag the run with its arm, and
group the exported rows by `tags["arm"]` to compare the labels and outcomes of whole versions of the
folder.

### Running many at once

One `Engine` per process serves every run at once; scale out with processes. A run in flight is tied
to the process that started it, so route a chat's later messages, its stop and its inbox to that
process. The tools of the folder run in the engine's own workers beside that process — one process
by default, and as many as the [`workers:`](#workers) section of `rfnry.yaml` declares — each process
taking 32 plain `def run` calls at once on its threads, or the `threads` its worker gives, and
`async def run` calls without a limit on its event loop.

The tools' file work (`Read`, `Grep`, `Glob`, `LS`, `Load`, the history search, and `Local`'s file reads and
writes; its commands are processes of their own) runs on the event loop's default executor,
`min(32, cpus + 4)` threads. When many runs search at once, size it with
`loop.set_default_executor(ThreadPoolExecutor(64))`. A turn that calls no file tool never waits on
it.

For an agent with `Read` or `Load`, the FILES map goes out with every model call, roughly 6 tokens
per file. A corpus of thousands of files belongs behind a retrieval tool, not in the folder.

With reasoning on, a run's CPU grows with the history it is sent. The window is yours; see
[Long conversations](#long-conversations).

## Tools

One concept, one word. A **tool** is a name, a description, an input schema and what runs when the
model calls it. A **`Tools`** is a set of them: what a model receives, and what your code can call
the same way.

```python
class Tools:
    def definitions(self) -> list[dict]                 # [{name, description, input_schema}]
    async def dispatch(self, name: str, args: dict) -> str | dict | list[Part]
    def __add__(self, other) -> Tools                    # union; a duplicate name is refused
    def only(self, *names) -> Tools
    def without(self, *names) -> Tools
    names: list[str]                                     # every tool's name, in order
```

A tool of the folder runs in a worker process the engine starts beside your server; a `Tool` of
your code runs in your process and rides a run. The model calls both the same way.

### Tools of the folder

A tool of the folder is a Python script, declared once under `tools:` in `rfnry.yaml` and picked by
name in an agent's `tools.include`. Its key is the name the model sees; neither the file's name nor
the function's decides anything, so `CallInfo.tool` and a tool's error text say the key:

```yaml
version: 1
models:
  strong: {provider: anthropic}
tools:
  refund:
    module: billing/tools/refund.py       # required: a .py file from the folder root
    schema: billing/tools/refund.json     # optional: description, input_schema, output_schema
    timeout: 20                           # optional: seconds per call
agents:
  billing:
    kind: generation
    model: strong
    instructions: billing/instructions.md
    tools:
      include: [refund]
```

`module` is a `.py` path from the folder root, since the workers run Python. `function` is what the
worker calls as `function(args, call)`, `async def` or plain `def`, `run` when left out, so two tools
may share a module, which loads once in a process; the file is checked to hold it at its top level
without being imported. `description` comes from the entry, else from the schema file, and one of
them gives it. `schema` is the path of a JSON file taking `description`, `input_schema` and
`output_schema` and no other key, a file rather than inline because the tests beside a tool read it
too:

```json
{
  "description": "Refund one charge of the customer's. Returns the refund's id and status.",
  "input_schema": {
    "type": "object",
    "properties": {
      "charge_id": {"type": "string", "description": "The charge to refund, such as ch_81"}
    },
    "required": ["charge_id"],
    "additionalProperties": false
  },
  "output_schema": {"type": "object", "required": ["refund_id", "status"]}
}
```

`description` is what the model reads about the tool — what it does and what it returns — and its
first line is what the prompt's list of tools shows. `input_schema` is a JSON Schema of the
arguments, its descriptions included, whose top level is `"type": "object"`; `output_schema`,
optional, is one of the result. When the engine loads, both schemas are compiled — any draft, 2020-12
included, and never a reference fetched from the network — and an unknown key, a missing or blank
description, an `input_schema` that is not an object or a schema that does not compile is a
`ConfigurationError` naming the file: `tools/folder: <path> has the unknown key 'inputSchema'; a
tool's JSON takes description, input_schema and output_schema`. The model is offered the name, the
description and the `input_schema`; `output_schema` never reaches it. A call's arguments are checked
against `input_schema` before they reach the script, and its result against `output_schema` after;
either that does not fit is a tool error the model reads, `InputError: tools/declared: invalid
arguments for refund: 5 is not of type "string" at charge_id`, so the script never sees arguments
its schema refuses and the model never reasons over a malformed result. A tool without a schema
takes any JSON object, checks its own arguments, and its result goes unchecked.

The script holds the logic, and ends with a footer that runs it [from a shell](#from-a-shell-with-its-tests-beside-it):

```python
import payments


async def run(args, call):
    refund = await payments.refund(call["scope"], args["charge_id"], reference=call.get("run_id"))
    return {"refund_id": refund.id, "status": refund.status}


if __name__ == "__main__":
    import _cli

    _cli.main(run)
```

`args` is what the model wrote, already checked. `call` is the call it serves, set by the engine,
never by the model: its `scope` — the tenant, always — its `tags`, and inside a run its `run_id`,
`agent`, `tool`, `call_id` and `by`, the fields of `CallInfo`, `by` naming the `Shell` call whose
command made the call through the [tool bridge](#workspace), `None` for the model's. `tags` are
the tags of the run the call serves, which a delegated or routed run inherits, and outside a run
the ones given to `EngineTools`, `{}` without. What is a tenant's travels in `call`, and so does
what the run is about: a tool bound to the ticket or the customer a run serves reads it from
`call["tags"]`, which your code set and the model can neither see nor change, never from an
argument the model writes. What every tenant shares, a URL or a token, comes from the environment,
which a worker inherits from your server as it is — the engine never reads it. `run` returns a
string; JSON — an object, a list, a number — the model reads as JSON; or a list of parts, each an
object with the `type` and the fields of a `Text`, `Image`, `Audio` or `Document`, media `data` in
base64, which reach the model as parts: a PDF the model reads itself, an image it looks at. A result
JSON cannot hold, a `datetime` or a `set`, is a tool error naming its type; arguments it cannot hold,
passed from code, are an `InputError` before any worker sees them (`workers/pool: the arguments for
refund are not JSON: Object of type datetime is not JSON serializable`).

A tool imports the standard library, installed packages and any module beside it, `payments.py`
here, whatever the module is named: what `python refund.py` reaches. A module beside a tool is a
`.py` file there: a package folder beside a tool is refused when the engine loads, as is an import
of a name below such a file (`import payments.sub`), and installed packages import as usual. A
module in a folder above is out of its reach, and an import that resolves to nothing is a
`ConfigurationError`, unless it sits under `try/except ImportError` or `if TYPE_CHECKING`. A module
beside it is imported at the top of the script, or under its footer, never inside a function or a
class, and neither it nor the tool's own module is named like a standard-library module:
`tools/time.py` is refused, since a worker would load one or the other depending on the name
(`tools/folder: <path> is named like the standard-library module 'time', and the worker can't load
both; rename it`). A script never imports the engine, nor your server's code: an import of
`rfnry_engine`, of a module relatively, or of a module whose file lies in the project that holds the
folder — the nearest folder at or above the root with a `pyproject.toml` — outside the standard
library, the installed packages and the modules beside it is a `ConfigurationError` naming the file
and the module. The installed packages are those of the site folders of your server's interpreter
and of the folders the interpreter that runs the workers passes to `site.addsitedir` as it starts —
its own `site-packages`, the user's, and every site folder a `.pth` file or `sitecustomize` adds, as
`uv run --with` does with your project's `.venv`. A folder a `.pth` line merely puts on the path, as
an editable install of your own app does, stays your server's code. To learn those folders, loading
runs that interpreter once, with `-P -S` and for at most ten seconds, and only when a script imports
a module of the project that your server's own site folders do not hold; that run executes your
`.pth` lines and `sitecustomize` as any start of it does, and when it cannot answer, the server's own
site folders are the installed packages. That catches a mistake; it is not a sandbox, and a script
is trusted as your server's code is.

Only what `tools:` names and the modules those bring in by an `import` statement go to a worker and
into the fingerprint of an agent that picks it, along with the tool's entry, so a test file or a
`conftest.py` beside a tool is just a file: nothing loads it, no worker receives it and nothing checks
its imports. A tool's module runs in a worker under a name of its own, so its footer stays inert
there. The module, the schema and the modules beside it that a tool imports leave the map of an
agent that picks it, as its instructions do; the rest of a folder `files.include` reaches stays on
the map.

An entry's `tools:` decides which tools its agent has:

```yaml
billing:
  kind: generation
  model: strong
  instructions: billing/instructions.md
  files:
    include: [billing/knowledge]
  tools:
    include: [refund, handoff]          # declared, given by the engine or given by the run
    exclude: [Grep]                     # applied after include, across the entry and its mode
    timeout: 30                         # seconds per call, under Engine(tool_timeout=)
    max_parallel: 4                     # tool calls of one run at once
```

An entry of `include` or `exclude` holding `*`, the only wildcard, is a pattern over the names under
`tools:` and never reaches a built-in or a tool a run gives. An entry without one is a name: a
declared tool, then a tool the entry's sections grant, then one the run gives through `tools=`, and a
run that does not give it fails before any model call (`tools/engine_tools: agent 'billing' lists
'handoff' in tools.include, which this run does not offer; its tools are Glob, Grep, LS, Load, Read,
refund`). The sections grant `Read`, `Grep`, `Glob`, `LS` and `Load` with `files` or `skills`,
`Delegate` with `orchestration.delegate_to`, and `Shell`, `Job`, `Jobs`, `Write` and `Edit` with
`workspace: true`, whether or not `include` names them; naming one when its section is missing is a
`ConfigurationError` naming the section, since no run can supply it. `History` comes with a run whose
`start=` is past the beginning of its history and `Output` with `expect=`: the run decides them, so
either one in `include` or `exclude` is a `ConfigurationError`. `exclude` applies after `include`,
across the entry and its mode, and drops exactly the tool it names, a tool a section grants
included: `exclude: [Delegate]` takes delegation away, and `exclude: [Write]` leaves `Edit` and the
rest. `Shell`, `Job` and `Jobs` go together or not at all, since `Shell` can start a job only `Job`
and `Jobs` can read or stop, so excluding one or two of them is a `ConfigurationError`. A name in
`exclude` that is neither declared nor given by the engine is read as one the run gives, and that
tool of the run is dropped, so a mistyped name there cannot be caught when the engine loads. A run's
`tools=` replaces a declared tool of the same name, which is how a test or a replay stubs a real
system; a tool of the run that `include` does not name is not offered. A pattern that matches
nothing, a declared tool that no entry or mode picks once `exclude` has applied, a declared tool named
like one the engine gives (`Read`, `Grep`, `Glob`, `LS`, `Load`, `Shell`, `Job`, `Jobs`, `Write`,
`Edit`, `Delegate`, `History`, `Output`), and a module, function or schema that is not there are
`ConfigurationError`s when the engine loads. The smallest of the tool's own `timeout`, the entry's
`tools.timeout` and `Engine(tool_timeout=)` bounds each call; a `Workspace`'s own `timeout` replaces
them for `Shell`, `Job`, `Write` and `Edit`. `max_parallel` is one of the [limits](#limits).

### From a shell, with its tests beside it

A tool runs from a plain shell as well as in a worker, so it can be tried and tested with no server
and no model. A `_cli.py` beside the tools reads the arguments as one JSON object, takes the run's
context from `--call`, awaits an `async def run`, and prints the result as JSON:

```python
import argparse
import asyncio
import inspect
import json


def json_object(text):
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(text)
    return value


def main(run):
    parser = argparse.ArgumentParser()
    parser.add_argument("args", type=json_object, help="the arguments, one JSON object")
    parser.add_argument("--call", type=json_object, default={}, help="the call, one JSON object")
    given = parser.parse_args()
    result = run(given.args, given.call)
    if inspect.isawaitable(result):
        result = asyncio.run(result)
    print(json.dumps(result))
```

With the footer the script above ends with, a tool runs from the folder root:

```sh
python billing/tools/refund.py '{"charge_id": "ch_81"}' --call '{"scope": "acme"}'
```

A tool run this way skips `approve`, the record and the deadline: it is for trying and testing the
tool, while an agent's calls go through the engine, and a command a [workspace](#workspace) runs
calls a tool through the tool bridge when it needs those. Arguments or a `--call` that are not one
JSON object end it before `run` with a usage error, and nothing on stdout.

Its tests sit beside it, `refund_test.py` beside `refund.py`, with a `conftest.py` in the folder
holding a `check` fixture: it validates the arguments against `input_schema`, calls `run`, and
validates the result against `output_schema`, as a call through the engine does:

```python
import asyncio
import importlib
import inspect
import json
from pathlib import Path

import jsonschema_rs
import pytest


@pytest.fixture
def check():
    def checked(tool, args, call):
        schema = json.loads(Path(__file__).with_name(f"{tool}.json").read_text())
        jsonschema_rs.validate(schema["input_schema"], args)
        result = importlib.import_module(tool).run(args, call)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        jsonschema_rs.validate(schema.get("output_schema", {}), result)
        return result

    return checked
```

```python
from types import SimpleNamespace

import jsonschema_rs
import payments
import pytest


async def refunded(scope, charge_id, reference=None):
    return SimpleNamespace(id="re_1", status="succeeded")


def test_a_charge_is_refunded(check, monkeypatch):
    monkeypatch.setattr(payments, "refund", refunded)
    refund = check("refund", {"charge_id": "ch_81"}, {"scope": "acme"})
    assert refund == {"refund_id": "re_1", "status": "succeeded"}


def test_a_charge_id_is_text(check):
    with pytest.raises(jsonschema_rs.ValidationError):
        check("refund", {"charge_id": 81}, {"scope": "acme"})
```

`pytest billing/tools` runs them: pytest puts the folder on the import path, so `check` imports a
tool by its module's name, and `jsonschema_rs` comes with rfnry-engine. The engine loads none of
these files, since only what `tools:` names is a tool.

### Workers

One engine serves every agent, and its tools run in workers. A worker is a named set of tool
processes, which the engine starts with your server's interpreter, so they have the packages your
server has:

```yaml
version: 1
workers:
  customers: {processes: 2}             # threads and max_result_bytes may be given too
  live: {processes: 1}
models:
  strong: {provider: anthropic}
agents:
  support:
    kind: generation
    model: strong
    instructions: support/instructions.md
    worker: customers
    modes:
      chat:
        worker: live
```

`workers:` declares them, each with `processes` (1 unless given), `threads` (32 unless given, the
plain `def run` calls at once in each process) and `max_result_bytes`, and `worker:` on an agent or
a mode picks one. An agent with no `worker:` uses the worker called `default`, one process unless
declared, so a file with no `workers:` section has only that one; a `worker:` naming no declared
worker, and a declared worker nothing picks, are a `ConfigurationError`. A worker holds the tools and
question scripts of the agents and modes that pick it ([Decision agents](#decision-agents)), a tool
picked through two workers loads in both, a call goes to the worker's process with the fewest calls
in flight, and a delegated or routed run uses its own agent's `worker:`, never its caller's. A
worker separates crashes and freezes of tool code: a process that dies fails the calls in flight in
that process, the next call starts a new one, and a call in another worker goes on. An `async def
run` that blocks
stalls every call in its process, since they share one event loop. Model calls and your own code
stay in your server's process. More processes in all than `Engine(max_worker_processes=)`, 16 unless
you set another number, is a `ConfigurationError` when the engine loads, before any process starts:
`engine: the workers of this folder run 17 processes in all, over max_worker_processes=16; lower
processes: under workers:, or raise Engine(max_worker_processes=)`. The `default` worker counts when
an entry or a mode uses it.

Workers run on POSIX systems only, Linux and macOS, as `Local` does: each process is a session of
its own, stopped with every process it started by signalling its process group. The scripts and the
modules beside them are read when the engine loads, as skills are — the text the import check read
and the fingerprint covers — and a worker runs that text, under each file's own path, so `__file__`
still finds the data beside it: an edit to a tool reaches the next `Engine`, never the one running,
whose records keep naming what ran, while a data file a script opens when it is called is read
from disk then. A worker starts at the first call that needs it, never when the engine is built,
so a worker nothing calls costs nothing, and again after a crash, and lives until you close the engine — `await engine.close()`, or leaving
`async with Engine(...) as engine:` — which stops every worker and whatever it started, and raises
the first error stopping one raised once all are stopped, logging the others; it waits for a worker
a call is still starting, to stop it too, and a cancel of `close()` reaches you once the workers on
your loop are stopped — one still starting is stopped as it comes up — and those on another loop are
handed to it, without waiting for that loop. The engine watches each worker with one thread of its
own, which waits on the worker's exit and ends with it, so a server counting its threads counts one
more for each worker running. A closed
engine stays closed: a run, a tool call or an `EngineTools` after `close()` is an `InputError`,
`engine: this engine was closed; build a new Engine`, before any model is called, so nothing is
recorded of it, and a run still in flight when it closed starts no other run — its `Delegate` call
reads that error, its route fails with it. A plain `def run` runs on one of the worker's 32 threads,
or the `threads` its worker gives, and calls past those wait; an `async def run` runs on its event loop, as many at once as come. The
worker goes by the `run` it loaded, whatever its file spells: a `run` a decorator made a coroutine
function runs on the event loop, any other on a thread, and what that one returns, if it can be
awaited — a coroutine, from a plain decorator around an `async def` — is awaited on the event loop.
What a script prints goes to the log, at DEBUG on the `rfnry_engine.workers.pool` logger, never into
its result: a record per line, and 64 KiB at a time of a line whose end has not come; with DEBUG off
it is dropped as it comes. A script that raises, `sys.exit()` included, is a tool error the model
reads, `WorkerError: ValueError: no order A-1`, and the worker goes on. What a script returns or
raises crosses back as JSON of at most `Engine(worker_result_bytes=)` bytes as the worker writes it,
a non-ASCII character escaped to six, 32 MiB unless you set another number, `None` for no limit, and
a worker's `max_result_bytes` lowers it for that worker and never raises it;
past it the worker sends a short error in its place, so a result too large never reaches your
server, the model reads `WorkerError: workers/process: refund sent back 40000023 bytes, over the
limit of 33554432 bytes`, and the worker goes on. A result under the limit still costs your server
about three times its size while it is read, for each worker with one in flight, and recording the
run holds its body, which stores a result twice, in its step and in the conversation, at most twice
over while it is written. A worker that exits — `os._exit()`, a signal, out of memory, a crash in a
C extension — fails every call it had in flight, even while a process it forked still holds its
pipes: `WorkerError: workers/pool: the worker for refund stopped: exited with code 3`, or `killed by
signal 9` when a signal ended it, the run goes on, and the next call starts a new worker. A worker
that sends back a line that is not a reply, as a process it forked writing to its pipe could, is
stopped the same way, its calls failing with `workers/pool: the worker for refund stopped: it sent a
line that is not a reply`. Past `tool_timeout` the model reads `TimeoutError: workers/pool: refund
timed out after 30s`, a deadline that counts the wait for a worker too: an `async def run` is
cancelled in the worker and holds none of its threads, unless it blocks the worker's event loop —
`time.sleep` or blocking I/O inside it — so that its cancel is still unanswered when a call sent
after it passes its deadline too, and then each of those calls holds a thread; a `def run` cannot be
stopped, so its thread runs on and its result is discarded, and once every thread of a worker is
held so, the next call stops that worker and starts a new one, and the calls the old one still had
in flight fail with `WorkerError: workers/pool: the worker for refund stopped: every thread was held
by a call past its deadline`.

An engine serves one event loop at a time, as a workspace's jobs live in one loop, and its workers
follow the loop that calls them: a call from a loop other than the one that started them stops
those workers and starts fresh ones on its own. A sync server keeps one loop running in a thread of
its own and hands each run to it, `asyncio.run_coroutine_threadsafe(engine.run(...), loop)`.
`asyncio.run` per request or per task, or a test's own loop, starts the workers anew each time, and
loops in several threads that call one engine at once keep stopping each other's workers, so give
each live loop its own engine. A worker that stops — for another loop, a retirement, a crash or
`close()` — is reaped and its pipes closed then and there, so a thousand `asyncio.run` calls leave
the engine holding no more files or processes than one does. `close()` from another loop stops the
workers at once: while their loop runs in another thread, the stop runs on that loop — or in the
caller, should that loop stop before it gets to it — and `close()` returns once it is done. A
worker that loop was still starting when it stopped is killed and reaped in the caller, while its
pipes wait for that loop: they close when it runs again. Close the engine before its loop ends: closing it later from another loop still stops every worker, but
Python warns of the worker pipes left on the closed loop (`ResourceWarning`).

### Tools of the run

Tools that must live in your process ride a run: `run(tools=)` takes a `Tools`, or anything with its
two verbs, for that one run — a tool bound to a tenant's connection, one that streams to a UI,
rfnry-memory's `MemoryTools`; a tool every run needs is the same `Tools` passed on every run. The
agent is offered the ones its entry's `tools.include` names, here `fetch_invoice`. Each is a `Tool`:
its name, its description, its arguments and the async function that runs it, in your process:

```python
class InvoiceArgs(BaseModel):
    """Which invoice to fetch."""

    invoice_id: str = Field(description="The invoice id, such as INV-204")

async def fetch_invoice(args):
    data = await invoices.get(args["invoice_id"])
    return [Document(media_type="application/pdf", name=f"{args['invoice_id']}.pdf", data=data)]

invoicing = Tools([Tool("fetch_invoice", "Fetch one of the customer's invoices, as its PDF",
                        body=fetch_invoice, args=InvoiceArgs)])
run = await engine.run("support", scope="acme", input=message, tools=invoicing)
```

The `description` is what the model reads about the tool and its first line is what the prompt's
list of tools shows. `args` is a pydantic model: its `Field(description=...)`s and its docstring
reach the model in the tool's schema, and it checks a call's arguments before the body runs. An
`input_schema` states the schema as a dict instead; a tool given only `input_schema`, and a `Tools`
of your own, receive the arguments as the model sent them. A body returns what a script does: a
string, a dict the model reads as JSON, or a list of parts. A run's tool replaces a declared tool of
the same name: that is how a test stubs a tool of the folder that calls a real system, and how you
swap in a version bound to a tenant's connection. Two tools of one run with one name are refused,
and so is a run's tool named like one the engine gives the agent.

A run's tools ride the runs it delegates and routes to, as its tags do, and each agent is offered of
them what its own `tools.include` names. A lead given rfnry-memory's `MemoryTools` that delegates to
a writer whose `tools.include` names none of them keeps the memory tools for itself; a writer whose
`tools.include` names `memory_search` is offered that one, and its runs need a caller that gives it.

Every tool call inside a run has a deadline, `Engine(tool_timeout=30.0)` by default and `None` to
turn it off. When it passes, the model gets the error
`TimeoutError: engine: lookup_order timed out after 30s` and the run continues. A folder tool's
deadline is its worker's, above. `History` reads your storage under the same deadline.
A run's tools are async and the deadline cancels them, so a body that runs blocking work in a thread
must put its own limit on that work: cancelling the await does not stop the thread. A workspace's
`Shell`, `Job`, `Write` and `Edit` keep the workspace's own `timeout` instead. `Delegate` has no
deadline. The child run it starts is bounded by its own tool calls and `max_model_calls`.

The builtins are bounded whatever the deadline. `Read` returns UTF-8 text and refuses anything else
with a pointer to `Load`; `offset` and `limit` select lines. `Load` takes any file up to 20 MB and
hands it over as an image, audio or a document part, whichever its type says. The type comes from
the name of the file it opens, so a link goes by its target's name (by its own where the system
cannot say which file it opened, as on macOS), and for PNG, JPEG, GIF, WebP and PDF the first bytes
must agree — a `.png` that is not one is refused rather than sent — while a file with no known name
is typed by its bytes or sent as `application/octet-stream`. What the model can take is the
adapter's `media_types`, and past that the vendor's. `Grep` searches text only and skips the rest;
`Glob` and `LS` list every file. Neither `Read` nor `Grep` reads a file over 5 MB. `Grep` stops
after scanning 50 MB and notes the cut. A search that runs past 10 s in one call is an `InputError`
the model can act on. `Glob` and `Grep` walk at most 100,000 entries and note the cut. A tool of
yours that reads the folder itself resolves paths with `Jail`, which refuses one that leaves the
folder, as the builtins do. `Jail.open` opens a file the way the builtins and `Local` do: from a
place it checks is still the folder it was given, one folder at a time without following a link, so
a link swapped in after the check is refused.

A run's tool learns the call it serves with `current_call()`, a `CallInfo` — the run's id, its
agent, scope and tags, the tool and the call's id, what a script reads in `call` — so a side effect
can carry the run that caused it, as an audit row or a header does:

```python
async def refund(args):
    call = current_call()                                  # None outside a run
    await payments.refund(args["charge_id"], reference=call.run_id if call else None)
    return {"refunded": args["charge_id"]}
```

The run's `approve` and its handlers for the call see it too, and so do tasks the body starts, after
the call ends. A delegated run's calls, and those of a run an `engine.tool` call starts, name their
own run, while the rest of such a run — its model calls, its text handlers — runs inside the
caller's call and sees that one; `None` outside a run's tool call.

Tools that belong to a tenant go on the call that names the tenant:

```python
run = await engine.run("support", scope="acme", input=text,
                       tools=MemoryTools(memory, corpus_id="acme"))
```

That is how rfnry-memory mounts: its `MemoryTools` is a `Tools`, neither library imports the other,
and the entry's `tools.include` names the memory tools its agent is offered. The set an agent runs
with is itself a first-class object, so you can drive it without a model — from a test, or from a
loop of your own. `EngineTools` holds exactly what the model of such a run is offered: the file
tools, `History`, the tools of the folder it picks, whose calls go to their worker under
`tool_timeout` with `call` holding the scope and the `tags` given to it, the workspace's, `Delegate`
and the tools of `tools` its `tools.include` names, each less what `tools.exclude` drops:

```python
surface = EngineTools(engine, agent="support", scope="acme")
surface.definitions()                                        # what the model would see
await surface.dispatch("Read", {"path": "support/knowledge/refunds.md"})
await surface.dispatch("Delegate", {"agent": "triage", "input": "It is down"})

combined = MemoryTools(memory, corpus_id="acme") + surface.only("Read", "Grep")
```

A `Delegate` dispatched this way runs the child as its own top-level run, with the scope, `tools=`,
`workspace=` and `tags=` given to `EngineTools` and the `questions` it is called with —
what a `Delegate` inside a run would give it, its input likewise never read for `/name`. A refusal
raises here, where inside a run the model reads it.

An agent is itself a tool. `engine.tool(agent, scope=...)` gives any runtime — a live voice model, a
loop of your own — one tool that runs the agent: named for the agent and described by its
`description`, from its entry or its instructions file's frontmatter, which it needs, taking
`{"input": str}` — and, as `Delegate` does,
`questions` when it is a decision agent — returning a generation agent's reply or a decision agent's
answers as a dict:

```python
consult = engine.tool("billing", scope="acme", name="consult",
                      context="Acme is on the enterprise plan.")
reply = await consult.dispatch("consult", {"input": "Was I charged twice?"})
triaged = await engine.tool("triage", scope="acme").dispatch("triage", {"input": "It is down"})
triaged["answers"]["urgent"]["probability"]                # 0.91
surface = consult + MemoryTools(memory, corpus_id="acme")
```

The tool's `input` is never read for `/name` and the run it starts takes no skill: the caller is a
runtime, not a person typing. A generation agent's tool returns its reply. A decision agent's
returns `{"answers": {…}}`, and when
it routed, also `"routed_to": {"agent": …, "reply": …}` — the routed run's result, nested the same
way when the target is itself a decision agent — so a caller that consults a routing agent gets the
answer it routed to, not only the answers.

It is a `Tools` of one, so it adds to any other. Each call is a top-level run under the scope it was
made with, recorded like any run — but not when the run that makes the call was started with
`record=False`, as a replay's is; `context=`, `tools=`, `workspace=` and `tags=` ride every call.
Mounted inside another run with `run(tools=)` it is a tool like any other, under `tool_timeout`, and
its run is not the caller's child — to nest with lineage, name the agent in
`orchestration.delegate_to`. As across
`Delegate`, `max_depth` counts the whole stack across it; a call through it starts a new input, so
the route chain resets there: an agent may reach a fresh run of itself through it, and only a route
into an agent already in the same chain of routes is refused.

Called inside a run, a consult is part of that run's foreground, as a delegated run is: its run
takes the caller's `stop` and `on_event`. The caller's `stop.now()` reaches the consult instead of
cancelling its call, so the consult ends with its record — one a script's call started included, as
"Calling tools from code" says — and whatever else the call was doing is cut once the consult has
taken the stop — a stop that lands before a call returns cuts it, even when its consult had just
finished, and that consult keeps its record — and the caller's step reads `stopped by the user`, as
a stopped `Delegate` step does; `after_step()` ends it after its current step. A consult a tool of
yours starts in a task of its own — two at once, say — is in the foreground too; one a script's call
started through the workspace's tool bridge ends with the script's command instead, unless
`stop.now()` ended the command, as "Calling tools from code" says. Its events reach the caller's
handler, told apart by `agent` and `run_id`:

```python
async def status(event):
    if isinstance(event, ToolCallStarted):
        print(event.agent, "is using", event.tool)          # billing is using lookup_order

consult, stop = engine.tool("billing", scope="acme", name="consult"), Stop()
run = await engine.run("support", scope="acme", input=message, tools=consult,
                       on_event=status, stop=stop)
```

A stop flows down, never up: a consult that is stopped ends alone, and its caller reads `stopped by
the user` and carries on. Outside a run — a live voice model, a loop of your own — pass the hooks
yourself. `on_event` rides every call; `stop` is a function called at each call that returns the
`Stop` of the turn the call belongs to, since a `Stop` belongs to one run and the tool lives as long
as the session:

```python
turn = {"stop": Stop()}

async def say_progress(event):
    if isinstance(event, ToolCallStarted):
        await voice.say("Checking that now.")

consult = engine.tool("billing", scope="acme", on_event=say_progress, stop=lambda: turn["stop"])

def barge_in():
    turn["stop"].now()                  # stops the consult running now; it is recorded, stopped
    turn["stop"] = Stop()               # the next turn's consults run normally
```

Given inside a run, these win over the caller's: a `stop` of your own, or `None` from the function,
leaves the consult out of the caller's foreground, and the caller's stop cancels its call as it
cancels any tool's. Outside a run, a stopped consult's `dispatch` returns what its run produced.
`EngineTools` takes the same two for the `Delegate` it dispatches outside a run.

## Storage

The refinery is the only part of the library that stores anything, and it stores in Postgres:
`PostgresStorage(pool)`, over an asyncpg pool you built, in a schema of its own. It sends and
reads every JSON value as text, cast in its SQL, so it works with any `json` or `jsonb` codec on
that pool, such as `json.dumps` and `json.loads` set for both, which never sees what it stores.
Each run's body reaches Postgres as a `bytea` argument, an object with the buffer protocol that
hands its JSON over and keeps no copy, encoding it again from the body each further time it is
asked, so the write holds it at most twice and a driver that asks twice sends the same JSON both
times; a `bytea` codec of your own on that pool must take any such object, as `bytes(value)` does,
and one that takes only `bytes`, such as a text codec calling `value.hex()`, fails every record
write with `StorageError`. A Postgres to try it on is one container, with engine's schema prepared
by the command the package installs:

```bash
docker run -d -p 127.0.0.1:5432:5432 -e POSTGRES_PASSWORD=postgres postgres:18-alpine
rfnry-engine create --url postgresql://postgres:postgres@127.0.0.1:5432/postgres
```

### Preparing the database

Your app never creates or changes engine's tables. `rfnry-engine`, the command the package
installs, prepares the schema before the app runs:

```sh
rfnry-engine create --schema engine          # the schema and its tables
rfnry-engine create --schema engine --sql    # print every statement, run nothing
```

`create` makes `engine_runs` and `engine_labels` and their indexes — among them `(scope,
parent_run_id)`, which a run's tree is found by, and a GIN index on the `tags` column of
`engine_runs`, which tag selection uses — and `engine_migrations`, its record of the migrations run.
While tables `initialize()` made sit in a shared schema such as `public`, `create` refuses and names
the `upgrade --from` that keeps their runs; `create --fresh` starts empty anyway. It takes the
connection string with `--url`, or reads Postgres' standard `PG*` variables, and never prints it.
`--sql` prints the statements as the command runs them — the `search_path` they run under, then one
transaction holding the migration lock — so they land in the same schema when you run them yourself.
Engine's tables never go in `public`. An app that connects as a role of its own, not the one that
ran the command, needs `USAGE` on the schema, `SELECT`, `INSERT`, `UPDATE` and `DELETE` on
`engine_runs` and `engine_labels`, and `SELECT` on `engine_migrations`.

Your pool points at the schema, and the app checks it at start:

```python
pool = await asyncpg.create_pool(dsn, server_settings={"search_path": "engine"})
await Refinery(PostgresStorage(pool)).check()
```

`check()` only reads. It raises a `ConfigurationError` saying how the schema does not fit this
version and the command that fixes it: no engine schema on the pool's `search_path`, a schema behind
or ahead of this version, tables an earlier version's `initialize()` made, or a schema or tables the
pool's role cannot use. When the pool's first schema is another library's, it points at engine's own
schema instead of offering that one.

A version of engine that changes its tables ships a migration, and the schema follows it in place,
its data kept:

```sh
rfnry-engine upgrade --schema engine --sql    # what would run
rfnry-engine upgrade --schema engine          # lists the steps, asks, runs them in one transaction
rfnry-engine check --schema engine            # exits 0 when the schema fits this version
```

Take a snapshot and stop the app instances running the old version first: `upgrade` reminds you,
asks before it runs unless `--yes`, and runs every pending step in one transaction under a lock.

Every run and label a released version recorded reads back in this one: the upgrade to schema
version 3 rewrites each stored body in the shape this version records, whatever release wrote it —
a generation run from before 0.1.4 gets its `sent` in place of the `messages` it held, a key added
since, such as a step's `by` or a run's `settings`, its value for nothing, and a run's `call` the
tags of the run that made it — so `refinery.run`, `export` and `label` read every row, and a body
already in this shape is not touched. A body no release recorded — one written by hand, or cut
short — stops the upgrade before anything changes, naming it: `refinery/migrations: run 'r9' of
scope 'acme' holds a body no release of rfnry-engine recorded (KeyError: 'messages'); delete it
or restore its body, then upgrade again`. The rewrite reads every row once, in pages of a thousand.
The upgrade to schema version 4 drops `engine_runs.task` and rewrites each stored body without its
`task`, the same way: a run records no task. The upgrade to schema version 5 adds the nullable
`engine_runs.mode`; a run recorded before it has no mode.

Tables made by `initialize()` in rfnry-engine 0.1.4 or earlier are adopted where they are:
`rfnry-engine upgrade --schema <their schema>` records them, and gives a table from before 0.1.4 its
`tags` column and the indexes it lacked. It fills `tags` again from each recorded run's own tags, in
a table 0.1.4 made too: before, a character written as an escape starting `\ud` — Hangul, emoji —
was stored rewritten, so two different tags could select each other's runs. A tag holding a NUL or
an unpaired surrogate, which Postgres cannot store, is kept and matched with U+FFFD in its place.
Tables in a shared schema such as `public`, where a pool with the default `search_path` put them,
move into engine's schema in the same transaction:

```sh
rfnry-engine upgrade --schema engine --from public
```

and the pool's `search_path` then names `engine`. Run again, it has nothing left to do; until it
runs, `check()` and the command point at it whenever engine's schema is missing or empty while
tables `initialize()` made sit in another.

A run's row keeps `at`, when the run started, and `recorded`, when the refinery stored it — one
stamp for a run and every child run under it — and `runs(since=)` reads `recorded`. The tables are
prefixed `engine_`, and each rfnry library keeps a schema of its own, so one database holds
rfnry-memory's and rfnry-thread's beside them, each through a pool that names its schema:

```python
memory_pool = await asyncpg.create_pool(dsn, server_settings={"search_path": "memory, public"})
engine_pool = await asyncpg.create_pool(dsn, server_settings={"search_path": "engine"})
memory = Memory(rfnry_memory.PostgresStorage(memory_pool), embedding=emb)
refinery = Refinery(rfnry_engine.PostgresStorage(engine_pool))
```

A storage of your own is any object with the methods below. `scope` comes first in every one, and
each reads and writes that scope only. `refinery.storage` is the storage the refinery was given, to
read its rows back yourself, and `refinery.record(run)` stores a run and every child run under it —
what an engine with a refinery does after each run not made with `record=False`:

```python
class Storage(Protocol):
    async def check(self) -> None
    async def put_runs(self, scope: str, rows: list[RunRow]) -> None  # every row of one recorded tree, in one transaction
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

class RunRow(BaseModel):   scope, run_id, parent_run_id, agent, kind, at, recorded, duration_ms, model, mode, body
class LabelRow(BaseModel): scope, label_id, run_id, source, at, body
class Exported(BaseModel): keys, rows, watermark
class Labeled(BaseModel):  judged, taught, errors
class Conflict(BaseModel): run_id, corrections
```

A row's `body` is the run's record, or the label, as JSON, with a run's children and routed run
left out: each is a row of its own, joined by `parent_run_id`. A generation row's body keeps its
`conversation` and `sent`, the number of its first messages sent at the start, in place of a second
copy as `messages`; `refinery.run` reads `messages` back as `conversation[:sent]`, and
`refinery.record` refuses a run whose `messages` are not the start of its `conversation`. `runs`
returns the newest first and `labels` the oldest first; `unlabeled_by="judge"` keeps the runs no
label from that source names. `run_tree` returns the runs a selector matches — `run_ids`, or `tags`
by containment — and every run under them, oldest first; `refinery.run` reads a run back from it.
`delete_runs` removes the same runs and every label on them in one transaction, for
`refinery.delete_runs`, which checks the selectors before it calls. `Exported.rows` counts the rows
written per dataset; `Conflict.corrections` lists each correction as `(label_id, at, text)`, oldest
first.

## Checking an agents folder

An agents repository checks its own folder in CI with no database and no models:

```sh
rfnry-engine folder check agents      # lists each agent, its kind, model and modes; exits 1 on a ConfigurationError
rfnry-engine folder schema > rfnry.schema.json    # the JSON Schema of rfnry.yaml, for an editor
```

`folder check` builds what `Engine(path)` builds, the default `max_worker_processes` included, and starts no worker; `--max-worker-processes N` sets the ceiling it checks against, for a host that raises it in `Engine(max_worker_processes=)`. A refusal prints one line to stderr.

## Tenancy

`scope` is the tenant. It is required on every run and every refinery verb, it is the first column
of every table, and no read accepts its absence. Tools pinned to a tenant are passed on the call.
`refinery.delete_scope(scope)` removes everything; `refinery.verify_absent(scope)` proves it.
The folder of agents is shared by every tenant: it holds how to do the work, never a tenant's
data, which comes per run, in the input, the history or the workspace. A run's record holds copies
of what it takes from the folder, and a model adapter copies of those, so no tenant's code
reaches another tenant's runs through them. The workers are shared too: a
script's module state — a client it builds when it loads, a cache it fills, a global it sets — is
shared by every tenant whose runs call it, and only `call` is the tenant's: its `scope` and `tags`
always, and the rest of what it carries — a tool call's `run_id`, `agent`, `tool`, `call_id` and `by`
inside a run, a question's `run_id`, `agent` and `question`. A script that keeps anything between calls keys it by
`call["scope"]`, as the engine keys every row.

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
the whole scope.

A run a tool started inside another run's call, such as a consult from `engine.tool`, is a run of
its own, and `delete_runs(run_ids=)` does not reach it: ids reach what a run delegated and routed,
not its consults. Erase those by tags. Give `engine.tool(tags=)`, or the `engine.run(tags=)` of your
own tool, the tags of the person it serves, and `delete_runs(tags=)` removes the consult with its
caller. Datasets you exported before are files you hold — drop the customer's rows from
them yourself; every row carries its run's tags.

## At a glance

Everything the package exports, by role.

- **Run things:** `Engine`, `AgentInfo`, `ModelChoice`, `ModelEntry`, `Stop`, `HistoryLike`,
  `VoiceStart`, `Spoken`, `SpokenLine`.
- **What a run returns:** `Generation`, `Decision`, `Voice`, `Run`, `Step`, `Fingerprint`; events
  `RunStarted`, `TextDelta`, `ReasoningDelta`, `MessagesTaken`, `ToolCallStarted`,
  `ToolResultReady`, `CommandOutput`, `Event`.
- **Speak to models:** `GenerationModel`, `StreamingGenerationModel`, `OnDelta`, `DecisionModel`,
  `VoiceModel`;
  `Message`, `ToolSpec`, `ToolCall`, `Reply`, `Usage`, `State`; parts `Text`, `Image`, `Audio`,
  `Document`, `Part`, `Content`; `Choice`, `Score`, `Check`, `Question`; `ChoiceAnswer`, `ScoreAnswer`,
  `CheckAnswer`, `Answer`, `Decided`; the shapes a question and a result are read by,
  `QuestionLike`, `ReplyLike`, `DecidedLike`, `AnswerLike`; `confidence`.
- **Give models tools:** `Tool`, `Tools`, `EngineTools`, `current_call`, `CallInfo`, `Jail`,
  `Workspace`, `Backend`, `Process`, `Local`.
- **Refine:** `Refinery`, `Review`, `Outcome`, `Label`, `Labeled`, `Exported`, `Conflict`, `Replay`,
  `Bucket`, `is_holdout`.
- **Store:** `Storage`, `PostgresStorage`, `RunRow`, `LabelRow`.
- **Fail loudly:** `EngineException`, `ConfigurationError`, `NotFoundError`, `InputError`,
  `ModelError`, `RunawayError`, `DelegationLoopError`, `DelegationDepthError`, `StorageError`,
  `WorkerError`.

## Errors

`ConfigurationError` for a folder, a model, a tool or a workspace the engine cannot use, its message
naming the file or the dotted path and what to change. When the engine loads: a root that is not a
directory or holds no `rfnry.yaml`; a file that is not UTF-8 or not valid YAML, an anchor, an alias
or a merge key, a key a section or an agent's kind does not take (`manifest: unknown key in
/srv/agents/rfnry.yaml: agents.frontdesk.orchestration.rout (it takes delegate_to, max_depth,
max_parallel, route)`), a key with no value, an empty list or mapping, a `version` other than 1, a
name that is not plain, and an agent with `instructions` neither on its entry nor on every mode; a
`model:` naming no entry under `models:`, a `worker:` naming no worker, a model, a worker or a
declared tool nothing picks, an entry whose provider `providers` does not hold, and workers running
more processes in all than `max_worker_processes`; a path not written from the folder root, a
pattern of `files`, `skills` or `questions` that reaches nothing, one naming `rfnry.yaml`, a file an
agent reaches that links out of the folder, a file an agent is made of that cannot be read, a name
that is not UTF-8, a `.gitignore` that cannot be read or decoded, and a preloaded file `Read` would
refuse; two skills of one name, a `skills.exclude` naming none of the agent's skills, a
`skills.preload` that `exclude` empties, and a `skills.public` name the agent does not offer or one
holding whitespace; a tool's module that is not a `.py` file, is named like a standard-library
module or holds no such function, a schema file that is not an object, holds an unknown key or a
schema that does not compile, a tool nothing describes, a tool named like one the engine gives, a
pattern of `tools.include` or `tools.exclude` that matches nothing, a name in `tools.include` whose
section is missing, `Output` or `History` in either, and one or two of `Shell`, `Job` and `Jobs` in
`tools.exclude` without the rest; a script or a module beside it that is not valid Python, imports
the engine, a module relatively, a module beside it inside a function, a package folder beside it,
a name below a `.py` file beside it, the server's own code, or a module that resolves to nothing
outside `try/except ImportError` and `if TYPE_CHECKING`; a question that is not a JSON object,
holds a key it does not take or one of another type, has no instructions or does not fit its type,
a question in markdown, a question's script without its `.json`, and two questions of one id; a
`delegate_to` or a route naming an agent the file does not hold, a voice agent or an agent that
needs a mode, a route not by a choice question the router asks, by one a script fills, of an option
the question does not have, or to a decision agent with nothing to ask; two agents that would export
under one key; and a `providers` that is not a mapping of names to functions, a `max_model_calls`,
`max_depth` or `max_worker_processes` that is not a positive integer, a `tool_timeout`,
`history_walk` or `history_chars` that is not positive, and a `worker_result_bytes` that is neither
a positive integer nor `None`. When a run starts: a model whose adapter has not its kind's method, a
`ModelChoice` naming a provider you did not give, a `models=` key no agent or mode runs on, a name
in `tools.include` the run does not offer, two tools of a run with one name or one named like a tool
the engine gives, and a missing workspace. And a `Local` root that is not a directory, a `shell`
that is not a list of strings or whose first word is not on `PATH`, a backend without `spawn`,
`read` and `write`, a `Workspace` without a scope, a `timeout` or `output_bytes` that is not
positive, an `approve` that returns anything but `True` or `False`, a judge that is not a decision
agent or a teacher that is not a generation agent. `NotFoundError` for
an agent the folder does not have — a judge, a teacher or a replay's agent in the candidate
included — a mode it does not have, a skill the agent does not have (`engine: agent 'support' has no skill 'refund'; its skills are
escalate`), found when the run starts, or a run the refinery does not. `InputError` for a call the
library refuses: an empty scope, an input that is not a string, a JSON object or a list of JSON
values and parts, a workspace of another scope, a history that breaks a tool call from its result, a
`start` outside the history or without one, tags outside their limits, runtime questions that do not
fit, a runtime id the run already asks, questions for a generation agent, a skill for a decision
agent, a run without a mode of an agent that needs one, a voice agent given to `engine.run` or
`engine.tool` and another kind given to `engine.talk`, a run given a skill both by `skill=` and by an input that starts with a public `/name`
(`engine: a run invokes one skill, by skill= or by an input that starts with /<name>; got
skill='escalate' and /close`), a decision run with no question to ask, media the decision model does
not declare, a media part with both or neither of `data` and `url`, a correction of the wrong shape
or for a question the run was not asked, a `delete_runs` or `verify_absent` given both `run_ids` and
`tags` or an empty one, a `delete_runs` given neither, an `export` or a `replay` given a
`holdout_percent` outside 0 to 100, a `replay` given empty `tags` or a `limit`, `concurrency` or
`repeat` below 1, a tool called with bad arguments, arguments JSON cannot hold or a path outside its
directory, and any use of an engine after `close()` (`engine: this engine was closed; build a new
Engine`). `ModelError` when an adapter breaks the contract — a result whose fields do not fit, a
missing answer, probabilities that do not cover the options, lie outside 0 and 1 or do not sum to 1,
a score outside its levels — or a folder tool's result does not match its `output_schema`, or a
question's script returns anything but an object, a key its JSON sets or one its type does not take,
or a question that does not fit. `RunawayError` when a loop passes its limit of model calls, the
smaller of `Engine(max_model_calls=)`, 200 unless you set another, and its entry's.
`DelegationLoopError` for a route into an agent already in the same chain of routes;
`DelegationDepthError` for nesting past `max_depth`, which reaches the model as a tool error through
`Delegate` and raises through a route. `StorageError` for the database. `WorkerError` for a tool or
a question's script that fails in its worker process: what its script raised, `sys.exit()` included,
a result JSON cannot hold, a result or error over `worker_result_bytes`, a worker that stopped while
the call was in flight (`workers/pool: the worker for lookup_order stopped: exited with code 3`) or
a call in flight when `close()` stopped its worker. Inside a run, a tool's error is the model's to
read, never raised. All under `EngineException` — the root takes the `Exception` suffix, every child
ends in `Error`, the same shape as rfnry-memory's. Every message names the module that raised it,
then says what went wrong in lowercase: `engine: agent 'ghost' was not found`, `tools/workspace:
path outside the workspace: ../x`. A tool's error reaches the model with its type in front:
`InputError: tools: unknown tool 'Ghost'`.