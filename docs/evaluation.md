# Evaluating and improving agents with rfnry

This guide is about knowing whether your agents do the job, and making them better week by week. It
uses the refinery in rfnry-engine, and, where they fit, rfnry-thread for the conversation and
rfnry-memory for knowledge. [`engine.md`](engine.md) is the reference for every verb used here.

## Golden data

An evaluation is only as good as the truth it compares against. In rfnry that truth comes from three
places, and none of them is a model:

- **The right answer.** A person who knows the work writes what the agent should have said: the
  support lead's reply, the team leader's correction in the thread.
- **The correct label.** A person's verdict on what the agent did: approved or not, the right
  department or not, urgent or not.
- **What really happened.** The world's answer: the refund cleared, the ticket reopened, the machine
  started after the fix.

This is golden data. It belongs to the company that produced it, whatever model wrote the draft the
person corrected, and it stays true when the model changes.

Models help with coverage. A **judge** — a decision agent — answers questions about every run, so
the runs nobody reviewed still carry a score; a **teacher** — a generation agent — answers the
questions a decision run was asked. Both are recorded with their source, never as a person's. They
tell you where to look; golden data tells you what is true.

## Capturing the signals

### Keep the run id next to what you show

Every run has a `run_id`. Keep it with whatever the user sees — the suggestion in the support tool,
the message in the thread — so a click, an edit or an outcome later can be attached to the run that
produced it:

```python
run = await engine.run("support", scope="acme", input=ticket_text)
await suggestions.save(ticket_id, text=run.reply, run_id=run.run_id)
```

### What people do with the answer

The strongest signal costs the user nothing: what they did with the agent's work. A support agent
who sends the suggestion unchanged approved it; one who edits it before sending wrote the right
answer; one who discards it objected.

```python
async def on_sent(run_id, suggested, sent):
    if sent == suggested:
        await refinery.review("acme", run_id, Review(stance="approval"))
    else:
        await refinery.review("acme", run_id, Review(correction=sent))

async def on_discarded(run_id):
    await refinery.review("acme", run_id, Review(stance="objection"))
```

### Buttons and corrections

An explicit verdict is a button under the answer. Keep it to one click, and ask why only when the
answer is thumbs down, with a few reasons to choose from and room to write:

```python
async def on_feedback(run_id, up, reason=None):
    stance = "approval" if up else "objection"
    await refinery.review("acme", run_id, Review(stance=stance, comment=reason))
```

A correction is the most valuable label there is: the right answer, written by someone who knows.
In a thread, a team leader who replies to the agent's message with the fix is writing one:

```python
await refinery.review("acme", run_id, Review(stance="objection", correction=leader_text))
```

A decision run takes a correction per question — the department it should have chosen:

```python
await refinery.review("acme", decision.run_id, Review(correction={"intent": "technical"}))
```

### What really happened

Outcomes arrive later, from your systems: the ticket closed or reopened, the order refunded, the
customer escalated. Attach them to the run:

```python
await refinery.outcome("acme", run_id, Outcome(status="failed", detail="Ticket reopened in 3 days."))
```

### How the market does it

The patterns are the same everywhere; rfnry records each of them with its source.

| Pattern | Seen in | In rfnry |
| --- | --- | --- |
| Thumbs up or down under each answer, reasons on thumbs down | Chat assistants, support copilots | `Review(stance=, comment=)` |
| Used as is, edited or discarded | Support copilots, coding assistants | `Review(stance=)` or `Review(correction=)` |
| Resolution, reopen, satisfaction score | Support platforms | `Outcome(status=, detail=)` |
| A review queue: a person scores a sample of runs against a rubric | Evaluation platforms | `Review` on the sampled runs |
| A model scores every run | Evaluation platforms | `refinery.label(judge=)` |

Clicks come mostly from the very happy and the very annoyed. A review queue — a person going
through a random sample every week with the same few questions — is the unbiased picture, and the
one to trust when the two disagree.

### Judges, teachers and conflicts

Run the judge over runs it has not labeled yet, on a schedule of your choosing:

```python
labeled = await refinery.label("acme", engine, judge="judge", teacher="teacher")
```

Check a judge against people before relying on it: on runs that have both a judge score and a
person's verdict, count how often they agree. When two people corrected the same run differently,
`refinery.conflicts("acme")` lists it; a person decides which one stands.

## Three levels of improvement

### Level 1: improve the agent

Most gains come from the agent, not the model. Corrections show which rule is wrong, which knowledge
is missing, which tool fails. The loop:

1. Export and read the runs that people objected to, corrected, or whose outcome failed.
2. Change the folder: a rule, a skill, a knowledge file. With rfnry-memory, a correction that is a
   fact can go straight into the corpus, and the next answer uses it:

   ```python
   await memory.remember("factory-a", "Press 4 hydraulic oil is ISO VG 46, not 68.", subject="press:4")
   ```

3. Replay the holdout on the new folder and compare, as below. Promote the change when it wins.

This works with any model, frontier or local, because nothing is trained.

### Replaying the holdout

About one run in five is held out, by a hash of its run id, so it never moves between exports.
Held-out runs with golden labels are your test set. Replay their inputs on a candidate — a new
folder, a new model — without recording, and grade the replies against the right answers with a
decision agent of your own:

```python
held = [row for row in rows if row["holdout"] and any(l["source"] == "human" for l in row["labels"])]
for row in held:
    history, question = row["messages"][:-1], row["messages"][-1]["content"]
    reply = await candidate.run("support", scope="acme", input=question, history=history, record=False)
    reference = next(l["correction"] or row["reply"] for l in row["labels"] if l["source"] == "human")
    grade = await engine.run("grader", scope="acme", record=False,
                             input={"question": question, "reference": reference, "reply": reply.reply})
    grade.answers["matches"].probability
```

`rows` are the lines of an exported generation file; `grader` is a decision agent of yours whose check
`matches` asks whether the reply says what the reference says. Its probability is a score; the rate across the holdout, next to the rate for the current version, is
the comparison. When you would rather see it live, route a slice of traffic to the candidate with
`tags={"arm": "candidate"}` and compare the golden labels the two arms collect.

### Level 2: choose models on evidence

When a new model ships, replay the same holdout with it. Build an engine on the candidate adapter —
the same folder, the same tools — and compare the grade rate, `reply.usage` and `reply.duration_ms`
against what you run today. The answer to "is the local model good enough for this task?" is a
number measured on your own labeled questions, not a benchmark.

### Level 3: train your own models

Training is worth it when a task has enough golden data and a reason to own the model: privacy,
cost, running without a connection. Train only on what may be trained on. Every row records the
model that produced each run and the source of every label, so the filter is mechanical:

| Part of a row | Written by | Train on it |
| --- | --- | --- |
| The user's message, your documents | Your company and its customers | Yes, within your data-protection duties |
| A correction, a verified decision (`truth_source: "human"`) | A person | Yes |
| An outcome | The world | Yes, as a reward |
| A reply from an open-weight model whose license allows it | That model | Yes |
| A reply from a frontier API model | That provider | Only if the provider's terms allow it |
| A judge's or teacher's answer | A model | Only if that model's terms allow it |

Frontier providers restrict training other models on their outputs; check each provider's terms
before any of its text enters a training set. Decision models are the easiest start: a decision row
whose state is the customer's message and whose truth a person set is training data you own. Every
new checkpoint replays the holdout before it replaces the old one, and the old one stays for
rollback.

A customer who asks to be forgotten is removed by tag: with every run tagged with its customer,
`refinery.delete_runs(scope, tags={"customer": customer_id})` deletes their runs, the runs those
started and every label on them — a dataset or a checkpoint built before still holds them, so
rebuild it from a fresh export.

Retrieval and training do different jobs. Facts that change — manuals, prices, procedures — belong
in rfnry-memory, where a correction takes effect on the next question. Training teaches how to
answer: the terms, the format, when to say the manuals do not cover it.

## What to watch

- **Approval and correction rates** per agent and per version: the share of runs people accepted,
  and the share they rewrote.
- **Outcome success**: the share of runs whose outcome succeeded.
- **Holdout grade rate**: the number every change and every candidate model is compared on.
- **Judge agreement**: how often the judge agrees with people, before trusting its scores.
- **Calibration** of decision models: whether answers given at 0.8 are right about 80% of the time,
  from the decision dataset's probabilities next to the truth.

## Two shapes

**A support copilot.** Agents read the ticket and draft a solution; people send, edit or discard it.
Record what they did with each draft, add a thumbs down with reasons, and attach the ticket's outcome
when it closes or reopens. Level 1 fixes the rules the corrections point at. Level 2 tells you
whether a cheaper model drafts as well. Level 3, when the volume is there, starts with the triage
decisions people verified.

**A knowledge assistant over manuals.** The manuals go into rfnry-memory, one corpus per site. Each
question is a thread; the user marks the answer right or wrong, and a team leader can reply with the
correction for the agent and the person who asked. Right and wrong are reviews, the leader's reply
is a correction, and a correction that is a fact is remembered in the corpus at once. Level 2
answers the question the customer asks first — can a local model answer our questions as well — on
the questions they have already labeled. Level 3 trains the local model on the leaders' corrections
and on the answers it gave that people marked right.

## What rfnry does not do

The buttons, the review queue screen and the dashboards are your product's; rfnry records what they
send. Training is a separate step with the tools your team chooses; rfnry gives it rows that say
where every part came from.
