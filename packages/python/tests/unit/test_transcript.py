from rfnry_engine import chain
from rfnry_engine.conversation import Conversation
from rfnry_engine.models import Message, Pause, ToolCall, Waiting
from rfnry_engine.transcript import Transcript

from .fakes import MemoryStore

KEY = ("acme", "c1")
HEAD = chain.opening(None, "Talk.", [], chain.preload_digest([]))
WORK = Message(role="user", content="work")
NOTICE = Message(role="user", content="[a helper finished]")


def stored(store: MemoryStore) -> Transcript:
    transcript = Transcript(Conversation(store, *KEY))
    transcript.start([], [], [WORK], None, HEAD, chain.Chain(HEAD))
    return transcript


def nothing_open() -> Pause:
    return Pause(agent="chat", run_id="r1")


def holding() -> Pause:
    return Pause(agent="chat", run_id="r1", held=[NOTICE])


def waiting() -> Pause:
    call = ToolCall(id="c1", name="refund", args={})
    return Pause(
        agent="chat",
        run_id="r1",
        waiting=[Waiting(id="w1", run_id="r1", agent="chat", call=call)],
    )


def marked(pause: Pause) -> Message:
    return Message(role="assistant", content="[paused]", pause=pause)


async def test_mark_appends_the_pause_as_the_last_message() -> None:
    store = MemoryStore()
    transcript = stored(store)
    pause = nothing_open()
    transcript.mark(pause)
    assert transcript.messages == [WORK, marked(pause)]
    assert transcript.open_mark is pause and transcript.open_at == 1
    assert transcript.after_mark() == [] and not transcript.past_mark()
    transcript.messages.append(NOTICE)
    assert transcript.after_mark() == [NOTICE] and transcript.past_mark()
    await transcript.save()
    assert store.held[KEY] == [WORK, marked(pause), NOTICE]


async def test_remark_appends_only_an_open_pause_or_one_that_holds_notices() -> None:
    store = MemoryStore()
    transcript = stored(store)
    transcript.remark(nothing_open())
    assert transcript.messages == [WORK] and transcript.open_mark is None
    held, paused = holding(), waiting()
    transcript.remark(held)
    transcript.remark(paused)
    assert transcript.messages == [WORK, marked(held), marked(paused)]
    assert transcript.open_mark is paused and transcript.open_at == 2
    await transcript.save()
    assert store.held[KEY] == [WORK, marked(held), marked(paused)]


async def test_close_appends_an_open_pause_and_spills_the_notices_of_one_with_nothing_open() -> (
    None
):
    store = MemoryStore()
    transcript = stored(store)
    paused = waiting()
    transcript.close(paused)
    transcript.close(holding())
    assert transcript.messages == [WORK, marked(paused), NOTICE]
    assert transcript.open_mark is paused and transcript.open_at == 1
    await transcript.save()
    assert store.held[KEY] == [WORK, marked(paused), NOTICE]


async def test_forget_mark_leaves_the_messages_and_drops_the_mark_position() -> None:
    store = MemoryStore()
    pause = waiting()
    store.held[KEY] = [
        WORK,
        Message(role="assistant", content="", tool_calls=[pause.waiting[0].call]),
        marked(pause),
    ]
    conversation = Conversation(store, *KEY)
    history = await conversation.opened()
    transcript = Transcript(conversation)
    transcript.start(history, [], [WORK], pause, HEAD, chain.Chain(HEAD))
    assert transcript.open_mark == pause and transcript.open_at == 2
    transcript.messages.append(NOTICE)
    assert transcript.after_mark() == [NOTICE]
    transcript.forget_mark()
    assert transcript.open_mark is None and transcript.after_mark() == []
    assert transcript.messages == [*history, NOTICE]
    await transcript.save()
    assert store.held[KEY] == [*history, NOTICE]


async def test_without_a_store_seen_is_the_messages_list_itself() -> None:
    transcript = Transcript(None)
    transcript.start([NOTICE], [], [WORK], None, HEAD, chain.Chain(HEAD))
    assert transcript.key is None and transcript.lost is None
    await transcript.windowed(await transcript.position())
    assert transcript.seen is transcript.messages
    assert transcript.sent == [NOTICE, WORK] and transcript.sent is not transcript.messages
    reply = Message(role="assistant", content="done")
    transcript.messages.append(reply)
    await transcript.windowed(await transcript.position())
    assert transcript.seen is transcript.messages and transcript.seen[-1] is reply
    assert transcript.sent == [NOTICE, WORK]
    await transcript.save()
