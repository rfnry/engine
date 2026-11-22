import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from rfnry_engine import CheckAnswer, Message, Notice, Reply, ToolCall, ToolSpec

import moon
from fakes import FakeGate, Limits, Profile, fakes
from providers import ProviderError
from tools import FakeWeb

HERE = Path(__file__).resolve().parent.parent
GRS = "https://pds-geosciences.wustl.edu/lunar/page"
REGION = "N15-30/2"


def profile(scale=0.0, **kw):
    return Profile(
        source=kw.pop("source", "test"),
        latency=kw.pop("latency", {}),
        size=kw.pop("size", {}),
        scale=scale,
    )


def build(prof=None, limits=None, pages=None, sleep=None):
    web = FakeWeb({GRS: ("text/html", b"<p>lunar</p>")} if pages is None else pages)
    extra = {} if sleep is None else {"sleep": sleep}
    return fakes(prof or profile(), limits or Limits(None), web=web, **extra)


def person(text):
    return Message(role="user", content=text)


def notice(job, agent, kind="finished"):
    return Message(
        role="user",
        content=f"[helper {job} · {agent} · {kind}]",
        notice=Notice(
            job=job, agent=agent, run_id=None if kind == "lost" else f"r-{job}", kind=kind
        ),
    )


def survey(regions, shares=None, second=()):
    given = {
        "name": "s",
        "regions": regions,
        "shares": shares if shares is not None else [regions],
        "second": list(second),
    }
    return f"<<<AGENT>>>\nlead\n\n<<<CONTEXT>>>\n{json.dumps(given)}"


START = "start the survey"


class Convo:
    def __init__(self, model, first, system="system"):
        self.model = model
        self.system = system
        self.messages = [person(first)]
        self.n = 0

    async def next(self):
        reply = await self.model.generate(self.system, self.messages, [])
        self.messages.append(
            Message(
                role="assistant",
                content=reply.text,
                tool_calls=reply.tool_calls,
                usage=reply.usage,
                model=reply.model,
            )
        )
        return reply

    def answer(self, reply, *contents, error=False):
        for call, content in zip(reply.tool_calls, contents, strict=True):
            body = content if isinstance(content, str) else json.dumps(content)
            self.messages.append(
                Message(role="tool", tool_call_id=call.id, content=body, is_error=error)
            )

    def tell(self, job, agent, kind="finished"):
        self.messages.append(notice(job, agent, kind))


def names(reply):
    return [call.name for call in reply.tool_calls]


async def shell(command, cwd):
    env = {
        **os.environ,
        "RFNRY_TREE": str(HERE / "agents"),
        "MOON_SURVEY_SRC": str(HERE / "src"),
    }
    process = await asyncio.create_subprocess_shell(
        command.replace("python ", f"{sys.executable} ", 1),
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await process.communicate()
    return {
        "exit_code": process.returncode,
        "stdout": out.decode(),
        "stderr": err.decode(),
        "timed_out": False,
    }


async def run_surveyor(tmp_path, region=REGION, claim=None, pages=None):
    convo = Convo(build(pages=pages)["surveyor"], json.dumps({"region": region}))
    seen = []
    while True:
        reply = await convo.next()
        if not reply.tool_calls:
            return convo, seen, reply
        seen.append(reply)
        results = []
        for call in reply.tool_calls:
            if call.name == "claim":
                results.append(claim or {"claimed": True, "by": "me"})
            elif call.name == "region":
                found = moon.region(region)
                results.append(
                    {
                        "region": region,
                        "centre": list(found.centre),
                        "neighbours": list(found.neighbours),
                        "crosscheck": "crosscheck",
                    }
                )
            elif call.name == "notes":
                results.append({"notes": []})
            elif call.name == "fetch":
                results.append({"url": call.args["url"], "status": 200, "text": "lunar"})
            elif call.name == "Shell":
                results.append(await shell(call.args["command"], tmp_path))
            elif call.name == "Write":
                (tmp_path / call.args["path"]).write_text(call.args["content"])
                results.append("ok")
            elif call.name == "note":
                results.append({"note": 1})
            else:
                raise AssertionError(call.name)
        convo.answer(reply, *results)


async def test_the_surveyor_claims_reads_measures_writes_checks_and_notes(tmp_path):
    convo, seen, last = await run_surveyor(tmp_path)
    order = [names(reply) for reply in seen]
    neighbours = list(moon.region(REGION).neighbours)
    assert order == [
        ["claim"],
        ["region"],
        ["notes"] * len(neighbours),
        ["Shell"],
        ["fetch"],
        ["Write"],
        ["Shell"],
        ["note"],
    ]
    assert seen[0].tool_calls[0].args == {"region": REGION}
    assert [c.args["region"] for c in seen[2].tool_calls] == neighbours
    assert "measure.py" in seen[3].tool_calls[0].args["command"]
    assert seen[4].tool_calls[0].args == {"url": GRS}
    assert seen[5].tool_calls[0].args["path"] == "dossier.json"
    assert 'check.py" dossier.json' in seen[6].tool_calls[0].args["command"]
    assert seen[7].tool_calls[0].args["kind"] == "finding"
    assert seen[7].tool_calls[0].args["region"] == REGION
    assert last.stop_reason == "end_turn" and last.text and "\n" not in last.text
    checked = await shell(seen[6].tool_calls[0].args["command"], tmp_path)
    assert checked["exit_code"] == 0 and checked["stdout"] == ""


async def test_the_surveyors_dossier_comes_from_measure_and_is_simulated(tmp_path):
    await run_surveyor(tmp_path)
    dossier = json.loads((tmp_path / "dossier.json").read_text())
    found = moon.measured(REGION)
    assert dossier["simulated"] is True and dossier["region"] == REGION
    for name in ("FeO", "TiO2", "Al2O3", "Th"):
        value = dossier["values"][name]
        mean, error = found[name].mean, found[name].error
        assert value["low"] == pytest.approx(max(0.0, mean - 2 * error))
        assert value["high"] == pytest.approx(mean + 2 * error)
        assert any("Lunar Prospector" in s and "data/lp" in s for s in value["sources"])
    weh = found["H"].mean * 9 / 10_000
    assert dossier["values"]["water_ice"]["low"] <= weh
    assert dossier["values"]["He3"]["unknown"]


def held(*exchanges):
    out = []
    for n, (name, args, body) in enumerate(exchanges):
        out.append(
            Message(role="assistant", tool_calls=[ToolCall(id=f"c{n}", name=name, args=args)])
        )
        out.append(Message(role="tool", tool_call_id=f"c{n}", content=json.dumps(body)))
    return out


async def test_the_surveyors_dossier_marks_unknown_where_a_measure_is_missing():
    measure = {
        "region": REGION,
        "values": {
            "FeO": {"mean": 5.0, "error": 0.1, "pixels": 3},
            "TiO2": {"mean": 1.0, "error": 0.1, "pixels": 3},
            "Al2O3": {"mean": 20.0, "error": 0.1, "pixels": 3},
            "Th": None,
            "H": None,
        },
    }
    messages = [person(json.dumps({"region": REGION}))] + held(
        ("claim", {"region": REGION}, {"claimed": True, "by": "me"}),
        ("region", {"region": REGION}, {"centre": [0, 0], "neighbours": []}),
        ("fetch", {"url": GRS}, {"status": 200}),
        (
            "Shell",
            {"command": "python measure.py x"},
            {"exit_code": 0, "stdout": json.dumps(measure), "stderr": ""},
        ),
    )
    reply = await build()["surveyor"].generate("s", messages, [])
    assert names(reply) == ["Write"]
    dossier = json.loads(reply.tool_calls[0].args["content"])
    assert dossier["values"]["water_ice"]["unknown"]
    assert dossier["values"]["Th"]["unknown"]
    assert dossier["values"]["FeO"]["high"] == pytest.approx(5.2)
    assert dossier["terrain"] == "highland"


async def test_a_surveyor_that_does_not_win_its_claim_ends_naming_the_holder(tmp_path):
    convo, seen, last = await run_surveyor(tmp_path, claim={"claimed": False, "by": "run-77"})
    assert [names(r) for r in seen] == [["claim"]]
    assert "run-77" in last.text and "\n" not in last.text
    assert not (tmp_path / "dossier.json").exists()


async def test_a_surveyor_ends_quoting_any_tool_error():
    model = build()["surveyor"]
    convo = Convo(model, json.dumps({"region": REGION}))
    reply = await convo.next()
    convo.answer(reply, "NotApproved: claim was not approved", error=True)
    final = await convo.next()
    assert not final.tool_calls
    assert "NotApproved: claim was not approved" in final.text
    assert "\n" not in final.text


async def test_a_surveyor_skips_the_fetch_when_the_web_has_no_page(tmp_path):
    convo, seen, _ = await run_surveyor(tmp_path, pages={})
    assert "fetch" not in [n for reply in seen for n in names(reply)]


def captain(shares=("R1", "R2", "R3"), second=("R2",)):
    return Convo(build()["captain"], json.dumps({"share": list(shares), "second": list(second)}))


async def test_a_captain_starts_every_surveyor_of_its_share_in_one_reply():
    convo = captain()
    reply = await convo.next()
    assert names(reply) == ["Delegate"] * 3
    assert [json.loads(c.args["input"]) for c in reply.tool_calls] == [
        {"region": "R1"},
        {"region": "R2"},
        {"region": "R3"},
    ]
    assert all(
        c.args["agent"] == "surveyor" and c.args["background"] is True for c in reply.tool_calls
    )


async def test_a_captain_waits_with_no_tool_call_while_its_helpers_run():
    convo = captain()
    reply = await convo.next()
    convo.answer(reply, {"job": "d1"}, {"job": "d2"}, {"job": "d3"})
    waiting = await convo.next()
    assert waiting.tool_calls == [] and waiting.stop_reason == "end_turn"
    convo.tell("d1", "surveyor")
    reply = await convo.next()
    assert names(reply) == ["Delegate"]
    assert reply.tool_calls[0].args == {
        "agent": "crosscheck",
        "input": json.dumps({"region": "R1"}),
        "background": True,
    }
    convo.answer(reply, {"job": "d4"})
    assert (await convo.next()).tool_calls == []


async def test_a_captain_reads_a_notice_by_its_first_line_as_the_engine_sends_it():
    convo = captain()
    reply = await convo.next()
    convo.answer(reply, {"job": "d1"}, {"job": "d2"}, {"job": "d3"})
    convo.messages.append(person("[helper d1 · surveyor · finished]\nR1: dossier written"))
    reply = await convo.next()
    assert names(reply) == ["Delegate"]
    assert json.loads(reply.tool_calls[0].args["input"]) == {"region": "R1"}
    for job in ("d2", "d3"):
        convo.messages.append(person(f"[helper {job} · surveyor · lost]\nwhat it did is unknown"))
    convo.answer(reply, {"job": "d4"})
    convo.messages.append(person("[helper d4 · crosscheck · finished]\nR1: agree"))
    note = await convo.next()
    assert names(note) == ["note"] and "1 finished" in note.tool_calls[0].args["text"]
    assert "2 lost" in note.tool_calls[0].args["text"]


async def test_a_captain_takes_the_second_crosscheck_for_a_region_of_its_second_list():
    convo = captain()
    reply = await convo.next()
    convo.answer(reply, {"job": "d1"}, {"job": "d2"}, {"job": "d3"})
    convo.tell("d1", "surveyor")
    convo.tell("d2", "surveyor")
    reply = await convo.next()
    assert [c.args["agent"] for c in reply.tool_calls] == ["crosscheck", "crosscheck_second"]
    assert [json.loads(c.args["input"])["region"] for c in reply.tool_calls] == ["R1", "R2"]


async def test_a_captain_ends_with_a_summary_note_and_a_verdict_once_every_job_has_a_notice():
    convo = captain()
    reply = await convo.next()
    convo.answer(reply, {"job": "d1"}, {"job": "d2"}, {"job": "d3"})
    convo.tell("d1", "surveyor")
    convo.tell("d2", "surveyor", "failed")
    convo.tell("d3", "surveyor", "lost")
    reply = await convo.next()
    assert [json.loads(c.args["input"])["region"] for c in reply.tool_calls] == ["R1"]
    convo.answer(reply, {"job": "d4"})
    assert (await convo.next()).tool_calls == []
    convo.tell("d4", "crosscheck")
    reply = await convo.next()
    assert names(reply) == ["note"]
    args = reply.tool_calls[0].args
    assert args["kind"] == "summary" and args["region"] == "R1"
    assert "1 finished" in args["text"] and "1 failed" in args["text"]
    assert "1 lost" in args["text"]
    convo.answer(reply, {"note": 1})
    final = await convo.next()
    assert final.tool_calls == [] and final.text and "\n" not in final.text


async def test_a_helper_that_could_not_start_counts_as_ended():
    convo = captain(shares=("R1",), second=())
    reply = await convo.next()
    convo.answer(reply, "ConfigurationError: past max_background", error=True)
    reply = await convo.next()
    assert names(reply) == ["note"]
    assert "1 failed" in reply.tool_calls[0].args["text"]


async def test_a_resumed_captain_goes_on_from_the_conversation_it_is_given():
    convo = captain()
    reply = await convo.next()
    convo.answer(reply, {"job": "d1"}, {"job": "d2"}, {"job": "d3"})
    convo.tell("d1", "surveyor")
    first = await build()["captain"].generate("s", convo.messages, [])
    again = await build()["captain"].generate("s", list(convo.messages), [])
    assert first == again and names(first) == ["Delegate"]


async def test_the_lead_starts_a_captain_for_each_share():
    models = build()
    plan = survey(["R1", "R2", "R3", "R4"], [["R1", "R2"], ["R3", "R4"]], ["R2", "R4"])
    convo = Convo(models["lead"], START, system=plan)
    reply = await convo.next()
    assert [c.args["agent"] for c in reply.tool_calls] == ["captain", "captain"]
    assert all(c.args["background"] is True for c in reply.tool_calls)
    assert [json.loads(c.args["input"]) for c in reply.tool_calls] == [
        {"share": ["R1", "R2"], "second": ["R2"]},
        {"share": ["R3", "R4"], "second": ["R4"]},
    ]
    convo.answer(reply, {"job": "d1"}, {"job": "d2"})
    assert (await convo.next()).tool_calls == []
    convo.tell("d1", "captain")
    assert (await convo.next()).tool_calls == []
    convo.tell("d2", "captain", "failed")
    note = await convo.next()
    assert note.tool_calls[0].args["kind"] == "summary"
    assert "2 captains" in note.tool_calls[0].args["text"]
    convo.answer(note, {"note": 1})
    assert (await convo.next()).tool_calls == []


async def test_a_lead_given_a_later_input_starts_captains_for_the_regions_left():
    models = build()
    plan = survey(["R1", "R2", "R3", "R4"], [["R1", "R2"], ["R3", "R4"]], ["R2", "R4"])
    convo = Convo(models["lead"], START, system=plan)
    reply = await convo.next()
    convo.answer(reply, {"job": "d1"}, {"job": "d2"})
    convo.tell("d1", "captain")
    convo.tell("d2", "captain", "lost")
    convo.system = survey(["R3", "R4"], [["R3"], ["R4"]], ["R4"])
    convo.messages.append(person("the process restarted; carry on"))
    reply = await convo.next()
    assert [c.args["agent"] for c in reply.tool_calls] == ["captain", "captain"]
    assert [json.loads(c.args["input"]) for c in reply.tool_calls] == [
        {"share": ["R3"], "second": []},
        {"share": ["R4"], "second": ["R4"]},
    ]
    convo.answer(reply, {"job": "d3"}, {"job": "d4"})
    assert (await convo.next()).tool_calls == []
    convo.tell("d3", "captain")
    convo.tell("d4", "captain")
    note = await convo.next()
    args = note.tool_calls[0].args
    assert args["region"] == "R3" and "2 captains: 2 finished, 0 failed, 0 lost" in args["text"]


async def test_flat_given_a_later_input_starts_surveyors_for_the_regions_left():
    models = build()
    convo = Convo(models["flat"], START, system=survey(["R1", "R2"]))
    reply = await convo.next()
    convo.answer(reply, {"job": "d1"}, {"job": "d2"})
    convo.tell("d1", "surveyor", "lost")
    convo.tell("d2", "surveyor", "lost")
    convo.system = survey(["R2"])
    convo.messages.append(person("the process restarted; carry on"))
    reply = await convo.next()
    assert [json.loads(c.args["input"]) for c in reply.tool_calls] == [{"region": "R2"}]


async def test_a_lead_without_its_survey_in_the_context_is_refused():
    with pytest.raises(Exception, match="fakes: the lead's survey is not in its context"):
        await build()["lead"].generate("system", [person(START)], [])


async def test_a_lead_whose_window_moved_past_its_input_reads_its_jobs():
    plan = survey(["R1", "R2", "R3", "R4"], [["R1", "R2"], ["R3", "R4"]])
    convo = Convo(build()["lead"], START, system=plan)
    convo.messages = [notice("d1", "captain")]
    reply = await convo.next()
    assert names(reply) == ["Jobs"]
    convo.answer(
        reply,
        {
            "jobs": [
                {"job": "d1", "agent": "captain", "running": False, "ended": "finished"},
                {"job": "d2", "agent": "captain", "running": True},
            ]
        },
    )
    waiting = await convo.next()
    assert waiting.tool_calls == [] and "1 helpers" in waiting.text
    convo.tell("d2", "captain")
    note = await convo.next()
    assert names(note) == ["note"] and note.tool_calls[0].args["region"] == "R1"
    assert "2 captains: 2 finished" in note.tool_calls[0].args["text"]


async def test_a_captain_whose_window_moved_crosschecks_from_its_notices():
    plan = survey(["R1", "R2"], [["R1", "R2"]], ["R2"])
    convo = Convo(build()["captain"], json.dumps({"share": ["R1", "R2"]}), system=plan)
    convo.messages = [
        person("[helper d1 · surveyor · finished]\nN00-15/3: dossier written"),
        person("[helper d2 · surveyor · finished]\nN00-15/4: dossier written"),
    ]
    plan = survey(["N00-15/3", "N00-15/4"], second=["N00-15/4"])
    convo.system = plan
    reply = await convo.next()
    assert names(reply) == ["Jobs"]
    convo.answer(
        reply,
        {
            "jobs": [
                {"job": "d1", "agent": "surveyor", "running": False, "ended": "finished"},
                {"job": "d2", "agent": "surveyor", "running": False, "ended": "finished"},
            ]
        },
    )
    reply = await convo.next()
    assert [(c.args["agent"], json.loads(c.args["input"])) for c in reply.tool_calls] == [
        ("crosscheck", {"region": "N00-15/3"}),
        ("crosscheck_second", {"region": "N00-15/4"}),
    ]
    convo.answer(reply, {"job": "d3"}, {"job": "d4"})
    convo.tell("d3", "crosscheck")
    convo.tell("d4", "crosscheck_second")
    note = await convo.next()
    assert names(note) == ["note"] and note.tool_calls[0].args["region"] == "N00-15/3"


async def test_flat_starts_a_surveyor_for_each_region_and_crosschecks_its_finished():
    models = build()
    convo = Convo(models["flat"], START, system=survey(["R1", "R2", "R3"], second=["R3"]))
    reply = await convo.next()
    assert [c.args["agent"] for c in reply.tool_calls] == ["surveyor"] * 3
    assert [json.loads(c.args["input"]) for c in reply.tool_calls] == [
        {"region": "R1"},
        {"region": "R2"},
        {"region": "R3"},
    ]
    convo.answer(reply, {"job": "d1"}, {"job": "d2"}, {"job": "d3"})
    convo.tell("d3", "surveyor")
    reply = await convo.next()
    assert reply.tool_calls[0].args["agent"] == "crosscheck_second"


async def test_the_crosscheck_reads_the_dossier_and_the_neighbours_notes_then_agrees():
    convo = Convo(build()["crosscheck"], json.dumps({"region": REGION}))
    reply = await convo.next()
    assert names(reply) == ["dossier"] and reply.tool_calls[0].args == {"region": REGION}
    convo.answer(reply, {"region": REGION, "landed": {"region": REGION}, "measured": {}})
    reply = await convo.next()
    assert names(reply) == ["region"]
    convo.answer(reply, {"region": REGION, "neighbours": ["N15-30/1", "N15-30/3"]})
    reply = await convo.next()
    assert names(reply) == ["notes", "notes"]
    assert [c.args["region"] for c in reply.tool_calls] == ["N15-30/1", "N15-30/3"]
    convo.answer(reply, {"notes": [{"kind": "finding"}]}, {"notes": [{"kind": "finding"}]})
    reply = await convo.next()
    assert names(reply) == ["note"]
    args = reply.tool_calls[0].args
    assert (args["region"], args["kind"]) == (REGION, "agree")
    assert "2 notes" in args["text"]
    convo.answer(reply, {"note": 3})
    final = await convo.next()
    assert final.tool_calls == [] and final.text and "\n" not in final.text


async def test_a_crosscheck_with_no_landed_dossier_ends_at_once():
    convo = Convo(build()["crosscheck"], json.dumps({"region": REGION}))
    reply = await convo.next()
    convo.answer(reply, {"region": REGION, "landed": None, "measured": {}})
    final = await convo.next()
    assert final.tool_calls == [] and "no landed dossier" in final.text


async def test_the_second_crosscheck_plays_the_same_script():
    convo = Convo(build()["crosscheck_second"], json.dumps({"region": REGION}))
    assert names(await convo.next()) == ["dossier"]


async def test_the_crosscheck_ends_quoting_an_error():
    convo = Convo(build()["crosscheck"], json.dumps({"region": REGION}))
    reply = await convo.next()
    convo.answer(reply, "BoardError: board: no such region", error=True)
    final = await convo.next()
    assert final.tool_calls == [] and "BoardError: board: no such region" in final.text


async def test_the_boss_checks_progress_starts_the_free_regions_and_checks_their_jobs():
    convo = Convo(build()["boss"], "{}")
    reply = await convo.next()
    assert names(reply) == ["progress"]
    convo.answer(reply, {"counts": {"landed": 0, "free": 2}, "free": ["R1", "R2"]})
    reply = await convo.next()
    assert names(reply) == ["start", "start"]
    assert [c.args for c in reply.tool_calls] == [{"region": "R1"}, {"region": "R2"}]
    convo.answer(reply, {"job": "d1"}, {"job": "d2"})
    reply = await convo.next()
    assert names(reply) == ["check", "check"]
    assert [c.args for c in reply.tool_calls] == [{"job": "d1"}, {"job": "d2"}]
    convo.answer(reply, {"running": False}, {"running": True})
    reply = await convo.next()
    assert [c.args for c in reply.tool_calls] == [{"job": "d2"}]
    convo.answer(reply, {"running": False})
    reply = await convo.next()
    assert names(reply) == ["progress"]
    convo.answer(reply, {"counts": {"landed": 2, "free": 0}, "free": []})
    final = await convo.next()
    assert final.tool_calls == [] and "landed" in final.text and "\n" not in final.text


async def test_the_boss_ends_quoting_a_tool_error():
    convo = Convo(build()["boss"], "{}")
    reply = await convo.next()
    convo.answer(reply, "NotApproved: progress was not approved", error=True)
    final = await convo.next()
    assert final.tool_calls == [] and "NotApproved: progress was not approved" in final.text


def questions():
    return {"fits": object()}


async def test_the_gate_says_yes_to_a_fitting_call():
    decided = await build()["gate"].decide({"tool": "claim", "args": {}}, questions())
    assert decided.answers["fits"] == CheckAnswer(probability=1.0)
    assert decided.model == "fake" and decided.usage.input > 0


async def test_the_gate_says_no_to_a_named_tool():
    gate = FakeGate(refuse=frozenset({"Shell"}))
    refused = await gate.decide({"tool": "Shell", "args": {}}, questions())
    allowed = await gate.decide({"tool": "Write", "args": {}}, questions())
    assert refused.answers["fits"] == CheckAnswer(probability=0.0)
    assert allowed.answers["fits"] == CheckAnswer(probability=1.0)


async def test_the_gate_raises_a_provider_error_when_down():
    with pytest.raises(ProviderError, match="providers: "):
        await FakeGate(down=True).decide({"tool": "claim"}, questions())


async def test_the_gate_reads_the_tool_of_the_state_and_nothing_else():
    gate = FakeGate(refuse=frozenset({"Shell"}))
    for state in ({"name": "Shell"}, {"call": {"tool": "Shell"}}, "Shell", {"tool": 3}):
        decided = await gate.decide(state, questions())
        assert decided.answers["fits"] == CheckAnswer(probability=1.0)


async def test_the_fakes_are_keyed_by_the_folders_model_names():
    assert set(build()) == {
        "lead",
        "flat",
        "captain",
        "surveyor",
        "crosscheck",
        "crosscheck_second",
        "boss",
        "gate",
    }


async def test_usage_is_characters_over_four_and_the_model_is_fake():
    model = build()["crosscheck"]
    first = json.dumps({"region": REGION})
    messages = [person(first)]
    tools = [ToolSpec(name="notes", description="read", input_schema={"type": "object"})]
    reply = await model.generate("be brief", messages, tools)
    assert isinstance(reply, Reply) and reply.model == "fake"
    sent = (
        len("be brief")
        + len(first)
        + len("notes")
        + len("read")
        + len(json.dumps({"type": "object"}))
    )
    call = reply.tool_calls[0]
    made = len(reply.text) + len(json.dumps({"name": call.name, "args": call.args}))
    assert reply.usage.input == sent // 4
    assert reply.usage.output == made // 4
    assert reply.usage.input > 0 and reply.usage.output > 0


async def test_a_reply_waits_a_latency_drawn_from_the_profile_and_repeats():
    slept = []

    async def sleep(seconds):
        slept.append(seconds)

    prof = profile(scale=1.0, latency={"crosscheck": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})
    model = build(prof, sleep=sleep)["crosscheck"]
    draws = []
    for region in ("R1", "R2", "R3", "R4", "R5", "R6"):
        await model.generate("s", [person(json.dumps({"region": region}))], [])
        draws.append(slept[-1])
    assert all(d in prof.latency["crosscheck"] for d in draws)
    assert len(set(draws)) > 1
    again = build(prof, sleep=sleep)["crosscheck"]
    await again.generate("s", [person(json.dumps({"region": "R1"}))], [])
    assert slept[-1] == draws[0]

    halved = build(profile(scale=0.5, latency=prof.latency), sleep=sleep)["crosscheck"]
    await halved.generate("s", [person(json.dumps({"region": "R1"}))], [])
    assert slept[-1] == draws[0] / 2


async def test_the_profile_loads_with_a_scale_and_labels_its_placeholders():
    path = HERE / "data" / "fake_profile.json"
    loaded = Profile.load(path)
    assert loaded.source == "placeholder until the calibration"
    assert loaded.scale == 1.0
    assert Profile.load(path, scale=0).scale == 0
    for name in ("lead", "flat", "captain", "surveyor", "crosscheck", "crosscheck_second", "boss"):
        assert loaded.latency[name] and loaded.size[name]
    assert json.loads(path.read_text())["source"] == "placeholder until the calibration"


async def test_a_scale_of_zero_runs_instantly():
    slept = []

    async def sleep(seconds):
        slept.append(seconds)

    loaded = Profile.load(HERE / "data" / "fake_profile.json", scale=0)
    model = build(loaded, sleep=sleep)["surveyor"]
    await model.generate("s", [person(json.dumps({"region": REGION}))], [])
    assert slept == [0]


class Clock:
    def __init__(self):
        self.now = 0.0
        self.slept = []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


async def test_limits_space_calls_and_log_each_wait_as_a_429():
    clock = Clock()
    rows = []

    async def log(provider, model, status, wait, headers):
        rows.append((provider, model, status, wait))

    limits = Limits(60, clock=clock, sleep=clock.sleep, log=log)
    for _ in range(3):
        await limits.wait()
    assert clock.slept == [1.0, 1.0]
    assert rows == [("fake", "fake", 429, 1.0), ("fake", "fake", 429, 1.0)]
    clock.now += 100
    await limits.wait()
    assert len(rows) == 2


async def test_limits_without_a_rate_never_wait():
    clock = Clock()
    limits = Limits(None, clock=clock, sleep=clock.sleep)
    for _ in range(5):
        await limits.wait()
    assert clock.slept == []


async def test_the_second_crosscheck_is_spaced_by_the_limits():
    clock = Clock()
    rows = []

    async def log(*row):
        rows.append(row)

    limits = Limits(60, clock=clock, sleep=clock.sleep, log=log)
    model = build(limits=limits)["crosscheck_second"]
    for _ in range(2):
        await model.generate("s", [person(json.dumps({"region": REGION}))], [])
    assert clock.slept == [1.0] and rows[0][0] == "fake"


async def test_a_refused_crosscheck_is_counted_under_crosschecks_not_surveyors():
    convo = captain(shares=("R1",), second=())
    reply = await convo.next()
    convo.answer(reply, {"job": "d1"})
    convo.tell("d1", "surveyor")
    reply = await convo.next()
    convo.answer(reply, "ConfigurationError: past max_background", error=True)
    note = await convo.next()
    text = note.tool_calls[0].args["text"]
    assert "1 surveyors: 1 finished, 0 failed" in text
    assert "1 crosschecks: 0 finished, 1 failed" in text


async def test_a_delegate_with_no_result_counts_as_lost():
    convo = captain(shares=("R1",), second=())
    reply = await convo.next()
    note = await convo.model.generate("s", convo.messages, [])
    assert names(note) == ["note"]
    assert "1 surveyors: 0 finished, 0 failed, 1 lost" in note.tool_calls[0].args["text"]
    assert reply.tool_calls


async def test_fakes_take_the_gate_it_is_given():
    gate = FakeGate(down=True)
    assert build_with(gate=gate)["gate"] is gate
    assert isinstance(build_with()["gate"], FakeGate)


def build_with(**kw):
    web = FakeWeb({})
    return fakes(profile(), Limits(None), web=web, **kw)


async def test_a_profile_missing_a_role_is_refused_at_load(tmp_path):
    raw = json.loads((HERE / "data" / "fake_profile.json").read_text())
    del raw["latency"]["captain"]
    path = tmp_path / "p.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(Exception, match="fakes: the profile has no latencies for 'captain'"):
        Profile.load(path)
    raw = json.loads((HERE / "data" / "fake_profile.json").read_text())
    del raw["size"]["boss"]
    path.write_text(json.dumps(raw))
    with pytest.raises(Exception, match="fakes: the profile has no sizes for 'boss'"):
        Profile.load(path)


async def test_one_account_limiter_spaces_calls_across_agents():
    clock = Clock()
    rows = []

    async def log(*row):
        rows.append(row)

    account = Limits(60, clock=clock, sleep=clock.sleep, log=log)
    models = build_with(account=account, sleep=clock.sleep)
    first = [person(json.dumps({"region": REGION}))]
    await models["crosscheck"].generate("s", first, [])
    await models["surveyor"].generate("s", first, [])
    await models["boss"].generate("s", [person("{}")], [])
    assert [w for w in clock.slept if w] == [1.0, 1.0]
    assert len(rows) == 2
    await models["crosscheck_second"].generate("s", first, [])
    assert [w for w in clock.slept if w] == [1.0, 1.0]
