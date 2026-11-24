# moon_survey

## Description

A team of agents surveys the Moon's 250 regions. A lead splits the regions among captains, each captain starts
surveyors, and a surveyor writes one region's dossier, and a second agent cross-checks some of them. The lead holds
the survey's plan in its context; the captains, surveyors and crosschecks under it read only the input they are
handed (`context: own` in `agents/rfnry.yaml`). The agents share a Postgres board where a region is claimed, noted
and landed, and the board never lets a landed region be claimed again. Every surveyor command runs in a bubblewrap
sandbox with no network, every tool call passes a fail-closed gate, and a store keeps each conversation in Postgres
so a crashed survey resumes where it stopped. The map is the result: a region's bounds, its measured values, and its
dossier checked against the Lunar Prospector data.

`mode: "fake"` runs every agent on fake models at no cost. `mode: "real"` uses Claude Haiku 5.5, Gemini and TypeSafe.

## Running

Needs Python 3.14, uv, bubblewrap (`bwrap`) and the shared Postgres 18.

```bash
export RFNRY_EXAMPLES_POSTGRES_URL=postgresql://harness:harness@127.0.0.1:55418/harness

# only for modes "real" and "hostile"
# export ANTHROPIC_API_KEY=...
# export GEMINI_API_KEY=...
# export TYPESAFE_API_KEY=...

# optional
# export MOON_SURVEY_WORK=...        where the sandboxes and runs are kept (default: .work/)
# export MOON_SURVEY_DATA=...        where the Lunar Prospector files are (default: data/lp/)
# export MOON_SURVEY_FAKE_SCALE=1    stretches or shrinks the fake models' delays

uv run rfnry-engine create --url "$RFNRY_EXAMPLES_POSTGRES_URL" --schema moon_engine
uv run src/setup.py tables
uv run src/setup.py data
uv run src/main.py
```

The app's tables go in the schema `moon` and engine's in `moon_engine`. `setup.py data` downloads the Lunar Prospector
files and checks them against `data/lp/SOURCES.md`. The service listens on `127.0.0.1:8000`. `.work/` (or
`MOON_SURVEY_WORK`) can be removed once a survey has ended.

`POST /surveys` takes a JSON body of `name` (1 to 64 letters, digits, `.`, `_` or `-`, the first a letter or a digit),
`mode` (`fake`, `real` or `hostile`, which is `real` with a hostile note planted on the board first), `shape` (`crews`:
a lead and its captains; `flat`: one agent for the whole survey; `app`: the app starts the surveyors a boss agent asks
for), `size` (the first regions of the walk, 1 to 250, default 250) or `regions` (a list of region ids), `crews` (the
number of captains, default one for every 25 regions and at least two) and `cap` (the spending cap in dollars). A `real`
or `hostile` survey is refused with 503 while `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` or `TYPESAFE_API_KEY` is not set.
`POST /surveys/{name}/resume` carries a stopped or interrupted survey on, with an optional new `cap`.

On fakes, in the `crews` and `flat` shapes, the gate may answer later: a `fits` probability from `UNSURE` (0.2) up to
`FITS` (0.5), where the gate is unsure, answers `Later()` instead of refusing the call. The call waits for a person, the
run that made it pauses once nothing below it runs, the pause travels up to the lead, and the survey ends `paused`, its
waiting calls in the lead's stored conversation. Its resume has a stand-in person approve each of them, and every paused
captain and surveyor goes on under its own job, so no region starts twice: a resumed run is a new run, which takes the
claim of the run its `RunStarted.resumes` names, and the gate judges its calls against that run's task. A real survey
has no person to hand a call to, and the `app` shape starts each run itself, so there a call below `FITS` is refused.

```bash
uv run src/merge.py <survey> map/moon.json
uv run src/measure.py <survey> <results.md>
```

`merge.py` writes a survey's map, the one `GET /map` shows. `measure.py` appends a survey's numbers to a results file,
as a section titled by the survey's name or by a third argument.

## Example

Start a small fake survey: four regions, two crews, a spending cap of 1 dollar.

```bash
curl -s -X POST localhost:8000/surveys \
  -d '{"name": "readme-fake", "mode": "fake", "shape": "crews", "size": 4, "cap": 1}'
```

```json
{"survey":"readme-fake","mode":"fake","shape":"crews","crews":2,"status":"running","counts":{"landed":0,"failed":0,"claimed":0,"free":4},"spend":0.0,"cap":1.0,"error":null,"reply":null}
```

Read it until its status is `done`:

```bash
curl -s localhost:8000/surveys/readme-fake
```

```json
{"survey":"readme-fake","mode":"fake","shape":"crews","crews":2,"status":"done","counts":{"landed":4,"failed":0,"claimed":0,"free":0},"spend":0.0,"cap":1.0,"error":null,"reply":"survey done: 2 captains: 2 finished, 0 failed, 0 lost, 0 stopped"}
```

Read its map:

```bash
curl -s "localhost:8000/map?survey=readme-fake"
```

```json
{"survey":"readme-fake","made":"2026-11-23T07:01:09.969596+00:00","regions":[{"region":"S75-90/0","bounds":{"lat_min":-90.0,"lat_max":-75.0,"lon_min":-180.0,"lon_max":180.0},"measured":{"FeO":{"mean":5.282289230769231,"error":0.033179831567982446,"pixels":195,"min":2.8418,"p10":4.06902,"p90":6.88676,"max":10.027,"unit":"wt%"},"TiO2":{...},"Al2O3":{...},"Th":{...},"H":{...}},"dossier":null,"checks":null,"data_check":null,"truth":null,"disputes":null,"provider":"fake"}, ...]}
```

A fake survey's regions land with `provider: "fake"` and are not shown: their dossiers are null, because a fake model
writes a stand-in, not a finding. A real survey's regions carry their dossier, its checks, its data check and its
disputes. `map/moon.json` holds the map of the real survey.
