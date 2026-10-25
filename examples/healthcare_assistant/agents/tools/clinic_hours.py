import json
from pathlib import Path

CLINICS = Path(__file__).parents[2] / "data" / "clinics.json"


def run(args: dict, call: dict) -> dict:
    clinic = json.loads(CLINICS.read_text())["clinics"][call["scope"]]
    day = args.get("day")
    hours = {day: clinic["hours"][day]} if day else clinic["hours"]
    return {"clinic": clinic["name"], "hours": hours}
