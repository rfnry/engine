import json
import math
import statistics
from pathlib import Path

READINGS = Path(__file__).parents[3] / "data" / "readings.json"
LAST = 3


def run(args: dict, call: dict) -> dict:
    machines = json.loads(READINGS.read_text())["plants"][call["scope"]]
    name = args["machine"].strip().upper()
    if name not in machines:
        return {"error": f"there are no vibration readings for {name} at this plant"}
    machine = machines[name]
    daily = machine["daily"]
    last = daily[-LAST:]
    return {
        "machine": name,
        "point": machine["point"],
        "since": machine["since"],
        "readings": len(daily),
        "mean": round(statistics.fmean(daily), 2),
        "rms": round(math.sqrt(statistics.fmean(value * value for value in daily)), 2),
        "slope_per_day": round(statistics.linear_regression(range(len(daily)), daily).slope, 3),
        "alert": machine["alert"],
        "last": last,
        "above_alert": all(value > machine["alert"] for value in last),
    }
