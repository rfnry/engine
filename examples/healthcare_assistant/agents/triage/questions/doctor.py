import json
from pathlib import Path

CLINICS = Path(__file__).parents[3] / "data" / "clinics.json"


def run(input: dict, call: dict) -> dict:
    doctors = json.loads(CLINICS.read_text())["clinics"][call["scope"]]["doctors"]
    return {"options": {**doctors, "none": "No doctor is named"}}
