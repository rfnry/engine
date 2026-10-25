import json
from pathlib import Path

ORDERS = json.loads((Path(__file__).parents[3] / "data" / "orders.json").read_text())["orders"]


def run(args: dict, call: dict) -> dict:
    order_id = args["order_id"].strip().upper()
    return ORDERS.get(order_id, {"error": f"there is no order {order_id}"})
