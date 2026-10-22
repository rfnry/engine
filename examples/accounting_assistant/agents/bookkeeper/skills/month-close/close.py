import csv
import os
import sys
from collections import defaultdict
from decimal import Decimal

if len(sys.argv) != 2:
    sys.exit("close: usage is close.py YYYY-MM")
month = sys.argv[1]
if not os.path.exists("ledger.csv"):
    sys.exit("close: no ledger.csv here; run close.py where the command starts, without cd")

with open("ledger.csv", newline="") as file:
    rows = [
        (line, row)
        for line, row in enumerate(csv.DictReader(file), start=2)
        if row["date"].startswith(f"{month}-")
    ]
if not rows:
    sys.exit(f"close: ledger.csv has no rows in {month}")

money_in, money_out = Decimal(0), Decimal(0)
expenses: dict[str, Decimal] = defaultdict(Decimal)
entries: dict[tuple[str, str, Decimal], list[int]] = defaultdict(list)
for line, row in rows:
    amount = Decimal(row["amount"])
    if amount < 0:
        money_out -= amount
        expenses[row["account"]] -= amount
    else:
        money_in += amount
    entries[(row["counterparty"], row["reference"], amount)].append(line)

written = f"close-{month}.csv"
with open(written, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["account", "expenses"])
    for account in sorted(expenses):
        writer.writerow([account, f"{expenses[account]:.2f}"])

print(f"month {month}: {len(rows)} rows")
print(f"money in {money_in:.2f}, money out {money_out:.2f}, net {money_in - money_out:.2f}")
print(f"expenses by account, written to {written}:")
for account in sorted(expenses):
    print(f"  {account} {expenses[account]:.2f}")
duplicates = [(key, lines) for key, lines in entries.items() if len(lines) > 1]
print("possible duplicates:" if duplicates else "possible duplicates: none")
for (counterparty, reference, amount), lines in duplicates:
    print(f"  {counterparty} {reference} {amount:.2f} on lines {', '.join(map(str, lines))}")
