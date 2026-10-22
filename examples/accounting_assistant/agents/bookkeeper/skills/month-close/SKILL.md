---
name: month-close
description: Close a month of the ledger with the firm's script — totals, expenses by account, possible duplicates
---
The firm closes a month with close.py, beside this file, so every client's close is computed the
same way. Run it as it is, with the month as YYYY-MM and no cd before it:

python3 /agent/skills/month-close/close.py 2026-09

Every command starts in the workspace, next to ledger.csv, and the script works there: it reads
ledger.csv, writes the expenses by account to close-2026-09.csv beside it, and prints the money in,
the money out, the net, the expenses by account and the possible duplicates by line number, as
rules/ledger.md defines them. Do not compute these yourself; the script's numbers are the close.

Then read each possible duplicate yourself: Load ledger.csv with where "workspace", look at both
lines, and say whether it reads like one charge taken twice or two real charges, and what the
accountant should check. The totals count every row, a duplicate included; say so when there is
one.

Reply with the net, the expenses by account, the duplicates and your read on each, and the file
the script wrote.
