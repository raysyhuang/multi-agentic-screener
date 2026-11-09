# Neo arithmetic reproduction of the committed annual Massive ledger.
# This script does not select names; it reprices the committed Queen list.
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "gpt/annual-ledger-20261109/gpt_annual_ledger_massive_20261109.csv"
OUT = Path(__file__).with_name("annual_top1_neo_massive.csv")
COST = 0.0025

rows = list(csv.DictReader(SRC.open()))
equity = 1.0
prior = None
out = []
for i, r in enumerate(rows):
    ticker = r["ticker"]
    nxt = rows[i + 1]["ticker"] if i + 1 < len(rows) else None
    buy_fee = 0.0 if prior == ticker else COST * equity
    after_buy = equity - buy_fee
    exit_value = after_buy * float(r["close_exit"]) / float(r["close_entry"])
    sell_fee = 0.0 if nxt == ticker else COST * exit_value
    ending = exit_value - sell_fee
    out.append({
        "year": r["year"], "ticker": ticker,
        "entry_date": r["entry_date"], "exit_date": r["exit_date"],
        "entry_close": r["close_entry"], "exit_close": r["close_exit"],
        "buy_fee": f"{buy_fee:.10f}", "sell_fee": f"{sell_fee:.10f}",
        "net_return": f"{ending/equity-1:.10f}", "equity": f"{ending:.10f}",
        "selection_status": "QUEEN_PICK_REPRICED_NOT_INDEPENDENTLY_SELECTED",
    })
    equity = ending
    prior = ticker

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=out[0].keys())
    w.writeheader(); w.writerows(out)
print(f"final_multiple={equity:.10f}")
print(f"cagr_10y={equity**0.1-1:.10f}")
