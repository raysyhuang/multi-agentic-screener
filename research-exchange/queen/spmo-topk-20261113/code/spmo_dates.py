"""TopK (no stop) on SPMO's own dates 2015-10-12 -> 2025-01-02, price and TR, vs SPMO/SPY. -> out/same_dates_as_spmo.csv"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spmo_topk import *
U = Universe('2015'); Ut = Universe('2015', tr=True)
tabs = []
for y in range(2014, 2026):
    for m in (3, 9):
        if (y, m) < (2014, 9) or (y, m) > (2025, 3): continue
        rows = score_table(U, y, m); tabs.append((rows[0]['exec'], [r['ticker'] for r in rows[:3]]))
out = []
for lab, Ux in (('price only', U), ('total return', Ut)):
    for K in (1, 2, 3):
        st = stats(simulate(Ux, '2015-10-12', '2025-01-02', K, [(e, t[:K]) for e, t in tabs])['nav'])
        out.append(dict(series='SPMO-rule Top%d' % K, basis=lab, cagr_pct=round(100 * st['cagr'], 3), mdd_pct=round(100 * st['mdd'], 2), end_value=round(st['end_value'], 4)))
    for t in ('SPMO', 'SPY'):
        st = stats(bh(Ux, t, '2015-10-12', '2025-01-02'))
        out.append(dict(series=t, basis=lab, cagr_pct=round(100 * st['cagr'], 3), mdd_pct=round(100 * st['mdd'], 2), end_value=round(st['end_value'], 4)))
with open(os.path.join(ROOT, 'out', 'same_dates_as_spmo.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, list(out[0])); w.writeheader(); w.writerows(out)
for o in out: print(o)
