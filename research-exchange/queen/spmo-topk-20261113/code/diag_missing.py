"""List names whose sigma is not from a verified daily series and that rank in the top 6 of any table (both eras,
with and without NVDA), plus top-3 picks that lack a daily series through their holding window."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spmo_topk import *
need = {}
for era, lo, hi in (('2005', 2004, 2014), ('2015', 2014, 2026)):
    U = Universe(era)
    sched = [(y, m) for y in range(lo, hi + 1) for m in (3, 9) if not (era == '2015' and (y, m) < (2014, 9)) and not (era == '2005' and (y, m) < (2004, 9))]
    for i, (y, m) in enumerate(sched):
        for ex in ('', 'NVDA'):
            rows = score_table(U, y, m, frozenset([ex]) if ex else frozenset())
            ref, exd = rebalance_dates(U, y, m)
            nxt = rebalance_dates(U, *sched[i + 1])[1] if i + 1 < len(sched) else U.SES[-1]
            for r in rows[:6]:
                bad = 'sigma=' in r['flag']
                held = r['rank'] <= 3 and (exd not in U.px[r['ticker']] or U.dates[r['ticker']][-1] < min(nxt, U.SES[-1]))
                if bad or held:
                    need.setdefault((era, r['ticker']), []).append('%d-%02d r%d%s%s %s' % (y, m, r['rank'], ' sigma' if bad else '', ' hold' if held else '', ex))
for k, v in sorted(need.items()): print(k, v)
