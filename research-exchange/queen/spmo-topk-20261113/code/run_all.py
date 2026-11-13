#!/usr/bin/env python3
"""Entry point: python3 code/run_all.py  -> writes out/*. Stdlib only. Inputs listed in SPEC.md section 5."""
import os, sys, csv, gzip, json, math, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spmo_topk import *
OUT = os.path.join(ROOT, 'out'); os.makedirs(OUT, exist_ok=True)
WIN = {'W2015_2025': ('2015-01-02', '2025-01-02', '2015'),
       'W2005_2015': ('2005-01-03', '2015-01-02', '2005'),
       'W2015_2026x': ('2015-01-02', '2026-11-12', '2015'),
       'W2025_2026x': ('2025-01-02', '2026-11-12', '2015')}
ERAS = {}
def era(e, tr=False):
    k = (e, tr)
    if k not in ERAS:
        ERAS[k] = Universe(e, tr)
    return ERAS[k]

TABLES = {}
def tables(e, excl):
    k = (e, excl)
    if k not in TABLES:
        U = era(e); rows = []
        lo, hi = ('2004-01-01', '2015-01-02') if e == '2005' else ('2014-01-01', '2026-11-12')
        for (y, m, ref, ex) in schedule(U, lo, hi):
            if ex < lo[:4] + '-03-01' or ex > hi: continue
            if e == '2005' and ex > '2015-03-31': continue
            if U.me(y - 1, m - 2) is None: continue
            rows.append(((y, m, ref, ex), score_table(U, y, m, frozenset([excl]) if excl else frozenset())))
        TABLES[k] = rows
    return TABLES[k]

summary = []; yearly_rows = []; nav_rows = []; event_rows = []; hold_rows = []
for e in ('2015', '2005'):
    for excl in ('', 'NVDA'):
        for (y, m, ref, ex), rows in tables(e, excl):
            for r in rows[:10]:
                hold_rows.append(dict(era=e, excl=excl or '-', reb='%d-%02d' % (y, m), ref_date=ref, exec_date=ex,
                    rank=r['rank'], ticker=r['ticker'], mom_value_12m_1=round(r['mv'], 6), sigma_daily=round(r['sigma'], 6),
                    ramv=round(r['ramv'], 4), z_raw=round(r['z_raw'], 4), z_wins=round(r['z'], 4), score=round(r['score'], 4),
                    universe_n=len(rows), flag=r['flag']))
# full score tables (no exclusion) for others to reproduce
with gzip.open(os.path.join(OUT, 'score_tables_full.csv.gz'), 'wt', newline='') as f:
    w = None
    for e in ('2015', '2005'):
        for (y, m, ref, ex), rows in tables(e, ''):
            for r in rows:
                d = dict(era=e, reb='%d-%02d' % (y, m), ref_date=ref, exec_date=ex, rank=r['rank'], ticker=r['ticker'],
                         form=r.get('form', ''), mv=r['mv'], sigma=r['sigma'], ramv=r['ramv'], z_raw=r['z_raw'], z=r['z'],
                         score=r['score'], flag=r['flag'])
                if w is None: w = csv.DictWriter(f, list(d)); w.writeheader()
                w.writerow(d)

def run(wname, K, stop, excl, tr):
    s, en, e = WIN[wname]
    tabs = [(ex, [r['ticker'] for r in rows[:K]]) for (y, m, ref, ex), rows in tables(e, excl)]
    U = era(e, tr)
    res = simulate(U, s, en, K, tabs, stop=stop, label=wname)
    st = stats(res['nav'])
    tot = math.log(st['end_value']) if st['end_value'] > 0 else float('nan')
    nv = res['contrib'].get('NVDA', 0.0)
    v = 'K%d|%s|%s|%s' % (K, 'stop30' if stop else 'nostop', 'exNVDA' if excl else 'all', 'TR' if tr else 'PX')
    summary.append(dict(window=wname, variant=v, K=K, stop='30% trail->cash' if stop else 'none', universe='ex-NVDA' if excl else 'all',
                        basis='total return' if tr else 'price only', start=res['start'], end=res['end'],
                        cagr_pct=round(100 * st['cagr'], 4), mdd_pct=round(100 * st['mdd'], 4), end_value=round(st['end_value'], 4),
                        nvda_logshare_pct=round(100 * nv / tot, 1) if tot and tot == tot and abs(tot) > 1e-9 else '',
                        n_stops=sum(1 for x in res['events'] if x['action'].startswith('STOP')),
                        n_cashouts=sum(1 for x in res['events'] if x['action'].startswith('series_end'))))
    for yr, r in yearly(res['nav']).items():
        yearly_rows.append(dict(window=wname, variant=v, year=yr, ret_pct=round(100 * r, 2)))
    for d, val, h in res['nav']:
        nav_rows.append((wname, v, d, round(val, 8), h))
    for x in res['events']:
        event_rows.append(dict(window=wname, variant=v, **x))
    return res

for wname in WIN:
    e = WIN[wname][2]
    for K in (1, 2, 3):
        for stop in (None, 0.30):
            for excl in ('', 'NVDA'):
                for tr in ((False, True) if e == '2015' or os.path.isdir(os.path.join(ROOT, 'data', 'fmp_divadj_2005')) else (False,)):
                    try:
                        run(wname, K, stop, excl, tr)
                    except Exception as ex:
                        summary.append(dict(window=wname, variant='K%d|%s|%s|%s' % (K, stop, excl, tr), cagr_pct='ERROR: %s' % ex))
# benchmarks
for wname, (s, en, e) in WIN.items():
    for tr in (False, True):
        try: U = era(e, tr)
        except Exception: continue
        for t in ('SPY', 'SPMO'):
            if t not in U.px or not U.px[t]: continue
            s2 = max(s, '2015-10-12') if t == 'SPMO' else s
            nav = bh(U, t, s2, en)
            if not nav or len(nav) < 50: continue
            st = stats(nav)
            summary.append(dict(window=wname, variant='%s buy&hold' % t, K='', stop='', universe='', basis='total return' if tr else 'price only',
                                start=nav[0][0], end=nav[-1][0], cagr_pct=round(100 * st['cagr'], 4), mdd_pct=round(100 * st['mdd'], 4),
                                end_value=round(st['end_value'], 4)))
            for yr, r in yearly(nav).items():
                yearly_rows.append(dict(window=wname, variant='%s %s' % (t, 'TR' if tr else 'PX'), year=yr, ret_pct=round(100 * r, 2)))

def wcsv(name, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(os.path.join(OUT, name), 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
wcsv('summary.csv', summary); wcsv('yearly_returns.csv', yearly_rows); wcsv('events.csv', event_rows)
wcsv('holdings_top10_by_rebalance.csv', hold_rows)
with gzip.open(os.path.join(OUT, 'nav_daily.csv.gz'), 'wt', newline='') as f:
    w = csv.writer(f); w.writerow(['window', 'variant', 'date', 'nav', 'holdings']); w.writerows(nav_rows)
print('done', len(summary))
