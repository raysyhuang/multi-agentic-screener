"""Builds the markdown tables used in REPORT.md from out/*.csv (so numbers are not hand-typed)."""
import csv, collections, os
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = list(csv.DictReader(open(os.path.join(R, 'out', 'summary.csv'))))
def g(w, v):
    for r in S:
        if r['window'] == w and r['variant'] == v: return r
def bm(w, t, basis):
    for r in S:
        if r['window'] == w and r['variant'] == t + ' buy&hold' and r['basis'] == basis: return r
L = []
for w, title in (('W2015_2025', 'Main 2015-01-02 → 2025-01-02'), ('W2005_2015', 'Academic main 2005-01-03 → 2015-01-02 (provisional membership & data)'),
                 ('W2015_2026x', 'Extension 2015-01-02 → 2026-11-12'), ('W2025_2026x', 'Out-of-window 2025-01-02 → 2026-11-12')):
    L.append('\n### ' + title + '\n')
    L.append('| Variant | CAGR % | Max DD % | $1 → | NVDA share of log growth % | stops |')
    L.append('|---|---:|---:|---:|---:|---:|')
    for K in (1, 2, 3):
        for v, lab in (('K%d|nostop|all|PX' % K, 'Top%d price, no stop' % K), ('K%d|stop30|all|PX' % K, 'Top%d price, 30%% stop→cash' % K),
                       ('K%d|nostop|exNVDA|PX' % K, 'Top%d price, no NVDA' % K), ('K%d|stop30|exNVDA|PX' % K, 'Top%d price, no NVDA + stop' % K),
                       ('K%d|nostop|all|TR' % K, 'Top%d total return, no stop' % K)):
            r = g(w, v)
            if r: L.append('| %s | %s | %s | %s | %s | %s |' % (lab, r['cagr_pct'], r['mdd_pct'], r['end_value'], r['nvda_logshare_pct'], r['n_stops']))
    for t in ('SPY', 'SPMO'):
        for basis in ('price only', 'total return'):
            r = bm(w, t, basis)
            if r: L.append('| %s %s (%s→) | %s | %s | %s | | |' % (t, basis, r['start'], r['cagr_pct'], r['mdd_pct'], r['end_value']))
Y = collections.defaultdict(dict)
for r in csv.DictReader(open(os.path.join(R, 'out', 'yearly_returns.csv'))):
    Y[(r['window'], r['variant'])][r['year']] = r['ret_pct']
for w, cols in (('W2015_2026x', ['K1|nostop|all|PX', 'K2|nostop|all|PX', 'K3|nostop|all|PX', 'K1|stop30|all|PX', 'K1|nostop|exNVDA|PX', 'SPY PX', 'SPMO PX']),
                ('W2005_2015', ['K1|nostop|all|PX', 'K2|nostop|all|PX', 'K3|nostop|all|PX', 'K1|stop30|all|PX', 'SPY PX'])):
    L.append('\n### Calendar-year returns (%%), %s (price only; first/last year partial)\n' % w)
    L.append('| Year | ' + ' | '.join(c.replace('|all', '').replace('|PX', '').replace('|nostop', '').replace('|', '+') for c in cols) + ' |')
    L.append('|---|' + '---:|' * len(cols))
    yrs = sorted({y for c in cols for y in Y[(w, c)]})
    for y in yrs: L.append('| %s | ' % y + ' | '.join(Y[(w, c)].get(y, '') for c in cols) + ' |')
open(os.path.join(R, 'out', 'report_tables.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
