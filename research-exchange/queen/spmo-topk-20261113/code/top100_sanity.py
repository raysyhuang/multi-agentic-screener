#!/usr/bin/env python3
"""Sanity check: rebuild an approximation of the S&P 500 Momentum Index (first quintile ~100 names,
weight = market cap x momentum score, cap = min(9%, 3 x S&P 500 cap weight)) and compare to SPMO.
Approximations: full (not float-adjusted) market cap from FMP historical-market-capitalization; no buffer rule;
weights set with reference-date prices, held as fixed shares from the effective-date close; no fees.
Run: python3 code/top100_sanity.py  -> out/top100_sanity_*.csv"""
import os, sys, json, csv, math, collections, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spmo_topk import *
OUT = os.path.join(ROOT, 'out'); os.makedirs(OUT, exist_ok=True)
U = Universe('2015'); Utr = Universe('2015', tr=True)
MC = {}
def mcap(t, d):
    if t not in MC:
        try:
            j = json.load(open(os.path.join(ROOT, 'data', 'fmp_mcap_2014_2026', SYMMAP15.get(t, t) + '.json')))
            MC[t] = {r['date']: float(r['marketCap']) for r in j if r.get('marketCap')} if isinstance(j, list) else {}
        except Exception:
            MC[t] = {}
    m = MC[t]
    i = bisect_left_(U.SES, d)
    for k in range(0, 11):
        if i - k >= 0 and U.SES[i - k] in m: return m[U.SES[i - k]]
    return None
def bisect_left_(a, x):
    import bisect; return bisect.bisect_left(a, x)
def cap_weights(raw, capv):
    w = {t: raw[t] / sum(raw.values()) for t in raw}
    for _ in range(200):
        over = {t for t in w if w[t] > capv[t] + 1e-12}
        if not over: break
        excess = sum(w[t] - capv[t] for t in over)
        for t in over: w[t] = capv[t]
        free = [t for t in w if t not in over and w[t] < capv[t]]
        tot = sum(w[t] for t in free)
        for t in free: w[t] += excess * w[t] / tot
    return w
rebs = []
for y in range(2015, 2025):
    for m in (3, 9):
        if (y, m) < (2015, 9): continue
        rows = score_table(U, y, m)
        ref, ex = rows[0]['ref'], rows[0]['exec']
        N = int(round(len(rows) / 5.0))
        allm = {r['ticker']: mcap(r['ticker'], ref) for r in rows}
        tot = sum(v for v in allm.values() if v)
        sel = [r for r in rows if allm.get(r['ticker'])][:N]
        miss = [r['ticker'] for r in rows[:N] if not allm.get(r['ticker'])]
        raw = {r['ticker']: allm[r['ticker']] * r['score'] for r in sel}
        capv = {t: min(0.09, 3 * allm[t] / tot) for t in raw}
        w = cap_weights(raw, capv)
        rebs.append(dict(y=y, m=m, ref=ref, exec=ex, N=N, w=w, missing_mcap=miss))
def run(Ux, start='2015-10-12', end='2025-01-02'):
    S = [d for d in Ux.SES if '2015-09-18' <= d <= end]
    nav = []; sh = {}; cash = 0.0; last = {}; V = 1.0
    rb = {r['exec']: r for r in rebs}
    for d in S:
        for t in list(sh):
            p = Ux.px[t].get(d)
            if p is None and Ux.dates[t][-1] < d:
                cash += sh[t] * last[t]; del sh[t]; continue
            if p is not None: last[t] = p
        V = cash + sum(sh[t] * last[t] for t in sh) if sh else V
        if d in rb:
            r = rb[d]
            # weights fixed at reference-date prices, then drifted to the effective date
            dr = {}
            for t, wt in r['w'].items():
                pr = Ux.price_on_or_before(t, r['ref'])[1]; pe = Ux.px[t].get(d) or Ux.price_on_or_before(t, d)[1]
                if pr and pe: dr[t] = wt * pe / pr
            s = sum(dr.values())
            sh = {}; cash = 0.0
            for t, x in dr.items():
                pe = Ux.px[t].get(d) or Ux.price_on_or_before(t, d)[1]
                sh[t] = V * x / s / pe; last[t] = pe
        nav.append((d, V))
    nav = [(d, v) for d, v in nav if d >= start]
    return [(d, v / nav[0][1]) for d, v in nav]
res = {}
for lab, Ux in (('PX', U), ('TR', Utr)):
    a = run(Ux); b = bh(Ux, 'SPMO', '2015-10-12', '2025-01-02'); c = bh(Ux, 'SPY', '2015-10-12', '2025-01-02')
    A = dict(a); B = dict(b)
    ds = [d for d, _ in a if d in B]
    ra = [A[ds[i]] / A[ds[i - 1]] - 1 for i in range(1, len(ds))]
    rb_ = [B[ds[i]] / B[ds[i - 1]] - 1 for i in range(1, len(ds))]
    te = statistics.pstdev([x - y for x, y in zip(ra, rb_)]) * math.sqrt(252)
    corr = statistics.correlation(ra, rb_)
    sa, sb, sc = stats(a), stats(b), stats(c)
    res[lab] = dict(basis=lab, approx_cagr_pct=round(100 * sa['cagr'], 3), spmo_cagr_pct=round(100 * sb['cagr'], 3),
                    tracking_diff_pp_per_yr=round(100 * (sa['cagr'] - sb['cagr']), 3), tracking_error_ann_pct=round(100 * te, 3),
                    daily_corr=round(corr, 4), approx_mdd_pct=round(100 * sa['mdd'], 2), spmo_mdd_pct=round(100 * sb['mdd'], 2),
                    spy_cagr_pct=round(100 * sc['cagr'], 3), start=a[0][0], end=a[-1][0])
    ya, yb = yearly(a), yearly(b)
    with open(os.path.join(OUT, 'top100_sanity_yearly_%s.csv' % lab), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['year', 'approx_pct', 'spmo_pct', 'diff_pp'])
        for yv in ya: w.writerow([yv, round(100 * ya[yv], 2), round(100 * yb.get(yv, float('nan')), 2), round(100 * (ya[yv] - yb.get(yv, float('nan'))), 2)])
with open(os.path.join(OUT, 'top100_sanity_summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, list(res['PX'])); w.writeheader(); w.writerows(res.values())
with open(os.path.join(OUT, 'top100_sanity_rebalances.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['reb', 'ref', 'exec', 'N', 'n_held', 'top10_by_weight', 'missing_mcap_in_topN'])
    for r in rebs:
        top = sorted(r['w'].items(), key=lambda x: -x[1])[:10]
        w.writerow(['%d-%02d' % (r['y'], r['m']), r['ref'], r['exec'], r['N'], len(r['w']), ' '.join('%s:%.1f%%' % (t, 100 * v) for t, v in top), ' '.join(r['missing_mcap'])])
for v in res.values(): print(v)
