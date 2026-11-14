#!/usr/bin/env python3
"""Current (as-of 2026-11-13) holdings of Queen's verified NDX rules.

Imports /workspace/ndx100-fill-20261114-v1.1 and
/workspace/ndx-riskadj-mom-check-20261114 READ-ONLY.
Run with system python3. Writes only under this folder.
"""
import sys, os, csv, json, math, datetime as dt, bisect
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)
ENG = '/workspace/ndx100-fill-20261114-v1.1'
RADJ = '/workspace/ndx-riskadj-mom-check-20261114'

sys.path.insert(0, ENG)
import lib as L
import engine as E
sys.path.insert(0, RADJ)
import riskadj as RA

SIGNAL = '2026-10-30'   # October 2026 month-end (Oct 31 is Saturday)
PRIOR = '2026-09-30'
LATEST = '2026-11-13'
ASOF = LATEST

# ---------- 1. Post rule rankings (mean of last 6 monthly returns) ----------
def dump_ranking(label, date, kind, rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def mean6_rows(date, topn=10):
    i = E.IDX[date]
    rk = E.ranking(i, W=6, kind='mean')
    n_mem = len(E.MEM[date]); n_ranked = len(rk)
    rows = []
    for n, (s, t, key) in enumerate(rk[:topn], 1):
        rows.append(dict(
            rule='post_mean6', signal=date, rank=n, ticker=t, series_key=key,
            score=s, score_4dp=round(s, 4), in_top3='Y' if n <= 3 else 'N',
            n_members=n_mem, n_ranked=n_ranked,
        ))
    return rows, n_mem, n_ranked

m6_cur, nmem_c, nrank_c = mean6_rows(SIGNAL, 10)
m6_pri, nmem_p, nrank_p = mean6_rows(PRIOR, 10)
# prior: keep full top10 in file, highlight top3 in report
dump_ranking('cur', SIGNAL, 'mean6', m6_cur, os.path.join(OUT, 'post_rule_top10_2026-10-30.csv'))
dump_ranking('pri', PRIOR, 'mean6', m6_pri, os.path.join(OUT, 'post_rule_top10_2026-09-30.csv'))

# ---------- 2. 12-1 risk-adjusted N=1 and pure 12-1 N=0 ----------
def ra_rows(date, L_back, skip, N, topn=10):
    i = E.IDX[date]
    rk = RA.ranking(i, ('ra', L_back, skip, N))
    rows = []
    for n, (score, t, key, ret, vol) in enumerate(rk[:topn], 1):
        rows.append(dict(
            rule=f'12-1_N{N}', signal=date, rank=n, ticker=t, series_key=key,
            score=score, score_4dp=round(score, 4),
            ret_12_1=ret, vol_ann=('' if vol is None else vol),
            in_top3='Y' if n <= 3 else 'N',
            n_ranked=len(rk), n_members=len(E.MEM[date]),
        ))
    return rows

ra_n1 = ra_rows(SIGNAL, 12, 1, 1, 10)
ra_n0 = ra_rows(SIGNAL, 12, 1, 0, 10)
dump_ranking('n1', SIGNAL, 'N1', ra_n1, os.path.join(OUT, 'riskadj_N1_12-1_top10_2026-10-30.csv'))
dump_ranking('n0', SIGNAL, 'N0', ra_n0, os.path.join(OUT, 'riskadj_N0_12-1_top10_2026-10-30.csv'))

# also prior-month RA top3 for context
ra_n1_p = ra_rows(PRIOR, 12, 1, 1, 3)
ra_n0_p = ra_rows(PRIOR, 12, 1, 0, 3)

# ---------- 3. Price stats for current post-rule Top3 ----------
# Load highs from adj files (same stamp convention as lib.load_daily).
def load_ohlc(sym):
    path = os.path.join(ENG, 'raw/adj', f'{sym}.csv')
    out = {}
    for r in csv.DictReader(open(path)):
        if not r.get('t'):
            continue
        d = L.d_of(r['t']).isoformat()
        out[d] = dict(o=float(r['o']), h=float(r['h']), l=float(r['l']), c=float(r['c']))
    return out

def trading_days_on_or_before(ks, d, n):
    i = bisect.bisect_right(ks, d)
    return ks[max(0, i-n):i]

def calendar_window(ks, d, days=365):
    end = dt.date.fromisoformat(d)
    start = (end - dt.timedelta(days=days)).isoformat()
    return [x for x in ks if start < x <= d]  # exclusive start, inclusive end; 365d lookback

TOP3 = [r['ticker'] for r in m6_cur[:3]]
ohlc = {t: load_ohlc(t) for t in TOP3}

price_rows = []
for t in TOP3:
    key = t  # none of MRNA/CRWD/MU is an alias at this date
    px0 = E.PX.at(key, SIGNAL)
    px1 = E.PX.at(key, LATEST)
    assert px0 and px1, (t, px0, px1)
    # stay in current segment
    seg = px1[2]
    ks_all = [d for d in E.PX.k[key] if E.PX.seg[key][d] == seg]
    c0, d0 = px0[0], px0[1]
    c1, d1 = px1[0], px1[1]
    ret = c1 / c0 - 1
    last50 = trading_days_on_or_before(ks_all, LATEST, 50)
    last200 = trading_days_on_or_before(ks_all, LATEST, 200)
    ma50 = sum(E.PX.D[key][d] for d in last50) / len(last50)
    ma200 = sum(E.PX.D[key][d] for d in last200) / len(last200)
    # 52-week high: max daily HIGH over last 365 calendar days in this segment
    win = calendar_window(ks_all, LATEST, 365)
    # highs from adj file; fall back to close if a day is missing high
    hi_pairs = []
    for d in win:
        if d in ohlc[t]:
            hi_pairs.append((ohlc[t][d]['h'], d))
        else:
            hi_pairs.append((E.PX.D[key][d], d))
    hi52, hi52_date = max(hi_pairs)
    # also 252-trading-day high of highs
    last252 = trading_days_on_or_before(ks_all, LATEST, 252)
    hi252_pairs = []
    for d in last252:
        if d in ohlc[t]:
            hi252_pairs.append((ohlc[t][d]['h'], d))
        else:
            hi252_pairs.append((E.PX.D[key][d], d))
    hi252, hi252_date = max(hi252_pairs)
    # consistency: adj close vs engine close on signal/latest
    adj_c0 = ohlc[t].get(d0, {}).get('c')
    adj_c1 = ohlc[t].get(d1, {}).get('c')
    price_rows.append(dict(
        ticker=t,
        signal_date=SIGNAL, signal_close_used=d0, signal_close=c0,
        latest_date=LATEST, latest_close_used=d1, latest_close=c1,
        ret_signal_to_latest=ret,
        ma50=ma50, ma50_n=len(last50), ma50_from=last50[0], ma50_to=last50[-1],
        above_ma50='Y' if c1 > ma50 else 'N',
        ma200=ma200, ma200_n=len(last200), ma200_from=last200[0], ma200_to=last200[-1],
        above_ma200='Y' if c1 > ma200 else 'N',
        hi52_calendar365=hi52, hi52_date=hi52_date,
        pct_below_hi52=c1 / hi52 - 1,
        hi252_trading=hi252, hi252_date=hi252_date,
        pct_below_hi252=c1 / hi252 - 1,
        adjfile_close_signal=adj_c0, adjfile_close_latest=adj_c1,
        segment=seg, segment_start=ks_all[0],
    ))

with open(os.path.join(OUT, 'top3_price_stats_2026-11-13.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(price_rows[0].keys()))
    w.writeheader(); w.writerows(price_rows)

# equal-weight basket (price only, no cost) from signal close to latest close
basket = sum(r['ret_signal_to_latest'] for r in price_rows) / 3
# QQQ price return same window
qqq_p, qqq_tr = E.qqq()
qqq_ret = qqq_p[LATEST] / qqq_p[SIGNAL] - 1
qqq_tr_ret = qqq_tr[LATEST] / qqq_tr[SIGNAL] - 1

# membership change for MRNA
from nasdaq_100_ticker_history import tickers_as_of
memb_changes = []
days = [d for d in E.DAYS if '2026-09-01' <= d <= LATEST]
prev = None
for d in days:
    y, m, dd = map(int, d.split('-'))
    mem = set(tickers_as_of(y, m, dd))
    if prev is not None:
        added, removed = mem - prev, prev - mem
        if added or removed:
            memb_changes.append(dict(date=d, added=' '.join(sorted(added)), removed=' '.join(sorted(removed))))
    prev = mem

with open(os.path.join(OUT, 'ndx_membership_changes_2026-09_to_11.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['date', 'added', 'removed'])
    w.writeheader(); w.writerows(memb_changes)

# calendar facts
cal = dict(
    signal=SIGNAL,
    signal_weekday=dt.date.fromisoformat(SIGNAL).strftime('%A'),
    october_31_2026=dt.date(2026, 10, 31).strftime('%A'),
    latest=LATEST,
    latest_weekday=dt.date.fromisoformat(LATEST).strftime('%A'),
    last_engine_day=E.DAYS[-1],
    last_me=E.ME[-1],
    n_members_signal=nmem_c,
    n_ranked_mean6_signal=nrank_c,
    n_members_prior=nmem_p,
    n_ranked_mean6_prior=nrank_p,
    basket_ew_ret_signal_to_latest=basket,
    qqq_price_ret_signal_to_latest=qqq_ret,
    qqq_total_ret_signal_to_latest=qqq_tr_ret,
    qqq_price_latest=qqq_p[LATEST],
    qqq_price_signal=qqq_p[SIGNAL],
)

# next trading day after signal (risk-adj next-open convention)
p = E.DAYS.index(SIGNAL)
next_td = E.DAYS[p+1] if p+1 < len(E.DAYS) else None
cal['next_session_after_signal'] = next_td

with open(os.path.join(OUT, 'calendar_and_basket.json'), 'w') as f:
    json.dump(cal, f, indent=2)

# ---------- print compact tables ----------
def fmt_pct(x, dp=2):
    return f'{x*100:+.{dp}f}%'

print('=== POST RULE mean6  Top10 at', SIGNAL, f'(n_members={nmem_c}, n_ranked={nrank_c}) ===')
for r in m6_cur:
    mark = '  TOP3' if r['in_top3']=='Y' else ''
    print(f"  {r['rank']:2d} {r['ticker']:<5}  score {r['score']:.6f}  ({r['score_4dp']:.4f}){mark}")
print('=== POST RULE mean6  Top3 at', PRIOR, f'(n_members={nmem_p}, n_ranked={nrank_p}) ===')
for r in m6_pri[:3]:
    print(f"  {r['rank']:2d} {r['ticker']:<5}  score {r['score']:.6f}  ({r['score_4dp']:.4f})")
print('=== 12-1 N=1 Top3 at', SIGNAL, '===')
for r in ra_n1[:3]:
    print(f"  {r['rank']:2d} {r['ticker']:<5}  score {r['score']:.6f}  ret {r['ret_12_1']:.4f}  vol {r['vol_ann']:.4f}")
print('=== 12-1 N=0 Top3 at', SIGNAL, '===')
for r in ra_n0[:3]:
    print(f"  {r['rank']:2d} {r['ticker']:<5}  score {r['score']:.6f}  ret {r['ret_12_1']:.4f}")
print('=== PRICE STATS post Top3', SIGNAL, '->', LATEST, '===')
for r in price_rows:
    print(r['ticker'],
          'sig', r['signal_close'], 'last', r['latest_close'],
          'ret', fmt_pct(r['ret_signal_to_latest']),
          'MA50', round(r['ma50'], 4), 'n', r['ma50_n'], 'above', r['above_ma50'],
          'MA200', round(r['ma200'], 4), 'n', r['ma200_n'], 'above', r['above_ma200'],
          'hi52', r['hi52_calendar365'], 'on', r['hi52_date'],
          'vs_hi52', fmt_pct(r['pct_below_hi52']),
          'hi252', r['hi252_trading'], 'on', r['hi252_date'])
print('EW basket', fmt_pct(basket), 'QQQ price', fmt_pct(qqq_ret), 'QQQ TR', fmt_pct(qqq_tr_ret))
print('membership changes', memb_changes)
print('next session after signal', next_td)
print('calendar', cal)

# agreement vs existing live_track / riskadj picks
print('\n=== vs existing live_track Top3 ===')
print('expected 2026-10-30 MRNA CRWD MU 0.3963 0.1644 0.1476')
print('got     ', ' '.join(r['ticker'] for r in m6_cur[:3]),
      ' '.join(f"{r['score_4dp']:.4f}" for r in m6_cur[:3]))
print('expected 2026-09-30 ALAB MU INTC 0.2906 0.2643 0.2590')
print('got     ', ' '.join(r['ticker'] for r in m6_pri[:3]),
      ' '.join(f"{r['score_4dp']:.4f}" for r in m6_pri[:3]))
print('expected N1 SNDK MU LITE 6.6285 4.4485 3.7819')
print('got     ', ' '.join(r['ticker'] for r in ra_n1[:3]),
      ' '.join(f"{r['score_4dp']:.4f}" for r in ra_n1[:3]))
print('expected N0 SNDK MRNA LITE 7.7287 6.0902 3.8187')
print('got     ', ' '.join(r['ticker'] for r in ra_n0[:3]),
      ' '.join(f"{r['score_4dp']:.4f}" for r in ra_n0[:3]))
