#!/usr/bin/env python3
"""SPMO-rule TopK backtest (Queen research, 2026-11-13 Asia/Taipei). Research only.

Selection follows the S&P 500 Momentum Index method (S&P DJI, "S&P Momentum Indices Methodology", Nov 2026):
  reference date = last business day of Feb / Aug; effective after close of 3rd Friday of Mar / Sep;
  momentum value MV = P(M-2)/P(M-14) - 1 (M = effective month; prices = last session of those months;
  if no price that day, the price from the prior day, up to 10 days back; else 9-month fallback P(M-2)/P(M-11)-1);
  risk-adjusted RAMV = MV / sigma(daily price returns over the same period);
  z-score across the eligible universe, winsorized at +-3; score = 1+z (z>0) or 1/(1-z) (z<0).
  Eligibility: S&P 500 member on the reference date, traded >= 10 months before the reference date.
Portfolio (Queen choices): Top K (1/2/3) by score (ties at the +3 cap broken by unwinsorized RAMV, then ticker),
  equal weight, trade at close of the effective date, 25 bps per side on traded value, liquidate at END.
Optional: excl={'NVDA'} (next name taken); stop=0.30 (post-entry-peak trailing stop -> cash until next rebalance).
Entry point: python3 code/run_all.py  (see README in SPEC.md)
"""
import json, csv, gzip, os, math, statistics, collections, datetime as dt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P15 = '/workspace/momentum432-inputs-pack-full-20261109'          # frozen 2026-11-09 pack (read-only)
P05 = '/workspace/w10h2-w12h4-oos-2005-2015-20261112/pack'         # provisional 2005-2015 pack (read-only)
FMP05 = '/workspace/w10h2-w12h4-oos-2005-2015-20261112/raw/fmp'     # FMP daily 2003-10..2016-01 (that packet's pull)
C = 0.0025
# renamed-but-continuing companies whose FMP history sits under the current symbol (validated against pack rank prices)
SYMMAP15 = {'FB': 'META', 'ANTM': 'ELV', 'BRK.B': 'BRK-B', 'BF.B': 'BF-B', 'LB': 'BBWI', 'ARNC': 'HWM', 'HRS': 'LHX', 'TMK': 'GL', 'BHGE': 'BKR'}

def rd(f):
    return list(csv.DictReader(gzip.open(f, 'rt') if f.endswith('.gz') else open(f)))

def load_fmp(path, field='close'):
    try:
        j = json.load(open(path))
    except Exception:
        return {}
    if not isinstance(j, list):
        return {}
    out = {}
    for r in j:
        v = r.get(field)
        if v is None and field == 'close':
            v = r.get('adjClose')
        if v is not None and v > 0:
            out[r['date']] = float(v)
    return out

class Universe:
    """membership + daily prices for one era ('2015' -> frozen pack membership, FMP 2013-2026;
    '2005' -> provisional 2005 pack membership, FMP 2003-2016 with the packet's alias routing)."""
    def __init__(s, era, tr=False):
        s.era = era; s.tr = tr
        pack = P15 if era == '2015' else P05
        s.iv = collections.defaultdict(list)
        for r in rd(os.path.join(pack, 'membership_intervals.csv')):
            s.iv[r['engine_ticker']].append((r['start_date'], r['end_date_exclusive'] or '9999-12-31'))
        s.CA = collections.defaultdict(list)
        for r in rd(os.path.join(pack, 'corporate_actions_applied.csv')):
            if r['kind'] in ('spin', 'special_div') and r['factor_F']:
                s.CA[r['engine_ticker']].append((r['ex_date'], float(r['factor_F'])))
        s.RP = {(r['engine_ticker'], r['month_end']): float(r['rank_price'])
                for r in rd(os.path.join(pack, 'rank_prices_month_end.csv.gz'))}
        s.px = {}; s.src = {}
        s.snap = None
        if era == '2015':
            # after the frozen pack's last signal (2025-11-28) membership = public reconstruction snapshot (same source file)
            hs = rd('/workspace/prior-quarter-winner-check-2026-10-20/sp500_hist_upd.csv')
            s.snap = [(r['date'], set(r['tickers'].split(','))) for r in hs if r['date'] >= '2025-06-01']
            for _, ts in s.snap:
                for t in ts:
                    if t not in s.iv: s.iv[t] = []
            d = os.path.join(ROOT, 'data', 'fmp_divadj_2013_2026' if tr else 'fmp_px_2013_2026')
            fld = 'adjClose' if tr else 'close'
            old05 = None
            for t in list(s.iv) + ['SPY', 'SPMO']:
                s.px[t] = load_fmp(os.path.join(d, SYMMAP15.get(t, t) + '.json'), fld)
                s.src[t] = 'fmp:' + SYMMAP15.get(t, t)
                if not any(k < '2016-01-29' for k in s.px[t]) and t in s.iv:
                    # whole pre-2016 segment from the 2005-2015 packet's FMP pull (alias-routed), no splicing
                    if old05 is None: old05 = Universe('2005', tr)
                    seg = {k: v for k, v in old05.px.get(t, {}).items() if k < '2016-01-29'}
                    if seg:
                        if s.px[t] and min(s.px[t]) <= '2016-02-29':
                            pass  # would splice two vendors' series; keep 2013-2026 pull only
                        else:
                            s.px[t].update(seg); s.src[t] += '+fmp05_pre2016'
        else:
            alias = collections.defaultdict(list)
            for r in rd(os.path.join(P05, 'alias_rules.csv')):
                alias[r['label']].append((r['symbol'], r['me_from'], r['me_to']))
            cache = {}
            def sym(x):
                if x not in cache:
                    a = load_fmp(os.path.join(FMP05, x + '.json'))
                    a.update(load_fmp(os.path.join(ROOT, 'data', 'fmp_px_2003_gap', x + '.json')))
                    if tr:
                        b = load_fmp(os.path.join(ROOT, 'data', 'fmp_divadj_2005', x + '.json'), 'adjClose')
                        a = b if b else {}
                    cache[x] = a
                return cache[x]
            for t in list(s.iv) + ['SPY']:
                base = dict(sym(t))
                for (sy, f, to) in alias.get(t, []):
                    other = sym(sy)
                    # route dates in [first day of month(f), to] to the alias symbol
                    lo = f[:8] + '01' if len(f) == 10 else f
                    for k in [k for k in base if lo <= k <= to]:
                        del base[k]
                    for k, v in other.items():
                        if lo <= k <= to:
                            base[k] = v
                s.px[t] = base
        # Massive daily (adjusted=true) overlays for names whose FMP daily series is missing / failed the pack check
        md = os.path.join(ROOT, 'data', 'massive_daily')
        s.massive = set()
        for fn in sorted(os.listdir(md)) if os.path.isdir(md) else []:
            if not fn.startswith(era + '_') or not fn.endswith('.csv'): continue
            t = fn[len(era) + 1:-4]
            rows = rd(os.path.join(md, fn))
            ds = [r['date'] for r in rows]
            mp = {r['date']: float(r['close']) for r in rows}
            old = s.px.get(t, {})
            base = {k: v for k, v in old.items() if not (ds[0] <= k <= ds[-1])}
            # drop the vendor segment on a side whose boundary level disagrees with Massive by > 2% (ticker reuse / basis)
            for side, dd in (('lo', ds[0]), ('hi', ds[-1])):
                if dd in old and abs(old[dd] / mp[dd] - 1) > 0.02 or (dd not in old):
                    base = {k: v for k, v in base.items() if (k > dd if side == 'lo' else k < dd)}
            base.update(mp)
            s.px[t] = base; s.src[t] = s.src.get(t, '') + '+massive[%s..%s]' % (ds[0], ds[-1]); s.massive.add(t)
        s.SES = sorted(s.px['SPY'])
        # basis-break repair (e.g. FMP unadjusted 2:1 splits such as COG 2007-04-02 / 2012-01-26):
        # level L(me) = vendor close / pack rank price at each month-end should be flat; a >25% step between consecutive
        # month-ends with a single daily jump inside that month => rescale all vendor prices before the jump.
        s.repairs = []
        mes = sorted({m for (_, m) in s.RP})
        for t in list(s.iv):
            p = s.px.get(t)
            if not p: continue
            for _rep in range(6):
                Ls = [(m, p[m] / s.RP[(t, m)]) for m in mes if m in p and (t, m) in s.RP]
                fixed = False
                for i in range(1, len(Ls)):
                    (m0, l0), (m1, l1) = Ls[i - 1], Ls[i]
                    step = l1 / l0
                    if abs(step - 1) <= 0.25: continue
                    seg = sorted(k for k in p if m0 < k <= m1)
                    prev = max([k for k in p if k <= m0], default=None)
                    if prev is None or not seg: continue
                    chain = [prev] + seg
                    jumps = [(chain[q], p[chain[q]] / p[chain[q - 1]]) for q in range(1, len(chain))]
                    dj, rj = min(jumps, key=lambda x: abs(math.log(x[1] / step)))
                    if abs(math.log(rj / step)) < 0.08:
                        for k in list(p):
                            if k < dj: p[k] *= rj
                        s.repairs.append((t, dj, round(rj, 4)))
                        fixed = True
                        break
                if not fixed: break
        s.dates = {t: sorted(v) for t, v in s.px.items()}

    def member(s, t, d):
        if s.snap is not None and d > '2025-11-28':
            c = [ts for (sd, ts) in s.snap if sd <= d]
            return t in c[-1]
        return any(a <= d < b for a, b in s.iv.get(t, []))

    def me(s, y, m):
        """last session of month (y,m)"""
        while m <= 0: m += 12; y -= 1
        while m > 12: m -= 12; y += 1
        pre = '%04d-%02d' % (y, m)
        c = [x for x in s.SES if x[:7] == pre]
        return c[-1] if c else None

    def price_on_or_before(s, t, d, maxback=10):
        """S&P NOTE 2: price on d, else the day prior, up to ten days prior (sessions)."""
        p = s.px.get(t, {})
        if d is None: return None, None
        if d in p: return d, p[d]
        i = _bisect_left(s.SES, d)
        for k in range(1, maxback + 1):
            if i - k < 0: break
            dd = s.SES[i - k]
            if dd in p: return dd, p[dd]
        return None, None

def _bisect_left(a, x):
    import bisect
    return bisect.bisect_left(a, x)

def third_friday(y, m):
    d = dt.date(y, m, 1)
    d += dt.timedelta(days=(4 - d.weekday()) % 7)
    return (d + dt.timedelta(days=14)).isoformat()

def rebalance_dates(U, y, m):
    """(reference date, execution date). Exec = 3rd Friday close; if market closed, last session before (Queen choice)."""
    ref = U.me(y, m - 1)
    tf = third_friday(y, m)
    ex = [x for x in U.SES if x <= tf][-1]
    return ref, ex

def score_table(U, y, m, excl=frozenset()):
    ref, ex = rebalance_dates(U, y, m)
    pe = U.me(y, m - 2); ps = U.me(y - 1, m - 2); ps9 = U.me(y, m - 11)
    hist_cut = U.me(y, m - 1 - 10)   # traded >= 10 months before reference date
    rows = []
    for t in U.iv:
        if t in excl or not U.member(t, ref): continue
        dts = U.dates.get(t, [])
        rec = dict(ticker=t, ref=ref, exec=ex, flag='')
        de, Pe = U.price_on_or_before(t, pe)
        ds, Ps = U.price_on_or_before(t, ps)
        a = U.RP.get((t, pe)); b = U.RP.get((t, ps))
        form = 'a'
        if Ps is None and b is None:
            # formula (b): nine months of history
            ds, Ps = U.price_on_or_before(t, ps9); b = U.RP.get((t, ps9)); form = 'b'
        pstart = ps if form == 'a' else ps9
        has_fmp = Pe is not None and Ps is not None and dts and dts[0] <= hist_cut
        # pack cross-check (pack month-end rank prices; spin/special-div F on the start price)
        mvp = None
        if a and b:
            F = 1.0
            for (exd, f) in U.CA.get(t, []):
                if pstart < exd <= pe: F *= f
            mvp = a / (b * F) - 1
        if has_fmp:
            # spin/special-div F inside the window: vendors (FMP, Massive) back-adjust some spins and not others.
            # Accept whichever version (raw ratio, or ratio with pack F) matches the pack's F-adjusted rank-price return.
            F = 1.0; exs = {}
            for (exd, f) in U.CA.get(t, []):
                if ds < exd <= de: F *= f; exs[exd] = f
            mv_raw = Pe / Ps - 1; mv_F = Pe / (Ps * F) - 1
            use_F = True
            if exs and mvp is not None and abs((1 + mv_raw) / (1 + mvp) - 1) <= 0.02 < abs((1 + mv_F) / (1 + mvp) - 1):
                use_F = False; rec['flag'] += 'vendor_already_spin_adjusted;'
            mv = mv_F if use_F else mv_raw
            seq = [x for x in dts if ds <= x <= de]
            r = []
            for i in range(1, len(seq)):
                q = U.px[t][seq[i]] / U.px[t][seq[i - 1]]
                if use_F and seq[i] in exs: q = q / exs[seq[i]]
                r.append(q - 1)
            sig = statistics.stdev(r) if len(r) > 20 else None
            if r and max(abs(x) for x in r) > 0.75:
                rec['flag'] += 'daily_jump>75%%(%.2f)->sigma_untrusted;' % max(abs(x) for x in r); sig = None
            if mvp is not None and abs((1 + mv) / (1 + mvp) - 1) > 0.02:
                rec['flag'] += 'fmp_vs_pack_mismatch(fmp=%.4f,pack=%.4f)->mv_from_pack,sigma_untrusted;' % (mv, mvp)
                mv = mvp; sig = None
            rec.update(mv=mv, sigma=sig, form=form)
        elif mvp is not None:
            rec.update(mv=mvp, sigma=None, form=form); rec['flag'] += 'no_fmp_daily;'
        else:
            rec['flag'] += 'excluded_no_price;'
            continue
        rows.append(rec)
    # names without an FMP daily series: sigma proxy from the pack's month-end rank prices over the same 12 months,
    # scaled to daily by the median ratio sigma_daily / sigma_monthly among names that have both (flagged).
    mes = [U.me(y - 1, m - 2 + k) for k in range(0, 13)]
    def msig(t):
        p = [U.RP.get((t, x)) for x in mes]
        if any(v is None for v in p): return None
        F = dict(U.CA.get(t, []))
        r = []
        for i in range(1, len(p)):
            q = p[i] / p[i - 1]
            for exd, f in U.CA.get(t, []):
                if mes[i - 1] < exd <= mes[i]: q /= f
            r.append(q - 1)
        return statistics.stdev(r)
    ratios = []
    for r in rows:
        if r.get('sigma'):
            ms = msig(r['ticker'])
            if ms: ratios.append(r['sigma'] / ms)
    kfac = statistics.median(ratios) if ratios else None
    sigs = [r['sigma'] for r in rows if r.get('sigma')]
    med = statistics.median(sigs)
    for r in rows:
        if not r.get('sigma'):
            ms = msig(r['ticker'])
            if ms and kfac:
                r['sigma'] = ms * kfac; r['flag'] += 'sigma=monthly_proxy(k=%.3f);' % kfac
            else:
                r['sigma'] = med; r['flag'] += 'sigma=universe_median;'
        r['ramv'] = r['mv'] / r['sigma']
    v = [r['ramv'] for r in rows]
    mu = statistics.mean(v); sd = statistics.pstdev(v)
    for r in rows:
        z = (r['ramv'] - mu) / sd
        r['z_raw'] = z
        zw = max(-3.0, min(3.0, z)); r['z'] = zw
        r['score'] = 1 + zw if zw > 0 else (1 / (1 - zw) if zw < 0 else 1.0)
    rows.sort(key=lambda r: (-r['score'], -r['ramv'], r['ticker']))
    for i, r in enumerate(rows): r['rank'] = i + 1
    return rows

def schedule(U, start, end):
    y0, y1 = int(start[:4]) - 1, int(end[:4])
    out = []
    for y in range(y0, y1 + 1):
        for m in (3, 9):
            if third_friday(y, m) < U.SES[0] or U.me(y, m - 1) is None: continue
            ref, ex = rebalance_dates(U, y, m)
            if ref is None: continue
            out.append((y, m, ref, ex))
    return out

def simulate(U, start, end, K, tables, stop=None, label=''):
    """tables: list of (exec_date, [tickers]) sorted. Returns dict with nav, trades, holdings, stats."""
    S = [d for d in U.SES if start <= d <= end]
    start, end = S[0], S[-1]
    tabs = sorted(tables)
    initial = [t for (e, t) in tabs if e <= start][-1]
    rebal = {e: t for (e, t) in tabs if start < e < end}
    cash = 1.0; sh = {}; peak = {}; lastp = {}; nav = []; events = []
    contrib = collections.defaultdict(float)  # log-growth contribution by ticker
    def px(t, d):
        p = U.px[t].get(d)
        return p
    def trade(d, targets):
        nonlocal cash
        prices = {t: lastp[t] for t in sh}
        for t in targets:
            p = px(t, d)
            if p is None: raise RuntimeError('no exec price %s %s %s' % (label, t, d))
            prices[t] = p
        h = {t: sh[t] * prices[t] for t in sh}
        E0 = cash + sum(h.values())
        Ep = E0
        for _ in range(100):
            T = {t: Ep / len(targets) for t in targets} if targets else {}
            fee = C * sum(abs(T.get(t, 0) - h.get(t, 0)) for t in set(T) | set(h))
            Ep2 = E0 - fee
            if abs(Ep2 - Ep) < 1e-14: break
            Ep = Ep2
        new = {t: T[t] / prices[t] for t in targets}
        for t in list(peak):
            if t not in new: del peak[t]
        for t in new:
            if t not in sh: peak[t] = prices[t]
            lastp[t] = prices[t]
        sh.clear(); sh.update(new); cash = 0.0 if targets else Ep
        events.append(dict(date=d, action='rebalance', targets=' '.join(targets), equity_after=Ep, fee=E0 - Ep))
    prevE = None; prevw = {}
    for i, d in enumerate(S):
        # corporate actions (spin / special div): keep value, scale shares and peak
        for t in list(sh):
            for (exd, f) in U.CA.get(t, []):
                if exd == d:
                    sh[t] /= f; peak[t] *= f
                    events.append(dict(date=d, action='ca_adjust', targets=t, equity_after=None, fee=f))
        # delisting / end of series inside a hold: cash out at last close with fee
        for t in list(sh):
            if U.px[t].get(d) is None and U.dates[t][-1] < d:
                v = sh[t] * lastp[t]; cash += v * (1 - C)
                events.append(dict(date=d, action='series_end_cashout(last %s)' % U.dates[t][-1], targets=t, equity_after=None, fee=v * C))
                del sh[t]; peak.pop(t, None)
        for t in sh:
            p = px(t, d)
            if p is not None: lastp[t] = p
        E = cash + sum(sh[t] * lastp[t] for t in sh)
        if prevE is not None and prevE > 0:
            R = E / prevE - 1
            if abs(R) > 1e-15:
                for t, (w, p0) in prevw.items():
                    if t in sh:
                        rt = lastp[t] / p0 - 1
                        contrib[t] += math.log1p(R) * (w * rt) / R
        if i == 0:
            trade(d, initial)
        elif d in rebal:
            trade(d, rebal[d])
        elif d == end:
            v = sum(sh[t] * lastp[t] for t in sh); cash += v * (1 - C)
            if sh: events.append(dict(date=d, action='END_liquidate', targets=' '.join(sh), equity_after=cash, fee=v * C))
            sh.clear()
        elif stop:
            for t in list(sh):
                peak[t] = max(peak[t], lastp[t])
                if lastp[t] <= (1 - stop) * peak[t]:
                    v = sh[t] * lastp[t]; cash += v * (1 - C)
                    events.append(dict(date=d, action='STOP(peak %.4f close %.4f)' % (peak[t], lastp[t]), targets=t, equity_after=None, fee=v * C))
                    del sh[t]; del peak[t]
        if not stop:
            for t in sh: peak[t] = max(peak.get(t, lastp[t]), lastp[t])
        E = cash + sum(sh[t] * lastp[t] for t in sh)
        prevE = E
        prevw = {t: (sh[t] * lastp[t] / E, lastp[t]) for t in sh} if E > 0 else {}
        nav.append((d, E, ' '.join(sorted(sh)) if sh else 'CASH'))
    return dict(nav=nav, events=events, contrib=dict(contrib), start=start, end=end)

def stats(nav):
    d0, d1 = nav[0][0], nav[-1][0]
    v1 = nav[-1][1]
    yrs = (dt.date.fromisoformat(d1) - dt.date.fromisoformat(d0)).days / 365.25
    pk = -1; mdd = 0
    for _, v, *_ in nav:
        pk = max(pk, v); mdd = min(mdd, v / pk - 1)
    return dict(cagr=v1 ** (1 / yrs) - 1, mdd=mdd, end_value=v1, years=yrs)

def yearly(nav):
    out = collections.OrderedDict(); prev = nav[0][1]; cur = nav[0][0][:4]; last = prev
    for d, v, *_ in nav:
        if d[:4] != cur:
            out[cur] = last / prev - 1; prev = last; cur = d[:4]
        last = v
    out[cur] = last / prev - 1
    return out

def bh(U, t, start, end):
    S = [d for d in U.SES if start <= d <= end and d in U.px.get(t, {})]
    if not S: return None
    nav = [(d, U.px[t][d] / U.px[t][S[0]]) for d in S]
    return nav
