#!/usr/bin/env python3
"""Claude input-data audit of shared/inputs-pack-full-20261109 (research only; reads the frozen pack, never writes to it).

Usage (from repo root, Python 3 standard library only):
    python research-exchange/claude/input-audit-20261112/audit_inputs.py

What it does
 1. Scans rank_prices_month_end for month-to-month moves shaped like unadjusted splits / identity switches.
 2. Applies the corrections listed in price_corrections.csv (each backed by an FMP split record in
    fmp_splits_fetched.csv) and re-ranks all 1,584 (signal, W) tables with an independent ranker.
 3. Reports every (signal, W, ticker) whose lookback return was wrong, and whether any Top1-3 pick changes.
 4. Compares the two price vendors inside the pack (Massive rank prices vs FMP execution closes) on monthly returns.
 5. Compares membership_intervals against eligibility_by_signal on all signal dates.
Outputs are written next to this script.
"""
import csv,gzip,collections,os,pickle,statistics as st
HERE=os.path.dirname(os.path.abspath(__file__))
P=os.path.join(HERE,'..','..','shared','inputs-pack-full-20261109')+os.sep
def rd(f): return list(csv.DictReader(gzip.open(P+f,'rt') if f.endswith('.gz') else open(P+f)))
def wr(name,rows,cols=None):
    with open(os.path.join(HERE,name),'w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=cols or list(rows[0])); w.writeheader(); w.writerows(rows)
sess=[r['session'] for r in rd('trading_sessions.csv')]
ME=[d for i,d in enumerate(sess) if i+1==len(sess) or sess[i+1][:7]!=d[:7]]; mei={d:i for i,d in enumerate(ME)}
eligrows=rd('eligibility_by_signal.csv.gz')
px0={(r['engine_ticker'],r['month_end']):float(r['rank_price']) for r in rd('rank_prices_month_end.csv.gz') if r['rank_price']!=''}
bytick=collections.defaultdict(dict)
for (t,d),p in px0.items(): bytick[t][d]=p
ov=collections.defaultdict(list)
for r in rd('rank_price_overrides.csv'): ov[(r['engine_ticker'],r['month_end'])].append((r['signal_from'],r['signal_to'],float(r['rank_price'])))
ca=collections.defaultdict(list)
for r in rd('corporate_actions_applied.csv'): ca[r['engine_ticker']].append(r)
elig=collections.defaultdict(list); member=collections.defaultdict(set)
for r in eligrows:
    if r['eligible_final']=='True': elig[r['signal_date']].append(r['engine_ticker'])
    if r['member_at_signal']=='True': member[r['signal_date']].add(r['engine_ticker'])
eligset=collections.defaultdict(set)
for S,v in elig.items():
    for t in v: eligset[t].add(S)

# ---- 1. split-shaped move scan -------------------------------------------------------------
scan=[]
for t,d in bytick.items():
    ds=sorted(d)
    for a,b in zip(ds,ds[1:]):
        if mei[b]-mei[a]!=1: continue
        r=d[b]/d[a]-1
        rel=[S for S in eligset[t] if b<=S and mei[S]-mei[b]<=11]   # move sits inside some usable lookback
        if rel and (r<=-0.20 or r>=0.60):
            scan.append(dict(month_end=b,engine_ticker=t,prev_month_end=a,prev_price=d[a],price=d[b],monthly_return=r,
                             pack_corporate_actions_in_month='|'.join(e['kind'] for e in ca[t] if a<e['ex_date']<=b),
                             n_signals_affected=len(rel)))
scan.sort(key=lambda x:(x['month_end'],x['engine_ticker']))
wr('scan_large_monthly_moves.csv',scan)

# ---- 2/3. corrections and re-rank ------------------------------------------------------------
FIX=[r for r in csv.DictReader(open(os.path.join(HERE,'price_corrections.csv')))]
def build(fix):
    px=dict(px0); el={S:list(v) for S,v in elig.items()}
    if fix:
        for f in FIX:
            if f['kind']=='scale_prices_on_or_before':
                for (t,d),p in px0.items():
                    if t==f['engine_ticker'] and d<=f['month_end']: px[(t,d)]=px[(t,d)]*float(f['factor'])
            elif f['kind']=='identity_stitch':      # TT history = old IR / factor ; new IR not seasoned for 12 signals
                for (t,d),p in px0.items():
                    if t=='IR' and d<=f['month_end']: px[('TT',d)]=p*float(f['factor'])
                for S in el:
                    if '2020-03-31'<=S<='2021-02-26':
                        if 'IR' in el[S]: el[S].remove('IR')
                        if 'TT' not in el[S]: el[S].append('TT')
    def price(t,d,S):
        for a,b,p in ov.get((t,d),()):
            if a<=S<=b: return p
        return px.get((t,d))
    out={}
    for S in sorted(el):
        for W in range(1,13):
            B=ME[mei[S]-W]; rows=[]; bad=[]
            for t in el[S]:
                p1=price(t,S,S); p0=price(t,B,S); ok=p1 is not None and p0 is not None; F=1.0
                for e in ca.get(t,()):
                    k=e['kind']
                    if k in('spin','special_div') and B<e['ex_date']<=S:
                        if e['factor_F']=='': ok=False
                        else: F*=float(e['factor_F'])
                    elif k=='unresolved_BLOCKER' and B<e['ex_date']<=S: ok=False
                    elif k=='legacy_data_error_BLOCKER' and e['blocker_from']<=S<=e['blocker_to']: ok=False
                if not ok: bad.append(t)
                else: rows.append((-(p1/(p0*F)-1),t))
            rows.sort(); out[(S,W)]=(bad,[(t,-r) for r,t in rows])
    return out
base=build(False); fixed=build(True)
cmp_rows=[]; aff=[]; names={f['engine_ticker'] for f in FIX}|{'TT','IR'}
for k in sorted(base):
    b=base[k][1]; f=fixed[k][1]
    cmp_rows.append(dict(signal_date=k[0],W=k[1],status='UNKNOWN' if base[k][0] else 'RANKED',
        pack_top1=b[0][0],pack_top2=b[1][0],pack_top3=b[2][0],pack_ret3=b[2][1],
        corrected_top1=f[0][0],corrected_top2=f[1][0],corrected_top3=f[2][0],corrected_ret3=f[2][1],
        top3_changed=[x[0] for x in b[:3]]!=[x[0] for x in f[:3]],top10_changed=[x[0] for x in b[:10]]!=[x[0] for x in f[:10]]))
    bi={t:(i+1,r) for i,(t,r) in enumerate(b)}; fi={t:(i+1,r) for i,(t,r) in enumerate(f)}
    for t in sorted(names):
        rb=bi.get(t); rf=fi.get(t)
        if rb==rf or (rb and rf and abs(rb[1]-rf[1])<1e-12): continue
        aff.append(dict(signal_date=k[0],W=k[1],engine_ticker=t,pack_return=rb[1] if rb else '',pack_rank=rb[0] if rb else '',
            corrected_return=rf[1] if rf else '',corrected_rank=rf[0] if rf else '',top3_threshold_return=f[2][1],
            gap_to_top3_pp=(f[2][1]-rf[1])*100 if rf else ''))
wr('top3_pack_vs_corrected_1584.csv',cmp_rows); wr('affected_rank_rows.csv',aff)

# ---- 4. vendor cross-check on monthly returns ------------------------------------------------
ex=collections.defaultdict(dict)
for r in rd('execution_daily_closes.csv.gz'): ex[(r['price_series'],r['execution_key'])][r['date']]=float(r['close'])
n=0; dis=[]
for (s,k),d in ex.items():
    if not s.startswith('fmp:') or k not in bytick: continue
    ms=sorted(x for x in d if x in bytick[k])
    for a,b in zip(ms,ms[1:]):
        if mei[b]-mei[a]!=1: continue
        n+=1; r1=d[b]/d[a]; r2=bytick[k][b]/bytick[k][a]
        if abs(r1/r2-1)>0.005: dis.append((s,a,b,r1-1,r2-1))

# ---- 5. intervals vs eligibility --------------------------------------------------------------
iv=rd('membership_intervals.csv'); mm=0
for S in sorted(member):
    a={r['engine_ticker'] for r in iv if r['start_date']<=S and (r['end_date_exclusive']=='' or S<r['end_date_exclusive'])}
    mm+= a!=member[S]

by=collections.defaultdict(list)
for r in aff: by[r['engine_ticker']].append(r)
lines=['tables ranked: %d'%len(base),
       'Top3 changed after corrections: %d'%sum(r['top3_changed'] for r in cmp_rows),
       'Top10 changed after corrections: %d %s'%(sum(r['top10_changed'] for r in cmp_rows),[(r['signal_date'],r['W']) for r in cmp_rows if r['top10_changed']]),
       'affected (signal, W, ticker) cells: %d'%len(aff)]
for t,v in sorted(by.items()):
    g=[x for x in v if x['corrected_rank']!='']
    lines.append('  %-6s cells %3d  signals %s..%s  worst pack return %7.1f%%  best corrected rank %s  closest gap to Top3 %s pp'%(
        t,len(v),min(x['signal_date'] for x in v),max(x['signal_date'] for x in v),
        min([x['pack_return'] for x in v if x['pack_return']!=''] or [0])*100,
        min([x['corrected_rank'] for x in g]) if g else 'n/a (removed)',
        round(min(x['gap_to_top3_pp'] for x in g),1) if g else 'n/a'))
lines+=['scan rows (monthly move <= -20%% or >= +60%% inside a usable lookback): %d'%len(scan),
        'vendor cross-check: %d monthly-return comparisons (FMP execution vs Massive rank), disagreements >0.5%%: %d'%(n,len(dis)),
        'membership intervals vs eligibility: %d of %d signal dates differ'%(mm,len(member))]
open(os.path.join(HERE,'audit_stdout.txt'),'w').write('\n'.join(lines)+'\n'); print('\n'.join(lines))

# ---- 6. dump the corrected rankings so the executor and stop test can be rerun on them (file is not committed) ----
pickle.dump({k:dict(bad=v[0],top=v[1][:5],n=len(v[0])+len(v[1])) for k,v in fixed.items()},open(os.path.join(HERE,'rank_corrected.pkl'),'wb'))
