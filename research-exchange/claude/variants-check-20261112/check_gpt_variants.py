#!/usr/bin/env python3
"""Claude's own implementation of GPT's two signal variants (skip last month; SPY 10-month trend filter) for
W10H2 and W12H4, Top1, 12 phases, on the frozen 2015-2025 pack. Written from GPT's REPORT.md definitions, not from
run_variants.py. Compares path by path with gpt/signal-variants-20261112/variant_paths.csv.
Usage (stdlib only): python3 check_gpt_variants.py"""
import csv,gzip,os,collections,datetime as dt,statistics as st
HERE=os.path.dirname(os.path.abspath(__file__)); R=os.path.join(HERE,'..','..')
P=os.path.join(R,'shared','inputs-pack-full-20261109')+os.sep; G=os.path.join(R,'gpt','signal-variants-20261112')+os.sep; C=0.0025
def rd(f): return list(csv.DictReader(gzip.open(P+f,'rt') if f.endswith('.gz') else open(P+f)))
sess=[r['session'] for r in rd('trading_sessions.csv')]; si={d:i for i,d in enumerate(sess)}
first={}
for d in sess: first.setdefault(d[:7],d)
ME=[d for i,d in enumerate(sess) if i+1==len(sess) or sess[i+1][:7]!=d[:7]]; mei={d:i for i,d in enumerate(ME)}
elig=collections.defaultdict(list)
for r in rd('eligibility_by_signal.csv.gz'):
    if r['eligible_final']=='True': elig[r['signal_date']].append(r['engine_ticker'])
px={(r['engine_ticker'],r['month_end']):float(r['rank_price']) for r in rd('rank_prices_month_end.csv.gz') if r['rank_price']!=''}
ov=collections.defaultdict(list)
for r in rd('rank_price_overrides.csv'): ov[(r['engine_ticker'],r['month_end'])].append((r['signal_from'],r['signal_to'],float(r['rank_price'])))
ca=collections.defaultdict(list)
for r in rd('corporate_actions_applied.csv'): ca[r['engine_ticker']].append(r)
def price(t,d,S):
    for a,b,p in ov.get((t,d),()):
        if a<=S<=b: return p
    return px.get((t,d))
def top1(S,W,skip):
    """skip=0: return from ME[i-W] to S.  skip=1: return from ME[i-W] to ME[i-1] (most recent month left out)."""
    i=mei[S]; B=ME[i-W]; E=ME[i-skip]; rows=[]; bad=False
    for t in elig[S]:
        p1=price(t,E,S); p0=price(t,B,S); ok=p1 is not None and p0 is not None; F=1.0
        for e in ca.get(t,()):
            k=e['kind']
            if k in('spin','special_div') and B<e['ex_date']<=E:
                if e['factor_F']=='': ok=False
                else: F*=float(e['factor_F'])
            elif k=='unresolved_BLOCKER' and B<e['ex_date']<=S: ok=False
            elif k=='legacy_data_error_BLOCKER' and e['blocker_from']<=S<=e['blocker_to']: ok=False
        if not ok: bad=True
        else: rows.append((-(p1/(p0*F)-1),t))
    return None if bad else min(rows)[1]
# SPY trend: month-end close above the mean of the latest ten month-end closes (signal month included); unadjusted closes
spy={r['date']:float(r['close']) for r in csv.DictReader(open(G+'spy_prices_fmp_2013-2025.csv'))}
spyadj={r['date']:float(r['adjClose']) for r in csv.DictReader(open(G+'spy_prices_fmp_2013-2025.csv'))}
def risk_on(S):
    i=mei[S]; v=[spy[ME[j]] for j in range(i-9,i+1)]
    return spy[S]>sum(v)/10
avail={}; availday=collections.defaultdict(list)
for r in rd('execution_availability.csv.gz'):
    avail[(r['trade_date'],r['engine_ticker'],r['window_end'])]=r; availday[(r['trade_date'],r['engine_ticker'])].append(r)
close={(r['price_series'],r['execution_key'],r['date']):float(r['close']) for r in rd('execution_daily_closes.csv.gz')}
lastd={(r['price_series'],r['execution_key']):r['series_last_date'] for r in rd('series_info.csv')}
hev=collections.defaultdict(list)
for r in rd('holding_events.csv'): hev[(r['price_series'],r['execution_key'])].append(r)
deal=collections.defaultdict(list)
for r in rd('untradable_deal_dates.csv'): deal[r['engine_ticker']].append(r['deal_date'])
def ekey(t,d):
    if t=='AGN': return 'AGN_LEGACY' if d<'2015-03-17' else 'AGN_PLC'
    return 'AGN_PLC' if t=='ACT' else t
def fl(x): return float(x) if x not in('',None) else 0.0
def ym(m): return '%04d-%02d'%(2015+m//12,m%12+1)
class Fail(Exception): pass
def run(W,H,ph,variant):
    END=first[ym(120+ph-1)]; rebs=[]; m=ph-1
    while m<120+ph-1: rebs.append(first[ym(m)]); m+=H
    nxt={d:(rebs[i+1] if i+1<len(rebs) else END) for i,d in enumerate(rebs)}
    D0=rebs[0]; hold=None; cash=1.0; peak=0.0; mdd=0.0; log=[]; cashdays=0; fb=0
    for i in range(si[D0],si[END]+1):
        d=sess[i]; prev=sess[i-1]
        if hold:
            key,s,n=hold
            evs=[e for e in hev.get((s,key),()) if prev<e['ex_date']<=d]
            if any(e['type']=='unresolved' or e['action']=='fail' for e in evs): raise Fail('corporate_action_unresolved')
            for e in evs:
                if e['action']!='apply': continue
                t=e['type']; cps=fl(e['cash_per_share'])
                if t in('spin','special','special_div'): cash+=n*fl(e['ratio'])*fl(e['child_first_close'])*(1-C)+n*cps
                elif t=='cash': cash+=n*cps; hold=None; break
                elif t=='stock': cash+=n*cps; hold=(e['child_or_acquirer'],e['acquirer_series'],n*fl(e['ratio'])); break
        mv=0.0
        if hold:
            key,s,n=hold; p=close.get((s,key,d))
            if p is None: raise Fail('corporate_action_unresolved' if d>lastd.get((s,key),'9999') else 'daily_gap')
            mv=n*p
        else: cashdays+=1
        if d in nxt:
            S=prev
            if variant=='spy_sma10' and not risk_on(S):
                cash=cash+mv*(1-C) if hold else cash; hold=None; eq=cash
            else:
                t=top1(S,W,1 if variant=='skip1' else 0)
                if t is None: raise Fail('rank_missing_unbounded')
                k=ekey(t,d); log.append((S,d,t))
                if hold and hold[0]==k: eq=cash+mv; n2=eq/close[(hold[1],k,d)]; hold=(k,hold[1],n2); cash=0.0   # keep; reinvest any cash (fee on the small top-up ignored only if zero)
                else:
                    a=avail.get((d,t,nxt[d]))
                    if a is None:
                        s0='fmp:'+k
                        if (s0,k,d) in close and (s0,k,nxt[d]) in close: a={'status':'ok','price_series':s0}; fb+=1
                        else: raise Fail('NOT_IN_PACK')
                    if a['status']=='not_tradable': raise Fail('pick_not_tradable')
                    if a['status']!='ok': raise Fail('exec_price_missing')
                    E0=cash+mv; Ep=(E0-C*mv)/(1+C)      # sell all (fee on mv), buy Ep (fee on Ep)
                    hold=(k,a['price_series'],Ep/close[(a['price_series'],k,d)]); cash=0.0; eq=Ep
        else: eq=cash+mv
        if d==END: eq=cash+mv*(1-C)
        peak=max(peak,eq); mdd=min(mdd,eq/peak-1)
    for S,d,t in log:
        if any(S<x<=d for x in deal.get(t,())): raise Fail('pick_not_tradable')
    yrs=(dt.date.fromisoformat(END)-dt.date.fromisoformat(D0)).days/365.25
    return eq,eq**(1/yrs)-1,mdd,cashdays/(si[END]-si[D0]+1),fb,D0,END
ref={(r['variant'],r['W'],r['H'],r['phase']):r for r in csv.DictReader(open(G+'variant_paths.csv'))}
rows=[]; res=collections.defaultdict(list); mx=0; mism=0
for (v,W,H,ph),q in ref.items():
    try:
        f,c,m,cp,fb,D0,END=run(int(W),int(H),int(ph),v)
        sp=(spyadj[END]/spyadj[D0])**(1/((dt.date.fromisoformat(END)-dt.date.fromisoformat(D0)).days/365.25))-1
        d=max(abs(c-float(q['CAGR'])),abs(m-float(q['daily_MDD']))) if q['status']=='KNOWN' else 9; mx=max(mx,d); mism+=d>1e-9
        res[(v,W,H)].append((c,m,cp,c>sp))
        rows.append(dict(variant=v,W=W,H=H,phase=ph,claude_status='KNOWN',claude_final=f,claude_CAGR=c,claude_MDD=m,claude_cash_pct=cp,fallback_series_uses=fb,gpt_status=q['status'],gpt_CAGR=q['CAGR'],gpt_MDD=q['daily_MDD'],abs_diff=d))
    except Fail as e:
        mism+=q['status']=='KNOWN'
        rows.append(dict(variant=v,W=W,H=H,phase=ph,claude_status='UNKNOWN:'+str(e),claude_final='',claude_CAGR='',claude_MDD='',claude_cash_pct='',fallback_series_uses='',gpt_status=q['status'],gpt_CAGR=q['CAGR'],gpt_MDD=q['daily_MDD'],abs_diff=''))
w=csv.DictWriter(open(os.path.join(HERE,'claude_vs_gpt_72_paths.csv'),'w',newline=''),fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('paths',len(rows),'mismatching (>1e-9 or status)',mism,'max abs diff %.2e'%mx)
for k,v in sorted(res.items(),key=lambda x:(x[0][1],x[0][0])):
    c=sorted(x[0] for x in v)
    print('W%sH%s %-9s n=%d median CAGR %.2f%% range %.2f%%..%.2f%% median MDD %.2f%% worst %.2f%% median cash %.1f%% beat SPY total %d/%d'%(k[1],k[2],k[0],len(v),st.median(c)*100,c[0]*100,c[-1]*100,st.median(x[1] for x in v)*100,min(x[1] for x in v)*100,st.median(x[2] for x in v)*100,sum(x[3] for x in v),len(v)))
