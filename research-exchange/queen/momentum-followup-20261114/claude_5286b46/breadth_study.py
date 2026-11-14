#!/usr/bin/env python3
"""Claude: stop tuning W and H; test ONE long-lookback rule and vary the number of holdings.  Stdlib only.

Engine: month-end proxy (buy/sell at month-end closes from each pack's rank prices, 25 bp one-way on traded value,
spin/special-dividend factor F applied, a name whose price series ends is carried as cash at its last price).
Price only: no dividends, no cash interest. Previously validated against the exact engine for Top1-3
(cell-median rank correlation 0.92-0.98, mean abs difference 1-3 pp).

Design
- Signals at each month-end S, over that month's eligible S&P 500 members (365-day seasoned):
    W12      return over the last 12 months
    W10-12   average cross-sectional rank of the 10-, 11- and 12-month returns (the three overlap ~65%, so they are one signal)
    12-1     12-month return skipping the latest month (the academic standard)
    W6, W1   shorter lookbacks, for contrast
- Holdings K: 1, 3, 5, 10, 20, 50, equal weight.  Benchmark: ALL eligible names equal weight ("EW universe"), same engine.
- Holding period H: 1, 3, 6, 12 months, run as H staggered tranches (each 1/H of capital, rebalanced every H months,
  all bought at the first month-end), so the result does not depend on a start month.
- Decades: 2014-12-31 -> 2024-12-31 (frozen pack, with the twelve known price corrections applied) and
  2004-12-31 -> 2014-12-31 (Queen's PROVISIONAL earlier-decade pack).
Usage: python3 breadth_study.py   -> breadth_results.csv, breadth_yearly.csv, stdout"""
import csv,gzip,os,collections,statistics as st,math
HERE=os.path.dirname(os.path.abspath(__file__)); R=os.path.join(HERE,'..','..'); C=0.0025
PACKS={'2015-2025':(os.path.join(R,'shared','inputs-pack-full-20261109'),'2014-12-31'),'2005-2015':(os.path.join(R,'queen','w10h2-w12h4-oos-2005-2015-20261112','pack'),'2004-12-31')}
FIX=[r for r in csv.DictReader(open(os.path.join(HERE,'..','input-audit-20261112','price_corrections.csv'))) if r['kind']=='scale_prices_on_or_before']
SIGNALS=['W12','W10-12','12-1','W6','W1']; KS=[1,3,5,10,20,50]; HS=[1,3,6,12]
def load(P,fix):
    rd=lambda f:list(csv.DictReader(gzip.open(os.path.join(P,f),'rt') if f.endswith('.gz') else open(os.path.join(P,f))))
    px=collections.defaultdict(dict)
    for r in rd('rank_prices_month_end.csv.gz'):
        if r['rank_price']: px[r['engine_ticker']][r['month_end']]=float(r['rank_price'])
    if fix:
        for f in FIX:
            for d in px.get(f['engine_ticker'],{}):
                if d<=f['month_end']: px[f['engine_ticker']][d]*=float(f['factor'])
    el=collections.defaultdict(list)
    for r in rd('eligibility_by_signal.csv.gz'):
        if r['eligible_final']=='True': el[r['signal_date']].append(r['engine_ticker'])
    ca=collections.defaultdict(list)
    for r in rd('corporate_actions_applied.csv'):
        if r['kind'] in('spin','special_div') and r['factor_F']: ca[r['engine_ticker']].append((r['ex_date'],float(r['factor_F'])))
    return px,el,ca,sorted({d for v in px.values() for d in v})
class Era:
    def __init__(s,P,first,fix,excl=()):
        s.px,s.el,s.ca,s.ME=load(P,fix); s.mi={d:i for i,d in enumerate(s.ME)}; s.i0=s.mi[first]; s.excl=set(excl); s.cache={}
    def F(s,t,a,b):
        f=1.0
        for e,x in s.ca.get(t,()):
            if a<e<=b: f*=x
        return f
    def ret(s,t,a,b):
        pa,pb=s.px[t].get(a),s.px[t].get(b)
        return None if pa is None or pb is None else pb/(pa*s.F(t,a,b))-1
    def ranking(s,i,sig):
        if (i,sig) in s.cache: return s.cache[(i,sig)]
        S=s.ME[i]; names=[t for t in s.el[S] if t not in s.excl]; out=None
        def by(a,b):
            v=[(s.ret(t,s.ME[a],s.ME[b]),t) for t in names]; return sorted(((-r,t) for r,t in v if r is not None))
        if sig=='W12': out=[t for _,t in by(i-12,i)]
        elif sig=='W6': out=[t for _,t in by(i-6,i)]
        elif sig=='W1': out=[t for _,t in by(i-1,i)]
        elif sig=='12-1': out=[t for _,t in by(i-12,i-1)]
        elif sig=='W10-12':
            rk=collections.defaultdict(list)
            for W in (10,11,12):
                for j,(_,t) in enumerate(by(i-W,i)): rk[t].append(j)
            out=[t for _,t in sorted((sum(v)/3,t) for t,v in rk.items() if len(v)==3)]
        elif sig=='ALL': out=sorted(names)
        s.cache[(i,sig)]=out; return out
    def tranche(s,sig,K,H,off):
        """monthly equity path of one tranche: bought at i0, then rebalanced every H months at months where (m-off)%H==0"""
        hold={}; cash=1.0; path=[1.0]; n=0; to=0.0
        for m in range(s.i0,s.i0+120):
            if m==s.i0 or (m-s.i0-off)%H==0:
                r=s.ranking(m,sig); names=[t for t in (r if K is None else r[:K]) if s.ME[m] in s.px[t]]
                E0=cash+sum(hold.values()); Ep=E0
                for _ in range(100):
                    T=Ep/len(names); new=E0-C*(sum(abs(T-hold.get(t,0.0)) for t in names)+sum(v for t,v in hold.items() if t not in names))
                    if abs(new-Ep)<1e-13: break
                    Ep=new
                if m>s.i0: to+=(sum(abs(Ep/len(names)-hold.get(t,0.0)) for t in names)+sum(v for t,v in hold.items() if t not in names))/E0/2; n+=1
                hold={t:Ep/len(names) for t in names}; cash=0.0
            a,b=s.ME[m],s.ME[m+1]
            for t in list(hold):
                r=s.ret(t,a,b)
                if r is None: cash+=hold.pop(t)
                else: hold[t]*=1+r
            path.append(cash+sum(hold.values()))
        path[-1]=cash+sum(hold.values())*(1-C)
        return path,(to/n if n else 0.0)
    def composite(s,sig,K,H):
        tr=[s.tranche(sig,K,H,off) for off in range(H)]
        comp=[sum(p[j] for p,_ in tr)/H for j in range(121)]
        return comp,[p[-1]**0.1-1 for p,_ in tr],st.mean(t for _,t in tr)
def stats(p):
    pk=0;mdd=0
    for v in p: pk=max(pk,v); mdd=min(mdd,v/pk-1)
    yr=[p[12*(k+1)]/p[12*k]-1 for k in range(10)]
    return p[-1]**0.1-1,mdd,yr
def active_t(p,b):
    d=[(p[j+1]/p[j])-(b[j+1]/b[j]) for j in range(120)]; return st.mean(d)*12,st.mean(d)/st.stdev(d)*math.sqrt(120),d
rows=[]; yearly=[]; keep={}
for dec,(P,first) in PACKS.items():
    for variant,excl in ((('all names',()),('no NVDA',('NVDA',))) if dec=='2015-2025' else (('all names',()),)):
        E=Era(P,first,dec=='2015-2025',excl)
        bench,_,bto=E.composite('ALL',None,12); bc,bm,by=stats(bench); keep[(dec,variant,'EW')]=bench
        rows.append(dict(decade=dec,universe=variant,signal='EW universe',K='all',H=12,CAGR=bc,monthend_MDD=bm,worst_year=min(by),best_year=max(by),tranche_min_CAGR='',tranche_max_CAGR='',annual_turnover=bto,active_vs_EW_per_year='',active_t_stat='',years_beating_EW=''))
        for sig in SIGNALS:
            for H in HS:
                if variant=='no NVDA' and not (sig=='W10-12' and H==12): continue
                for K in KS:
                    comp,tc,to=E.composite(sig,K,H); c,m,y=stats(comp); a,t,d=active_t(comp,bench); keep[(dec,variant,sig,K,H)]=d
                    rows.append(dict(decade=dec,universe=variant,signal=sig,K=K,H=H,CAGR=c,monthend_MDD=m,worst_year=min(y),best_year=max(y),tranche_min_CAGR=min(tc),tranche_max_CAGR=max(tc),annual_turnover=to*12/H,active_vs_EW_per_year=a,active_t_stat=t,years_beating_EW=sum(1 for u,v in zip(y,by) if u>v)))
                    if sig=='W10-12' and H==12:
                        for k,(u,v) in enumerate(zip(y,by)): yearly.append(dict(decade=dec,universe=variant,K=K,year=int(first[:4])+1+k,rule_return=u,EW_universe_return=v))
w=csv.DictWriter(open(os.path.join(HERE,'breadth_results.csv'),'w',newline=''),fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
w=csv.DictWriter(open(os.path.join(HERE,'breadth_yearly.csv'),'w',newline=''),fieldnames=list(yearly[0])); w.writeheader(); w.writerows(yearly)
g=lambda dec,var,sig,K,H:[r for r in rows if (r['decade'],r['universe'],r['signal'],r['K'],r['H'])==(dec,var,sig,K,H)][0]
print('MAIN RULE: W10-12 blend, hold 12 months, 12 staggered tranches, equal weight.  (CAGR / month-end max drawdown / worst calendar year / range across the 12 tranches)')
for dec in PACKS:
    for var in (('all names','no NVDA') if dec=='2015-2025' else ('all names',)):
        b=g(dec,var,'EW universe','all',12); print(f'  {dec} {var}: EW universe {b["CAGR"]*100:5.1f}% / {b["monthend_MDD"]*100:4.0f}% / {b["worst_year"]*100:4.0f}%')
        for K in KS:
            r=g(dec,var,'W10-12',K,12); print(f'     Top{K:<3d} {r["CAGR"]*100:5.1f}% / {r["monthend_MDD"]*100:4.0f}% / {r["worst_year"]*100:4.0f}% / tranches {r["tranche_min_CAGR"]*100:5.1f}%..{r["tranche_max_CAGR"]*100:5.1f}% | vs EW {r["active_vs_EW_per_year"]*100:+5.1f} pp/yr (t={r["active_t_stat"]:+.2f}), beat EW in {r["years_beating_EW"]}/10 years, turnover {r["annual_turnover"]*100:.0f}%/yr')
print('\nSIGNAL x HOLD, Top20, CAGR minus EW universe (pp per year), [t-stat]:')
for dec in PACKS:
    print(' ',dec)
    for sig in SIGNALS: print('    %-7s'%sig,'  '.join('H%-2d %+5.1f [%+.1f]'%(H,g(dec,'all names',sig,20,H)['active_vs_EW_per_year']*100,g(dec,'all names',sig,20,H)['active_t_stat']) for H in HS))
print('\nPOOLED 20 YEARS (240 months), main rule, monthly return minus EW universe:')
for K in KS:
    d=keep[('2005-2015','all names','W10-12',K,12)]+keep[('2015-2025','all names','W10-12',K,12)]
    print('   Top%-3d mean %+5.1f pp/yr, t=%+.2f, months ahead %.0f%%'%(K,st.mean(d)*1200,st.mean(d)/st.stdev(d)*math.sqrt(len(d)),100*sum(1 for x in d if x>0)/len(d)))
