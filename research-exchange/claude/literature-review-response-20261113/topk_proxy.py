#!/usr/bin/env python3
"""Claude: Top1 vs Top2 vs Top3 (equal weight) in both decades, month-end PROXY (same simplifications as
grid_persistence_proxy.py), validated against Queen's exact 2015-2025 answers. Stdlib only."""
import csv,gzip,os,collections,statistics as st
HERE=os.path.dirname(os.path.abspath(__file__)); R=os.path.join(HERE,'..','..'); C=0.0025
def load(P):
    rd=lambda f:list(csv.DictReader(gzip.open(os.path.join(P,f),'rt') if f.endswith('.gz') else open(os.path.join(P,f))))
    px=collections.defaultdict(dict)
    for r in rd('rank_prices_month_end.csv.gz'):
        if r['rank_price']: px[r['engine_ticker']][r['month_end']]=float(r['rank_price'])
    el=collections.defaultdict(list)
    for r in rd('eligibility_by_signal.csv.gz'):
        if r['eligible_final']=='True': el[r['signal_date']].append(r['engine_ticker'])
    ca=collections.defaultdict(list)
    for r in rd('corporate_actions_applied.csv'):
        if r['kind'] in('spin','special_div') and r['factor_F']: ca[r['engine_ticker']].append((r['ex_date'],float(r['factor_F'])))
    return px,el,ca,sorted({d for v in px.values() for d in v})
def F(ca,t,a,b):
    f=1.0
    for e,x in ca.get(t,()):
        if a<e<=b: f*=x
    return f
def run(P,first_signal,cells,Ks=(1,2,3)):
    px,el,ca,ME=load(P); mi={d:i for i,d in enumerate(ME)}; cache={}
    def top(S,W):
        if (S,W) not in cache:
            B=ME[mi[S]-W]; cache[(S,W)]=[t for _,t in sorted((-(px[t][S]/(px[t][B]*F(ca,t,B,S))-1),t) for t in el[S] if S in px[t] and B in px[t])[:3]]
        return cache[(S,W)]
    out={}; i0=mi[first_signal]
    for (W,H) in cells:
        for K in Ks:
            res=[]
            for ph in range(12):
                s=i0+ph; e=s+120
                if e>=len(ME) or ME[s+(119//H)*H] not in el: continue
                hold={}; cash=1.0; k=s; pk=0; mdd=0
                while k<e:
                    names=top(ME[k],W)[:K]; E0=cash+sum(hold.values()); Ep=E0
                    for _ in range(100):
                        T=Ep/K; new=E0-C*(sum(abs(T-hold.get(t,0.0)) for t in names)+sum(v for t,v in hold.items() if t not in names))
                        if abs(new-Ep)<1e-13: break
                        Ep=new
                    hold={t:Ep/K for t in names}; cash=0.0
                    for m in range(k+1,min(k+H,e)+1):        # month by month, for drawdown
                        a,b=ME[m-1],ME[m]
                        for t in list(hold):
                            if b in px[t] and a in px[t]: hold[t]*=px[t][b]/(px[t][a]*F(ca,t,a,b))
                            elif a in px[t]: cash+=hold.pop(t)       # series ended: carried as cash
                        v=cash+sum(hold.values()); pk=max(pk,v); mdd=min(mdd,v/pk-1)
                    k=min(k+H,e)
                eq=cash+sum(hold.values())*(1-C); res.append((eq**0.1-1,mdd))
            out[(W,H,K)]=res
    return out
cells=[(W,H) for W in (10,11,12) for H in (2,3,4)]
g15=run(os.path.join(R,'shared','inputs-pack-full-20261109'),'2014-12-31',cells)
g05=run(os.path.join(R,'queen','w10h2-w12h4-oos-2005-2015-20261112','pack'),'2004-12-31',cells)
exact=collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(R,'queen','answer-key-full-20261109','paths_summary.csv'))):
    if r['status']!='UNKNOWN': exact[(int(r['W']),int(r['H']),int(r['K']))].append(float(r['CAGR']))
rows=[]
for K in (1,2,3):
    a=[x[0] for c in cells for x in g15[c+(K,)]]; d=[x[1] for c in cells for x in g15[c+(K,)]]
    b=[x[0] for c in cells for x in g05[c+(K,)]]; e=[x[1] for c in cells for x in g05[c+(K,)]]
    ex=[x for c in cells for x in exact[c+(K,)]]
    print('Top%d | 2015-25 proxy median CAGR %.1f%% (Queen exact %.1f%%), month-end MDD median %.0f%% | 2005-15 proxy median CAGR %.1f%% (range %.1f%%..%.1f%%, share of paths > 0: %.0f%%), month-end MDD median %.0f%%, worst %.0f%%'%(K,st.median(a)*100,st.median(ex)*100,st.median(d)*100,st.median(b)*100,min(b)*100,max(b)*100,100*sum(1 for x in b if x>0)/len(b),st.median(e)*100,min(e)*100))
    rows.append(dict(K=K,paths_2015_2025=len(a),proxy_median_CAGR_2015_2025=st.median(a),queen_exact_median_CAGR_2015_2025=st.median(ex),proxy_median_monthend_MDD_2015_2025=st.median(d),paths_2005_2015=len(b),proxy_median_CAGR_2005_2015=st.median(b),proxy_min_CAGR_2005_2015=min(b),proxy_max_CAGR_2005_2015=max(b),proxy_median_monthend_MDD_2005_2015=st.median(e),proxy_worst_monthend_MDD_2005_2015=min(e)))
for c in ((10,2),(12,4)):
    print(c,[('Top%d'%K,'%.1f%%'%(st.median(x[0] for x in g05[c+(K,)])*100)) for K in (1,2,3)],'2005-15 |',[('Top%d'%K,'%.1f%%'%(st.median(x[0] for x in g15[c+(K,)])*100)) for K in (1,2,3)],'2015-25')
w=csv.DictWriter(open(os.path.join(HERE,'topk_proxy.csv'),'w',newline=''),fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
