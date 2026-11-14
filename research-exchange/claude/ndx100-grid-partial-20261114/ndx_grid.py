#!/usr/bin/env python3
"""Claude: Nasdaq-100 momentum grid, PARTIAL data.  W 1-12 (lookback months) x H 1-12 (holding months) x K 1-4 x 12 start months = 6,912 paths.
Membership: ../ndx100-partial-test-20261114/ndx100_membership_month_end.csv (point-in-time). Prices: frozen S&P pack month-end rank prices,
~84% of Nasdaq-100 member-months (49 non-S&P tickers have no price and are left out) -> NOT a full Nasdaq-100 backtest.
Engine: month-end proxy (signal and trade at the same month-end close, 25 bp one-way, price only, takeover targets cashed out before deal date).
Each path: start at month-end 2014-12 + (phase-1), run 120 months, rebalance every H months to equal-weight Top K by W-month return.
Usage: python3 ndx_grid.py -> paths_6912*.csv, cells_576*.csv, stdout"""
import os,csv,collections,statistics as st
H_=os.path.dirname(os.path.abspath(__file__)); B=os.path.join(H_,'..','next-steps-20261114')
exec(open(os.path.join(B,'breadth_study.py')).read().split("rows=[]; yearly=[]")[0].replace("HERE=os.path.dirname(os.path.abspath(__file__))","HERE=B"))
mem={r['month_end']:[{'PCLN':'BKNG'}.get(t,t) for t in r['tickers'].split()] for r in csv.DictReader(open(os.path.join(H_,'..','ndx100-partial-test-20261114','ndx100_membership_month_end.csv')))}
P,first=PACKS['2015-2025']
def rank(E,i,W):
    k=(i,W)
    if k not in E.cache:
        v=[(E.ret(t,E.ME[i-W],E.ME[i]),t) for t in E.el[E.ME[i]]]; E.cache[k]=[t for _,t in sorted((-r,t) for r,t in v if r is not None)]
    return E.cache[k]
def path(E,W,Hh,K,ph):
    s=E.i0+ph-1; hold={}; cash=1.0; pk=0.0; mdd=0.0; picks=collections.Counter()
    for m in range(s,s+120):
        if (m-s)%Hh==0:
            names=(rank(E,m,W)[:K] if K else list(E.el[E.ME[m]])); E0=cash+sum(hold.values()); Ep=E0
            for _ in range(100):
                T=Ep/len(names); new=E0-C*(sum(abs(T-hold.get(t,0.0)) for t in names)+sum(v for t,v in hold.items() if t not in names))
                if abs(new-Ep)<1e-13: break
                Ep=new
            hold={t:Ep/len(names) for t in names}; cash=0.0
            for t in names: picks[t]+=1
        eq=cash+sum(hold.values()); pk=max(pk,eq); mdd=min(mdd,eq/pk-1)
        a,b=E.ME[m],E.ME[m+1]
        for t in list(hold):
            r=E.ret(t,a,b)
            if r is None: cash+=hold.pop(t)
            else: hold[t]*=1+r
    eq=cash+sum(hold.values())*(1-C); mdd=min(mdd,eq/max(pk,eq)-1)
    return eq,eq**0.1-1,mdd,picks
for tag,ex in (('',()),('_no_nvda',('NVDA',))):
    E=Era(P,first,True,ex)
    for d in E.ME: E.el[d]=[t for t in mem[d] if d in E.px.get(t,{}) and t not in ex]
    ew=[path(E,1,1,None,ph)[1] for ph in range(1,13)]
    rows=[]; cell=collections.defaultdict(list); pk=collections.defaultdict(collections.Counter)
    for W in range(1,13):
        for Hh in range(1,13):
            for K in range(1,5):
                for ph in range(1,13):
                    eq,c,m,p=path(E,W,Hh,K,ph); rows.append(dict(W=W,H=Hh,K=K,phase=ph,final_multiple=round(eq,4),CAGR=round(c,6),monthend_MDD=round(m,4)))
                    cell[(W,Hh,K)].append((c,m)); pk[(W,Hh,K)].update(p)
    w=csv.DictWriter(open(os.path.join(H_,f'paths_6912{tag}.csv'),'w',newline=''),fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    cr=[]
    for (W,Hh,K),v in cell.items():
        cs=[c for c,_ in v]; cr.append(dict(W=W,H=Hh,K=K,median_CAGR=round(st.median(cs),6),min_CAGR=round(min(cs),6),max_CAGR=round(max(cs),6),worst_MDD=round(min(m for _,m in v),4),top_picks=' '.join(f'{t}:{n}' for t,n in pk[(W,Hh,K)].most_common(5))))
    w=csv.DictWriter(open(os.path.join(H_,f'cells_576{tag}.csv'),'w',newline=''),fieldnames=list(cr[0])); w.writeheader(); w.writerows(cr)
    print(f'===== {"WITHOUT NVDA" if ex else "ALL COVERED NAMES"} =====  EW covered members, monthly rebalance: median CAGR {st.median(ew)*100:.1f}% (12 phases)')
    best=max(rows,key=lambda r:r['CAGR']); print('  best single path:',best)
    for K in range(1,5):
        ck=sorted([c for c in cr if c['K']==K],key=lambda c:-c['median_CAGR']); allm=[c['median_CAGR'] for c in ck]
        print(f'  K={K}: median of all 144 cells {st.median(allm)*100:.1f}% | cells beating 17.3% (QQQ price) {sum(1 for x in allm if x>0.173)}/144 | paths losing money {sum(1 for r in rows if r["K"]==K and r["CAGR"]<0)}/1728')
        for c in ck[:5]: print(f'     W{c["W"]:<2d} H{c["H"]:<2d} median {c["median_CAGR"]*100:5.1f}%  range {c["min_CAGR"]*100:5.1f}..{c["max_CAGR"]*100:5.1f}%  worst MDD {c["worst_MDD"]*100:4.0f}%  picks {c["top_picks"]}')
        print('     worst cell:',f'W{ck[-1]["W"]} H{ck[-1]["H"]} median {ck[-1]["median_CAGR"]*100:.1f}%')
    for K in (1,3):
        print(f'  K={K} median CAGR %, rows W1..W12, cols H1..H12')
        for W in range(1,13): print('    W%-2d '%W+' '.join('%4.0f'%(st.median([c for c,_ in cell[(W,Hh,K)]])*100) for Hh in range(1,13)))
    band=lambda Ws,Hs,K: st.median([c for W in Ws for Hh in Hs for c,_ in cell[(W,Hh,K)]])*100
    print('  bands (median of paths): '+' | '.join(f'K{K}: W10-12xH2-4 {band(range(10,13),range(2,5),K):.1f}%, W5-7xH1 {band(range(5,8),[1],K):.1f}%, W1-3xH1-3 {band(range(1,4),range(1,4),K):.1f}%, W10-12xH10-12 {band(range(10,13),range(10,13),K):.1f}%' for K in range(1,5)))
