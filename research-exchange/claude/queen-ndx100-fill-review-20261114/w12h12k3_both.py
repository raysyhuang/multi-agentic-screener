#!/usr/bin/env python3
"""Claude: like-for-like W12 x H12 x Top3 x 12 start months, S&P 500 universe vs Nasdaq-100 universe (my month-end engine, 25 bp, price only).
Also the whole-grid Top3 medians.  Usage: python3 w12h12k3_both.py"""
import os,csv,collections,statistics as st,datetime as dt
H=os.path.dirname(os.path.abspath(__file__)); V=os.path.join(H,'..','..','queen','ndx100-fill-20261114-v1.1')
exec(open(os.path.join(H,'verify_queen_ndx.py')).read().split("P=lambda x:")[0].replace("Q=os.path.join(H,'..','..','queen','ndx100-fill-20261114')","Q=V"))
g={'B':os.path.join(H,'..','next-steps-20261114')}
exec(open(os.path.join(g['B'],'breadth_study.py')).read().split("rows=[]; yearly=[]")[0].replace("HERE=os.path.dirname(os.path.abspath(__file__))","HERE=B"),g)
E=g['Era'](g['PACKS']['2015-2025'][0],'2014-12-31',True,()); C=0.0025
def sp_path(W,Hh,K,ph,n=120):
    s=E.i0+ph-1; hold={}; cash=1.0; eq=[1.0]
    for m in range(s,s+n):
        if (m-s)%Hh==0:
            S=E.ME[m]; v=sorted((-(E.ret(t,E.ME[m-W],S)),t) for t in E.el[S] if E.ret(t,E.ME[m-W],S) is not None); names=[t for _,t in v[:K]]
            E0=cash+sum(hold.values()); Ep=E0
            for _ in range(200):
                T=Ep/len(names); new=E0-C*(sum(abs(T-hold.get(t,0.0)) for t in names)+sum(x for t,x in hold.items() if t not in names))
                if abs(new-Ep)<1e-14: break
                Ep=new
            hold={t:Ep/len(names) for t in names}; cash=0.0
        for t in list(hold):
            r=E.ret(t,E.ME[m],E.ME[m+1])
            if r is None: cash+=hold.pop(t)
            else: hold[t]*=1+r
        eq.append(cash+sum(hold.values()))
    eq[-1]=cash+sum(hold.values())*(1-C); pk=0;mdd=0
    for x in eq: pk=max(pk,x); mdd=min(mdd,x/pk-1)
    return eq[-1]**(12/n)-1,mdd
R=Run(); Rn=Run(excl=('NVDA',)); i0=IDX['2014-12-31']
def show(lab,v): print(f'  {lab}: 中位数 {st.median([c for c,_ in v])*100:.1f}% | 12 个起始月 {min(c for c,_ in v)*100:.1f}%..{max(c for c,_ in v)*100:.1f}% | 最深月末回撤 {min(m for _,m in v)*100:.0f}%')
print('W12/H12/Top3, 12 个起始月, 各 120 个月 (2014-12..2015-11 起)')
show('标普500 (我的月末引擎)',[sp_path(12,12,3,ph) for ph in range(1,13)])
show('纳指100',[R.path(12,3,12,i0+ph-1,120)[1:] for ph in range(1,13)])
show('纳指100 去掉NVDA',[Rn.path(12,3,12,i0+ph-1,120)[1:] for ph in range(1,13)])
ak=[r for r in csv.DictReader(open(os.path.join(H,'..','..','queen','answer-key-full-20261109','paths_summary.csv')))]
kc=[c for c in ak[0] if c.lower() in ('cagr','cagr_pct','annualized_return')] or [c for c in ak[0] if 'cagr' in c.lower()]; print('  answer-key columns:',list(ak[0])[:12])
try:
    Wc=[c for c in ak[0] if c.lower() in('w','window','lookback_months')][0]; Hc=[c for c in ak[0] if c.lower() in('h','hold','holding_months')][0]; Kc=[c for c in ak[0] if c.lower() in('k','top_k','topk')][0]
    v=[float(r[kc[0]]) for r in ak if (r[Wc],r[Hc],r[Kc])==('12','12','3') and r[kc[0]] not in('','UNKNOWN')]; print(f'  标普500 精确引擎 (答案): W12/H12/K3 中位数 {st.median(v):.4g} ({len(v)} 条)')
    cells=collections.defaultdict(list)
    for r in ak:
        if r[Kc]=='3' and r[kc[0]] not in('','UNKNOWN'): cells[(r[Wc],r[Hc])].append(float(r[kc[0]]))
    print(f'  标普500 精确引擎: Top3 全部 {len(cells)} 格的中位数 {st.median([st.median(x) for x in cells.values()]):.4g}')
except Exception as ex: print('  answer key parse:',ex)
gm=collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(H,'grid_cell_medians.csv'))):
    if r['K']=='3': gm[r['run']].append(float(r['median_CAGR']))
for k,v in gm.items(): print(f'  纳指100 网格 {k}: Top3 144 格中位数 {st.median(v)*100:.1f}%')
print('纳指100 W12/H12/Top3 延长到 2026-11-13 (标普数据包只到 2025-11, 无法延长)')
v=[]
for ph in range(1,13):
    s=i0+ph-1; n=len(ME)-1-s; e,c,m=R.path(12,3,12,s,n); y=(dt.date.fromisoformat(ME[-1])-dt.date.fromisoformat(ME[s])).days/365.25; v.append((e[-1]**(1/y)-1,m))
show('纳指100 至 2026-11-13',v)
