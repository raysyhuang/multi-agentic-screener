#!/usr/bin/env python3
"""Claude: same rule (mean of 6 monthly returns, Top3, monthly, 25 bp, price only), S&P 500 universe vs Nasdaq-100 universe, 2014-12 -> 2024-12.
How much do the picks overlap, and where does the return gap come from?  Usage: python3 sp500_vs_ndx_overlap.py"""
import os,collections,statistics as st
H=os.path.dirname(os.path.abspath(__file__)); V=os.path.join(H,'..','..','queen','ndx100-fill-20261114-v1.1')
exec(open(os.path.join(H,'verify_queen_ndx.py')).read().split("P=lambda x:")[0].replace("Q=os.path.join(H,'..','..','queen','ndx100-fill-20261114')","Q=V"))
g={'H':os.path.join(H,'..','ndx-top3-post-check-20261114'),'__file__':os.path.join(H,'..','ndx-top3-post-check-20261114','sp500_analog.py')}
exec(open(g['__file__']).read().split("for dec,(P,first)")[0],g)
E=g['E2'](g['PACKS']['2015-2025'][0],'2014-12-31',True,()); i0=IDX['2014-12-31']
R=Run(); lg=[]; e,c,m=R.path('M6',3,1,i0,120,lg)
sp,_=E.tranche('M6avg',3,1,0); print(f'S&P 500 universe Top3: {(sp[-1]**0.1-1)*100:.1f}%/yr | Nasdaq-100 universe Top3: {c*100:.1f}%/yr')
both=0; ndx_in_sp=0; same3=0; contrib=collections.Counter(); cn=collections.Counter(); mem_ov=[]
for j,(a,b,pr,gm) in enumerate(lg):
    s=set(E.ranking(E.i0+j,'M6avg')[:3]); spu=set(E.el[a]); n=set(pr)
    both+=len(s&n); same3+=(s==n); mem_ov.append(len(set(MEM[a])&spu)/len(MEM[a]))
    for t,r in pr.items():
        k='also in S&P 500 that month' if t in spu else 'Nasdaq-100 only'; contrib[k]+=(r or 0)/3; cn[k]+=1
        if t not in spu: contrib['  '+t]+=(r or 0)/3; cn['  '+t]+=1
print(f'Nasdaq-100 members that are also S&P 500 eligible (avg per month): {st.mean(mem_ov)*100:.0f}%')
print(f'Top3 picks shared between the two universes: {both}/360 = {both/360*100:.0f}% | months with identical Top3: {same3}/120')
for k in ('also in S&P 500 that month','Nasdaq-100 only'): print(f'Nasdaq Top3 holdings {k}: {cn[k]}/360 slots, sum of weight x monthly return = {contrib[k]:+.2f}')
print('Nasdaq-only picks:',sorted(((k.strip(),cn[k],round(v,2)) for k,v in contrib.items() if k.startswith('  ')),key=lambda x:-x[2]))
