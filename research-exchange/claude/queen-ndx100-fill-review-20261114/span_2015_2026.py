#!/usr/bin/env python3
"""Claude: post rule on the full Nasdaq-100 universe, 2014-12-31 -> 2026-11-13 (Queen v1.1 prices, my engine, 25 bp, price only).  Usage: python3 span_2015_2026.py"""
import os,datetime as dt,math
H=os.path.dirname(os.path.abspath(__file__)); V=os.path.join(H,'..','..','queen','ndx100-fill-20261114-v1.1')
exec(open(os.path.join(H,'verify_queen_ndx.py')).read().split("P=lambda x:")[0].replace("Q=os.path.join(H,'..','..','queen','ndx100-fill-20261114')","Q=V"))
i0=IDX['2014-12-31']; n=len(ME)-1-i0; yrs=(dt.date.fromisoformat(ME[-1])-dt.date(2014,12,31)).days/365.25
print('span',ME[i0],'->',ME[-1],'%.2f years'%yrs,n,'periods')
q=679.89/103.25; print('QQQ price only (FMP 103.25 -> 679.89): x%.2f, %.1f%%/yr'%(q,(q**(1/yrs)-1)*100))
for lab,run in (('all',Run()),('no NVDA',Run(excl=('NVDA',)))):
    for K in (1,3,5,10):
        lg=[]; e,c,m=run.path('M6',K,1,i0,n,lg); print(f'{lab} Top{K}: x{e[-1]:.1f}, {(e[-1]**(1/yrs)-1)*100:.1f}%/yr, month-end MDD {m*100:.0f}%')
        if K==3:
            ys={}
            for a,b,pr,g in lg: ys[b[:4]]=ys.get(b[:4],1)*(1+g)
            print('   yearly:',' '.join(f'{y}:{(v-1)*100:+.0f}%' for y,v in ys.items()))
            j=IDX['2024-12-31']-i0; print(f'   2014-12→2024-12 x{e[j]:.1f} | 2024-12→2026-11-13 x{e[-1]/e[j]:.2f} | peak→now {e[-1]/max(e)-1:+.0%} (peak {ME[i0+e.index(max(e))]})')
