#!/usr/bin/env python3
"""Claude: the social-media "Nasdaq-100 momentum" rule (rank by the MEAN of the last 6 monthly returns, hold Top3 equal weight,
rebalance monthly) run on the S&P 500 packs we already have. NOT a Nasdaq-100 test: we have no point-in-time Nasdaq-100 membership.
Reuses the month-end proxy engine from ../next-steps-20261114/breadth_study.py (25 bp one-way, price only).
Usage: python3 sp500_analog.py"""
import os,statistics as st
H=os.path.dirname(os.path.abspath(__file__)); src=open(os.path.join(H,'..','next-steps-20261114','breadth_study.py')).read().split("rows=[]; yearly=[]")[0]
src=src.replace("HERE=os.path.dirname(os.path.abspath(__file__))","HERE=os.path.join(H,'..','next-steps-20261114')"); exec(src)
class E2(Era):
    def ranking(s,i,sig):
        if sig!='M6avg': return Era.ranking(s,i,sig)
        if (i,sig) in s.cache: return s.cache[(i,sig)]
        out=[]
        for t in s.el[s.ME[i]]:
            if t in s.excl: continue
            r=[s.ret(t,s.ME[j-1],s.ME[j]) for j in range(i-5,i+1)]
            if None not in r: out.append((-sum(r)/6,t))
        out=[t for _,t in sorted(out)]; s.cache[(i,sig)]=out; return out
for dec,(P,first) in PACKS.items():
    for var,ex in ((('all names',()),('no NVDA',('NVDA',))) if dec=='2015-2025' else (('all names',()),)):
        E=E2(P,first,dec=='2015-2025',ex)
        if 'APC' in E.px: E.px['APC']={d:v for d,v in E.px['APC'].items() if d<='2019-07-31'}  # pack shows APC 73.66 -> 12.76 in Aug 2019 (acquired by OXY for ~72.77 USD/sh): bad rank price, treat as cash-out at last good price
        b,_=E.tranche('ALL',None,12,0); bc,bm,by=stats(b)
        print(f'{dec} {var}: EW universe {bc*100:.1f}% / MDD {bm*100:.0f}%')
        for K in (1,3,5,10):
            p,to=E.tranche('M6avg',K,1,0); c,m,y=stats(p); a,t,d=active_t(p,b)
            print(f'   Top{K:<2d} monthly: CAGR {c*100:5.1f}% / MDD {m*100:4.0f}% / worst yr {min(y)*100:4.0f}% best {max(y)*100:4.0f}% | vs EW {a*100:+5.1f} pp/yr t={t:+.2f} | best month {max(p[j+1]/p[j]-1 for j in range(120))*100:+.0f}% worst {min(p[j+1]/p[j]-1 for j in range(120))*100:+.0f}% | turnover {to*1200:.0f}%/yr')
            if K==3: print('      yearly:',' '.join('%+.0f%%'%(x*100) for x in y))
