#!/usr/bin/env python3
"""Claude: check of a social-media "risk-adjusted momentum" rule on the Nasdaq-100.
Rule in the post: score = (12-1 return) / (annualised RMS of daily returns over the same window)^N, N=1; hold Top3 equal weight, monthly.
Here: Queen v1.1 month-end prices for returns, Queen's raw Massive daily closes (raw/adj) for the volatility, my membership export and engine.
Differences from the post: I trade at the signal month-end close (post: next open), 25 bp one-way (post: 15 bp), price only (post: dividend adjusted).
Daily files are vendor split-adjusted only; the days of the spin-offs Queen corrected are dropped from the volatility. Month-end drawdowns.
Usage: python3 grjmom_check.py"""
import os,csv,datetime as dt,statistics as st,math,bisect,collections
HERE=os.path.dirname(os.path.abspath(__file__)); RV=os.path.join(HERE,'..','queen-ndx100-fill-review-20261114'); V=os.path.join(HERE,'..','..','queen','ndx100-fill-20261114-v1.1')
exec(open(os.path.join(RV,'verify_queen_ndx.py')).read().split("P=lambda x:")[0].replace("H=os.path.dirname(os.path.abspath(__file__))","H=RV").replace("Q=os.path.join(H,'..','..','queen','ndx100-fill-20261114')","Q=V"))
SKIP={('WDC','2025-02-24'),('EXC','2022-02-02'),('ADP','2007-04-02'),('ADP','2014-10-01'),('MAR','2011-11-22'),('HON','2016-10-03'),('HON','2018-10-29'),('BIIB','2017-02-01'),('BIIB','2017-02-02')}
PRED={v:k for k,v in SUCC.items()}; DD={}
def daily(t):
    if t not in DD:
        p=os.path.join(V,'raw','adj',t+'.csv'); d=[];r=[]
        if os.path.exists(p):
            rows=sorted((int(x['t']),float(x['c'])) for x in csv.DictReader(open(p)))
            for (t0,c0),(t1,c1) in zip(rows,rows[1:]):
                day=(dt.datetime.fromtimestamp(t1/1000,dt.timezone.utc)-dt.timedelta(hours=4)).date().isoformat()
                if (t1-t0)<12*86400000 and (t,day) not in SKIP and c0>0: d.append(day); r.append(c1/c0-1)
        DD[t]=(d,r)
    return DD[t]
def vol(t,a,b):
    best=None
    for s in (t,PRED.get(t),SUCC.get(t)):
        if not s: continue
        d,r=daily(s); x=r[bisect.bisect_right(d,a):bisect.bisect_right(d,b)]
        if best is None or len(x)>len(best): best=x
    if not best or len(best)<150: return None
    return math.sqrt(sum(v*v for v in best)/len(best)*252)
class RA(Run):
    def rank(s,i,sig):
        if not (isinstance(sig,tuple) and sig[0]=='RA'): return Run.rank(s,i,sig)
        k=(i,sig)
        if k in s.cache: return s.cache[k]
        N=sig[1]; out=[]
        for t in s.univ(i):
            pa,pb=px[t].get(ME[i-12]),px[t].get(ME[i-1])
            if pa is None or pb is None: continue
            v=vol(t,ME[i-12],ME[i-1]) if N else 1.0
            if v is None: continue
            out.append((-(pb/pa-1)/v**N,t))
        s.cache[k]=[(t,-x) for x,t in sorted(out)]; return s.cache[k]
R=RA(); R15=RA(C=0.0015)
def stats(sig,i0,n,run=R):
    lg=[]; e,c,m=run.path(sig,3,1,i0,n,lg); yrs=(dt.date.fromisoformat(ME[i0+n])-dt.date.fromisoformat(ME[i0])).days/365.25
    r=[x[3] for x in lg]; return e[-1]**(1/yrs)-1,m,st.pstdev(r)*math.sqrt(12),min(r),lg
rows=[]
SIG=[('纯涨幅 12-1 (N=0)',('RA',0)),('风险调整 N=0.5',('RA',0.5)),('风险调整 N=1 (帖子)',('RA',1)),('风险调整 N=1.5',('RA',1.5)),('风险调整 N=2',('RA',2)),('6个月均值 (上一个帖子)','M6')]
for lab,a,b in (('帖子区间 2008-01→2026-08','2008-01-31','2026-08-31'),('2008-2014','2007-12-31','2014-12-31'),('2015→2026-11-13','2014-12-31',ME[-1])):
    i0=IDX[a]; n=IDX[b]-i0; print(f'== {lab} ({n} 个月)  年化 / 月末最大回撤 / 年化波动(月) / 最差月')
    for name,sig in SIG:
        c,m,v,w,lg=stats(sig,i0,n); print(f'  {name:26s} {c*100:6.1f}% / {m*100:5.0f}% / {v*100:4.0f}% / {w*100:5.0f}%'); rows.append(dict(period=lab,rule=name,CAGR=round(c,4),monthend_MDD=round(m,4),vol=round(v,4),worst_month=round(w,4)))
    if lab.startswith('帖子'):
        c,m,v,w,lg1=stats(('RA',1),i0,n,R15); print(f'  风险调整 N=1, 单边15bp           {c*100:6.1f}% / {m*100:5.0f}%   (帖子: 36.74% / -52.91% 日口径)')
        c,m,v,w,lg0=stats(('RA',0),i0,n,R15); print(f'  纯涨幅 12-1, 单边15bp            {c*100:6.1f}% / {m*100:5.0f}%   (帖子: 31.63% / -60.85%)')
        c,m,v,w,lg6=stats('M6',i0,n,R15); print(f'  6个月均值, 单边15bp              {c*100:6.1f}% / {m*100:5.0f}%   (帖子: 35.76% / -50.94%)')
        ov=st.mean(len(set(x[2])&set(y[2]))/3 for x,y in zip(lg1,lg0)); print(f'  N=1 与纯涨幅 Top3 平均重合 {ov*100:.1f}% (帖子 65.6%)')
        cnt=collections.Counter(); [cnt.update({t:(r or 0)/3}) for x in lg1 for t,r in x[2].items()]; print('  N=1 贡献最大:',[(t,round(v*100,1)) for t,v in cnt.most_common(5)],'最差:',[(t,round(v*100,1)) for t,v in cnt.most_common()[-3:]])
        d=[x[3]-y[3] for x,y in zip(lg1,lg0)]; print(f'  N=1 减 N=0 的月收益差: 均值 {st.mean(d)*1200:+.1f}pp/年, t={st.mean(d)/st.stdev(d)*math.sqrt(len(d)):+.2f}')
i=IDX['2026-09-30']; rk=R.rank(i,('RA',1)); r0=[t for t,_ in R.rank(i,('RA',0))]
print('2026-09-30 信号 (观察期 2025-09-30→2026-08-31): N=1 前 8:',[(t,round(s,3)) for t,s in rk[:8]])
for t in ('FTNT','INTC'):
    print(f'  {t}: 累计涨幅 {(px[t][ME[i-1]]/px[t][ME[i-12]]-1)*100:.2f}% 年化波动 {vol(t,ME[i-12],ME[i-1])*100:.2f}% 原始排名 {r0.index(t)+1} 调整后排名 {[x for x,_ in rk].index(t)+1}   (帖子: FTNT 103.29/42.42/12/7, INTC 166.80/77.99/7/8)')
w=csv.DictWriter(open(os.path.join(HERE,'grjmom_results.csv'),'w',newline=''),fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
