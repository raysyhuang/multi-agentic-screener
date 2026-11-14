#!/usr/bin/env python3
"""Claude: can simple overlays cut the drawdown of the Nasdaq-100 Top3 momentum rule without giving up the return?
Base: post rule (mean of 6 monthly returns, Top3, monthly, 25 bp, price only), Queen v1.1 prices, my engine and membership.
Overlays act on the monthly return series with an exposure x in [0,1] decided at the signal month-end (no look-ahead); cash earns 0;
changing exposure costs 25 bp of the change (approximation).  QQQ month-end closes from Queen's raw Massive daily files (QQQQ before 2011-03-22).
Usage: python3 dd_overlays.py"""
import os,csv,datetime as dt,statistics as st,math
HERE=os.path.dirname(os.path.abspath(__file__)); H=HERE; RV=os.path.join(H,'..','queen-ndx100-fill-review-20261114'); V=os.path.join(H,'..','..','queen','ndx100-fill-20261114-v1.1')
exec(open(os.path.join(RV,'verify_queen_ndx.py')).read().split("P=lambda x:")[0].replace("H=os.path.dirname(os.path.abspath(__file__))","H=RV").replace("Q=os.path.join(H,'..','..','queen','ndx100-fill-20261114')","Q=V"))
def daily(f):
    o={}
    for r in csv.DictReader(open(os.path.join(V,'raw','adj',f))): o[(dt.datetime.fromtimestamp(int(r['t'])/1000,dt.timezone.utc)-dt.timedelta(hours=4)).date().isoformat()]=float(r['c'])
    return o
qd=daily('QQQQ.csv'); qd.update(daily('QQQ.csv')); ds=sorted(qd)
import bisect
qm=[qd[ds[bisect.bisect_right(ds,d)-1]] for d in ME]   # QQQ close at each month-end
R=Run()
def series(K,i0,n):
    lg=[]; R.path('M6',K,1,i0,n,lg); return [g for a,b,pr,g in lg]
def run(r,x,i0):
    eq=[1.0]; px_=0.0
    for j,(rt,xt) in enumerate(zip(r,x)):
        eq.append(eq[-1]*(1+xt*rt-0.0025*abs(xt-px_)*(j>0))); px_=xt
    pk=0;mdd=0
    for v in eq: pk=max(pk,v); mdd=min(mdd,v/pk-1)
    yrs=(dt.date.fromisoformat(ME[i0+len(r)])-dt.date.fromisoformat(ME[i0])).days/365.25
    return eq[-1]**(1/yrs)-1,mdd,min(a*b for a,b in zip(r,x)),st.mean(x)
rows=[]
for lab,i0,n in (('2007-2014',IDX['2006-12-29'],96),('2015-2026',IDX['2014-12-31'],len(ME)-1-IDX['2014-12-31'])):
    r3=series(3,i0,n); r10=series(10,i0,n); idx=range(i0,i0+n)
    q=[qm[m+1]/qm[m]-1 for m in idx]
    sma=[1.0 if (m>=9 and qm[m]>st.mean(qm[m-9:m+1])) else 0.0 for m in idx]
    mom=[1.0 if (m>=12 and qm[m]>qm[m-12]) else 0.0 for m in idx]
    def vt(r,target):
        x=[]
        for j in range(len(r)):
            h=r[max(0,j-6):j]; v=st.pstdev(h)*math.sqrt(12) if len(h)>=3 else target; x.append(min(1.0,target/v) if v>0 else 1.0)
        return x
    def ddstop(r,lim):   # step aside the month after the strategy's own curve is more than `lim` below its high; re-enter when the shadow curve makes a 3-month high
        x=[];sh=[1.0];out=False
        for j in range(len(r)):
            pk=max(sh)
            if not out and sh[-1]/pk-1<-lim: out=True
            elif out and len(sh)>3 and sh[-1]>=max(sh[-4:]): out=False
            x.append(0.0 if out else 1.0); sh.append(sh[-1]*(1+r[j]))
        return x
    one=[1.0]*n
    tests=[('QQQ 买入持有',q,one),('Top3 原规则',r3,one),('Top3 + QQQ 低于10月均线就空仓',r3,sma),('Top3 + QQQ 12个月收益为负就空仓',r3,mom),
           ('Top3 半仓 (50% 现金)',r3,[0.5]*n),('Top3 一半 + QQQ 一半',[0.5*a+0.5*b for a,b in zip(r3,q)],one),('Top3 波动目标 30%',r3,vt(r3,0.30)),('Top3 自身回撤超25%就退出',r3,ddstop(r3,0.25)),
           ('Top10 原规则',r10,one),('Top10 + QQQ 低于10月均线就空仓',r10,sma),('QQQ + 低于10月均线就空仓',q,sma)]
    print(f'== {lab} ({n} 个月)   年化 / 月末最大回撤 / 最差月 / 平均仓位 / 年化÷回撤')
    for name,r,x in tests:
        c,m,w,ax=run(r,x,i0); print(f'  {name:34s} {c*100:6.1f}% / {m*100:5.0f}% / {w*100:5.0f}% / {ax*100:4.0f}% / {c/abs(m):.2f}'); rows.append(dict(period=lab,test=name,CAGR=round(c,4),monthend_MDD=round(m,4),worst_month=round(w,4),avg_exposure=round(ax,3)))
w=csv.DictWriter(open(os.path.join(HERE,'dd_overlays_results.csv'),'w',newline=''),fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
