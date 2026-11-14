#!/usr/bin/env python3
"""Queen independent check of Claude 0f146b0 drawdown overlays (PROVISIONAL, research only).
Runs from inside /workspace/ndx100-fill-20261114-v1.1 (read-only use of engine.py); writes only to this script's folder.
Overlays replicate Claude dd_overlays.py: exposure x decided at signal month-end, cash 0%, extra cost 25bp*|dx| (not on first month),
SMA10: QQQ ME close > mean of last 10 ME closes (incl. current); MOM12: QQQ ME close > close 12 ME ago;
vol-target 30%: x=min(1,0.30/(pstdev(last<=6 in-period monthly rule returns)*sqrt12)), x=1 until 3 obs;
DD-stop 25%: out the month after shadow curve < -25% from its peak; back in when shadow curve at 3-month high.
Monthly series = Queen engine run() net returns (Top3/Top10, 25bp). Daily MDD from engine daily path scaled by exposure.
CAGR reported two ways: calendar days/365.25 (Claude) and months/12 (Queen engine stats)."""
import os,sys,csv,math,statistics as st,datetime as dt,bisect
OUT=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.getcwd())
from engine import *
P,TR=qqq(); qd=sorted(P)
def qat(d): return P[qd[bisect.bisect_right(qd,d)-1]]
def months_paths(res,i0,i1):
    """per month: list of daily relative values g_j (end = 1+net)"""
    out=[]; dn=res['dnav']; k=1; navs=res['navs']
    for m,i in enumerate(range(i0,i1)):
        nd=len(days_between(ME[i],ME[i+1])); base=navs[m][1]
        out.append([dn[k+j]/base for j in range(nd)]); k+=nd
    return out
def qpaths(i0,i1):
    return [[qat(d)/qat(ME[i]) for d in days_between(ME[i],ME[i+1])] for i in range(i0,i1)]
def mdd(seq):
    pk=-1e9;m=0
    for v in seq: pk=max(pk,v); m=min(m,v/pk-1)
    return m
def evaluate(gp,x,i0,n):
    nav=1.0; me=[1.0]; dl=[1.0]; px=0.0
    for j,(g,xt) in enumerate(zip(gp,x)):
        c=0.0025*abs(xt-px)*(j>0); px=xt
        for v in g: dl.append(nav*(1-c)*(1+xt*(v-1)))
        nav=nav*(1-c)*(1+xt*(g[-1]-1)) if False else nav*(1+xt*(g[-1]-1)-c)   # Claude form: 1+x*r-cost
        dl[-1]=nav; me.append(nav)
    yrs=(dt.date.fromisoformat(ME[i0+n])-dt.date.fromisoformat(ME[i0])).days/365.25
    return dict(CAGR_cal=nav**(1/yrs)-1,CAGR_n12=nav**(12/n)-1,MDD_monthend=mdd(me),MDD_daily=mdd(dl),avg_exposure=st.mean(x),total=nav)
claude={('2007-2014','QQQ_buyhold'):(0.1151,-0.4997),('2007-2014','Top3_base'):(0.0913,-0.4203),('2007-2014','Top3_SMA10'):(0.0617,-0.3807),('2007-2014','Top3_MOM12'):(0.0576,-0.4079),
('2007-2014','Top3_half_cash'):(0.0526,-0.2299),('2007-2014','Top3_half_QQQ'):(0.1095,-0.4442),('2007-2014','Top3_vol30'):(0.086,-0.4162),('2007-2014','Top3_ddstop25'):(0.0094,-0.4889),
('2007-2014','Top10_base'):(0.1053,-0.6152),('2007-2014','Top10_SMA10'):(0.1188,-0.3628),('2007-2014','QQQ_SMA10'):(0.0789,-0.2663),
('2015-2026','QQQ_buyhold'):(0.1721,-0.3307),('2015-2026','Top3_base'):(0.383,-0.4566),('2015-2026','Top3_SMA10'):(0.2085,-0.4566),('2015-2026','Top3_MOM12'):(0.3311,-0.4566),
('2015-2026','Top3_half_cash'):(0.2008,-0.2419),('2015-2026','Top3_half_QQQ'):(0.29,-0.376),('2015-2026','Top3_vol30'):(0.3462,-0.4206),('2015-2026','Top3_ddstop25'):(0.336,-0.3706),
('2015-2026','Top10_base'):(0.2057,-0.4088),('2015-2026','Top10_SMA10'):(0.1245,-0.4088),('2015-2026','QQQ_SMA10'):(0.118,-0.2063)}
rows=[]; expo=[]
for lab,a,b in (('2007-2014','2006-12-29','2014-12-31'),('2015-2026','2014-12-31','2026-11-13')):
    i0,i1=IDX[a],IDX[b]; n=i1-i0
    r3s=run(3,i0,i1); r10s=run(10,i0,i1)
    r3=[r['net'] for r in r3s['rec']]; g3=months_paths(r3s,i0,i1); g10=months_paths(r10s,i0,i1); gq=qpaths(i0,i1)
    for g,r in ((g3,r3),):
        assert all(abs(gg[-1]-1-rr)<1e-9 for gg,rr in zip(g,r))
    qm=[qat(ME[m]) for m in range(len(ME))]
    sma=[1.0 if (m>=9 and qm[m]>st.mean(qm[m-9:m+1])) else 0.0 for m in range(i0,i1)]
    mom=[1.0 if (m>=12 and qm[m]>qm[m-12]) else 0.0 for m in range(i0,i1)]
    def vt(r,t):
        x=[]
        for j in range(len(r)):
            h=r[max(0,j-6):j]; v=st.pstdev(h)*math.sqrt(12) if len(h)>=3 else t; x.append(min(1.0,t/v) if v>0 else 1.0)
        return x
    def ddstop(r,lim):
        x=[];sh=[1.0];out=False
        for j in range(len(r)):
            pk=max(sh)
            if not out and sh[-1]/pk-1<-lim: out=True
            elif out and len(sh)>3 and sh[-1]>=max(sh[-4:]): out=False
            x.append(0.0 if out else 1.0); sh.append(sh[-1]*(1+r[j]))
        return x
    one=[1.0]*n
    blend=[[0.5*u+0.5*v for u,v in zip(a_,b_)] for a_,b_ in zip(g3,gq)]
    tests=[('QQQ_buyhold',gq,one),('Top3_base',g3,one),('Top3_SMA10',g3,sma),('Top3_MOM12',g3,mom),('Top3_half_cash',g3,[0.5]*n),
           ('Top3_half_QQQ',blend,one),('Top3_vol30',g3,vt(r3,0.30)),('Top3_ddstop25',g3,ddstop(r3,0.25)),('Top10_base',g10,one),('Top10_SMA10',g10,sma),('QQQ_SMA10',gq,sma)]
    for name,g,x in tests:
        e=evaluate(g,x,i0,n); cc,cm=claude[(lab,name)]
        d=dict(period=lab,test=name,months=n,**{k:round(v,4) for k,v in e.items()},claude_CAGR=cc,claude_MDD_monthend=cm,
               diff_CAGR_pp=round((e['CAGR_cal']-cc)*100,2),diff_MDD_pp=round((e['MDD_monthend']-cm)*100,2))
        d['flag']='MISMATCH' if abs(d['diff_CAGR_pp'])>1 or abs(d['diff_MDD_pp'])>3 else 'ok'
        rows.append(d); print('%-9s %-15s Q cal %6.2f%% n12 %6.2f%% MDDme %6.1f%% MDDd %6.1f%% | C %6.2f%% %6.1f%% | d %+5.2f %+5.2f %s'%(lab,name,e['CAGR_cal']*100,e['CAGR_n12']*100,e['MDD_monthend']*100,e['MDD_daily']*100,cc*100,cm*100,d['diff_CAGR_pp'],d['diff_MDD_pp'],d['flag']))
        if name in('Top3_SMA10','Top3_MOM12','Top3_vol30','Top3_ddstop25'):
            for i,xx in zip(range(i0,i1),x): expo.append(dict(period=lab,test=name,signal=ME[i],exposure=round(xx,4)))
    # monthly base series dump
    for rr in r3s['rec']: expo.append(dict(period=lab,test='Top3_base_net',signal=rr['signal'],exposure=round(rr['net'],6)))
with open(os.path.join(OUT,'overlay_check_results.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open(os.path.join(OUT,'overlay_exposures_and_base_returns.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(expo[0])); w.writeheader(); w.writerows(expo)
