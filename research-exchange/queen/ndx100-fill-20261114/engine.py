"""Queen engine for the post rule on Nasdaq-100 PIT members (Massive split-adjusted closes, price only)."""
from lib import *
import math,statistics as st
DAYS,ME_ALL=calendar()
PX=Px()
ME=[d for d in ME_ALL if d>='2006-06']
IDX={d:i for i,d in enumerate(ME)}
MEM=membership([d for d in ME if d>='2006-12'])
MASK=None  # optional dict month_end -> set of tickers allowed (e.g. Claude pack availability)
ALIASM={'BKNG':'PCLN'}
def members(d,exclude=()):
    return [(t,series_key(t,d)) for t in MEM[d] if t not in exclude and (MASK is None or t in MASK.get(d,()) or ALIASM.get(t) in MASK.get(d,()))]
def mom_signal(key,i,W=6,kind='mean'):
    if i-W<0: return None
    pts=[PX.at(key,ME[j]) for j in range(i-W,i+1)]
    if any(p is None for p in pts): return None
    if key in PX.k and len({p[2] for p in pts})>1: return None
    if kind=='mean': return sum(pts[j+1][0]/pts[j][0]-1 for j in range(W))/W
    return pts[-1][0]/pts[0][0]-1   # total W-month return
_SIG={}
def ranking(i,exclude=(),W=6,kind='mean'):
    ck=(i,tuple(exclude),W,kind,id(MASK))
    if ck in _SIG: return _SIG[ck]
    d=ME[i]; sc=[]
    for t,key in members(d,exclude):
        s=mom_signal(key,i,W,kind)
        if s is not None: sc.append((s,t,key))
    sc.sort(key=lambda x:(-x[0],x[1])); _SIG[ck]=sc; return sc
def days_between(a,b):
    lo=bisect.bisect_right(DAYS,a); hi=bisect.bisect_right(DAYS,b); return DAYS[lo:hi]
def run(K,i0,i1,exclude=(),cost=0.0025,W=6,kind='mean',daily=True):
    """signals at ME[i0..i1-1], holding to next month-end. returns dict"""
    nav=1.0; navs=[(ME[i0],1.0)]; dnav=[1.0]; rec=[]; prev={}  # prev drifted weights
    contrib={}; held={}
    for i in range(i0,i1):
        rk=ranking(i,exclude,W,kind); picks=rk[:K]
        tgt={p[1]:1.0/len(picks) for p in picks} if picks else {}
        traded=sum(abs(tgt.get(t,0)-prev.get(t,0)) for t in set(tgt)|set(prev))
        c=cost*traded
        rets={}
        for s,t,key in picks:
            r=PX.hold_ret(key,ME[i],ME[i+1]); rets[t]=r[0] if r else 0.0
        R=sum(tgt[t]*rets[t] for t in tgt)
        net=(1-c)*(1+R)-1
        if daily:
            dd=days_between(ME[i],ME[i+1]); 
            paths={t:PX.daily_path(key,ME[i],ME[i+1],dd) for s,t,key in picks}
            for j in range(len(dd)):
                v=sum(tgt[t]*paths[t][j] for t in tgt) if tgt else 1.0
                dnav.append(nav*(1-c)*v)
        nav*=1+net; navs.append((ME[i+1],nav))
        if daily: dnav[-1]=nav
        prev={t:tgt[t]*(1+rets[t])/(1+R) for t in tgt} if R>-1 else {}
        for t in tgt:
            contrib[t]=contrib.get(t,0)+tgt[t]*rets[t]; held[t]=held.get(t,0)+1
        rec.append(dict(signal=ME[i],end=ME[i+1],picks=' '.join(t for s,t,k in picks),
                        scores=' '.join('%.4f'%s for s,t,k in picks),
                        stock_rets=' '.join('%+.4f'%rets[t] for s,t,k in picks),gross=R,cost=c,net=net,traded=traded,n_ranked=len(rk)))
    return dict(navs=navs,dnav=dnav,rec=rec,contrib=contrib,held=held)
def stats(res):
    navs=res['navs']; n=len(navs)-1; yrs=n/12
    cagr=navs[-1][1]**(1/yrs)-1
    def mdd(seq):
        pk=-1e9; m=0
        for v in seq: pk=max(pk,v); m=min(m,v/pk-1)
        return m
    mr=[r['net'] for r in res['rec']]
    return dict(months=n,CAGR=cagr,MDD_monthly=mdd([v for d,v in navs]),MDD_daily=mdd(res['dnav']) if len(res['dnav'])>1 else None,
                worst_month=min(mr),worst_month_date=res['rec'][mr.index(min(mr))]['end'],best_month=max(mr),
                turnover_oneway_per_year=sum(r['traded'] for r in res['rec'])/2/yrs, total=navs[-1][1])
def ew(i0,i1,exclude=()):
    nav=1.0; navs=[(ME[i0],1.0)]; dnav=[1.0]; rec=[]
    for i in range(i0,i1):
        d=ME[i]; ks=[(t,k) for t,k in members(d,exclude) if PX.at(k,d)]
        rs=[]; paths=[]
        dd=days_between(ME[i],ME[i+1])
        for t,k in ks:
            r=PX.hold_ret(k,d,ME[i+1])
            if r is None: continue
            rs.append(r[0]); paths.append(PX.daily_path(k,d,ME[i+1],dd))
        R=sum(rs)/len(rs)
        for j in range(len(dd)): dnav.append(nav*sum(p[j] for p in paths)/len(paths))
        nav*=1+R; navs.append((ME[i+1],nav)); dnav[-1]=nav
        rec.append(dict(signal=d,end=ME[i+1],net=R,traded=0,n=len(rs),n_members=len(MEM[d])))
    return dict(navs=navs,dnav=dnav,rec=rec)
# QQQ
def qqq():
    a=load_daily(os.path.join(H,'raw/adj/QQQQ.csv')); b=load_daily(os.path.join(H,'raw/adj/QQQ.csv'))
    pa=[k for k in sorted(a) if k<'2011-03-23']; k0=min(b); f=b[k0]/a[pa[-1]]
    # junction QQQQ->QQQ is same fund (ticker change 2011-03-23); scale by ratio (junction-day return assumed 0)
    p={x:a[x]*f for x in pa}; p.update(b)
    div={}
    for r in csv.DictReader(open(os.path.join(H,'raw/qqq_dividends_massive.csv'))):
        div[r['ex_dividend_date']]=div.get(r['ex_dividend_date'],0)+float(r['cash_amount'])*(f if r['ex_dividend_date']<'2011-03-23' else 1)
    ks=sorted(p); tr={ks[0]:1.0}
    for x,y in zip(ks,ks[1:]): tr[y]=tr[x]*(p[y]+div.get(y,0))/p[x]
    return p,tr
def bench(series,i0,i1):
    navs=[(ME[i0],1.0)]; base=series[ME[i0]]
    for i in range(i0,i1): navs.append((ME[i+1],series[ME[i+1]]/base))
    ks=[k for k in sorted(series) if ME[i0]<=k<=ME[i1]]; dn=[series[k]/base for k in ks]
    rec=[dict(net=navs[j+1][1]/navs[j][1]-1,end=navs[j+1][0],traded=0) for j in range(len(navs)-1)]
    return dict(navs=navs,dnav=dn,rec=rec)
