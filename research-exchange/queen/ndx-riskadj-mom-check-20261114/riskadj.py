"""Queen replication of momentum-lab.net 'ndx-grjmom' (risk-adjusted 12-1 momentum) on Queen's verified NDX100 PIT engine.
Imports /workspace/ndx100-fill-20261114-v1.1 (engine.py, lib.py) READ-ONLY. Run with system python3."""
import sys,os,math,csv,bisect,json
sys.dont_write_bytecode=True
ENG='/workspace/ndx100-fill-20261114-v1.1'
sys.path.insert(0,ENG); os.chdir(ENG)
import engine as E, lib as L
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '/workspace/ndx-riskadj-mom-check-20261114')
OUT='/workspace/ndx-riskadj-mom-check-20261114/out'
PX,ME,IDX,DAYS=E.PX,E.ME,E.IDX,E.DAYS
DPOS={d:i for i,d in enumerate(DAYS)}
# ---------- open/close ratio per (key,date), stitched exactly like lib.build_series (ratios are scale-free) ----------
def load_oc(path):
    out={}
    for r in csv.DictReader(open(path)):
        if not r.get('t'): continue
        o,c=float(r['o']),float(r['c'])
        if c>0 and o>0: out[L.d_of(r['t']).isoformat()]=o/c
    return out
import glob
OC={os.path.basename(f)[:-4]:load_oc(f) for f in glob.glob(os.path.join(ENG,'raw/adj/*.csv'))}
for p,s,d in L.RENAMES:
    P=OC.get(p) or {}; S=OC.get(s) or {}
    pb=[x for x in sorted(P) if x<d]; sa=[x for x in sorted(S) if x>=d]
    if not pb or not sa: continue
    new={x:P[x] for x in pb}; new.update({x:S[x] for x in sa}); OC[s]=new
    OC[p]={x:P[x] for x in pb}; OC[p].update({x:S[x] for x in sa})
FALLBACK={'open_missing_used_prev_close':0,'monthly_only_close_exec':0,'vol_from_monthly':0}
def adj_open(key,d,seg):
    if key not in PX.k: return None
    c=PX.D[key].get(d)
    if c is None or PX.seg[key][d]!=seg: return None
    r=OC.get(key,{}).get(d)
    return c*r if r else None
# ---------- signals ----------
_dr={}
def daily_rets(key,d0,d1,seg):
    ks=PX.k[key]; a=bisect.bisect_right(ks,d0)-1; b=bisect.bisect_right(ks,d1)-1
    xs=[ks[j] for j in range(max(a,0),b+1) if PX.seg[key][ks[j]]==seg]
    v=np.array([PX.D[key][x] for x in xs]); return v[1:]/v[:-1]-1
def signal(key,i,L_back,skip,N):
    """L_back-skip momentum: return from ME[i-L_back] to ME[i-skip]; vol = RMS daily ret on same window *sqrt(252)."""
    if i-L_back<0: return None
    pts=[PX.at(key,ME[j]) for j in range(i-L_back,i+1)]
    if any(p is None for p in pts): return None
    if key in PX.k and len({p[2] for p in pts})>1: return None
    ret=pts[L_back-skip][0]/pts[0][0]-1
    if N==0: return ret,ret,None
    ck=(key,i,L_back,skip)
    if ck not in _dr:
        if key in PX.k:
            r=daily_rets(key,ME[i-L_back],ME[i-skip],pts[0][2]); vol=math.sqrt(np.mean(r*r))*math.sqrt(252) if len(r)>20 else None
        else:  # month-only series: RMS of monthly returns * sqrt(12)
            m=np.array([pts[j+1][0]/pts[j][0]-1 for j in range(L_back-skip)]); vol=math.sqrt(np.mean(m*m))*math.sqrt(12); FALLBACK['vol_from_monthly']+=1
        _dr[ck]=vol
    vol=_dr[ck]
    if not vol: return None
    return ret/vol**N,ret,vol
def mean6(key,i):
    s=E.mom_signal(key,i,6,'mean'); return (s,s,None) if s is not None else None
_RK={}
def ranking(i,spec,exclude=()):
    ck=(i,spec,exclude)
    if ck in _RK: return _RK[ck]
    sc=[]
    for t,key in E.members(ME[i],exclude):
        s=mean6(key,i) if spec[0]=='mean6' else signal(key,i,spec[1],spec[2],spec[3])
        if s is not None: sc.append((s[0],t,key,s[1],s[2]))
    sc.sort(key=lambda x:(-x[0],x[1])); _RK[ck]=sc; return sc
# ---------- execution ----------
def exec_day(i,lag):
    """lag=0: month-end close; lag>=1: open of the lag-th trading day after ME[i]. None if beyond data."""
    if lag==0: return ME[i]
    p=DPOS[ME[i]]+lag
    return DAYS[p] if p<len(DAYS) else None
def px_exec(key,d,seg,lag,is_entry):
    """price for trading at day d: open if lag>=1 else close. Falls back to last segment close on/before d."""
    if lag>=1:
        o=adj_open(key,d,seg)
        if o is not None: return o,d,'open'
    ks=PX.k[key]; j=bisect.bisect_right(ks,d)-1
    while j>=0 and PX.seg[key][ks[j]]!=seg: j-=1
    if j<0: return None
    if lag>=1: FALLBACK['open_missing_used_prev_close']+=1
    return PX.D[key][ks[j]],ks[j],'close'
def run(spec,K,i0,i1,cost,lag=1,exclude=()):
    nav=1.0; dnav=[]; ddates=[]; navs=[]; rec=[]; prev={}
    for i in range(i0,i1):
        ent=exec_day(i,lag); ext=exec_day(i+1,lag)
        if ext is None: ext=ME[i+1]; ext_close=True   # last period: no next open in data -> exit at last close
        else: ext_close=(lag==0)
        if i==i0: navs.append((ent,1.0)); dnav.append(1.0); ddates.append(ent)
        rk=ranking(i,spec,exclude); picks=rk[:K]
        tgt={p[1]:1.0/len(picks) for p in picks}
        traded=sum(abs(tgt.get(t,0)-prev.get(t,0)) for t in set(tgt)|set(prev)); c=cost*traded
        lo=DPOS[ent]+ (1 if lag==0 else 0); hi=DPOS[ext]
        hold_days=DAYS[lo:hi]  # closes observed during holding (for lag>=1 includes entry day close)
        rets={}; paths={}
        for s,t,key,_,_ in picks:
            if key not in PX.k:  # month-only symbol: month-end close to month-end close
                FALLBACK['monthly_only_close_exec']+=1
                r=PX.hold_ret(key,ME[i],ME[i+1]); rets[t]=r[0] if r else 0.0
                paths[t]=[1.0]*len(hold_days)+[1+rets[t]]; continue
            seg=PX.at(key,ME[i])[2]
            e=px_exec(key,ent,seg,lag,True)
            x=px_exec(key,ext,seg,0 if ext_close else lag,False)
            rets[t]=x[0]/e[0]-1
            last=1.0; pth=[]
            for d in hold_days:
                v=PX.D[key].get(d)
                if v is not None and PX.seg[key][d]==seg and d<=x[1]: last=v/e[0]
                pth.append(last)
            pth.append(1+rets[t]); paths[t]=pth
        R=sum(tgt[t]*rets[t] for t in tgt); net=(1-c)*(1+R)-1
        pts=hold_days+[ext+('C' if ext_close else 'O')]
        for j in range(len(pts)):
            v=sum(tgt[t]*paths[t][j] for t in tgt); dnav.append(nav*(1-c)*v); ddates.append(pts[j])
        nav*=1+net; dnav[-1]=nav; navs.append((ext,nav))
        prev={t:tgt[t]*(1+rets[t])/(1+R) for t in tgt} if R>-1 else {}
        rec.append(dict(signal=ME[i],entry=ent,exit=ext,picks=' '.join(p[1] for p in picks),
            scores=' '.join('%.4f'%p[0] for p in picks),stock_rets=' '.join('%+.4f'%rets[p[1]] for p in picks),gross=R,cost=c,net=net,traded=traded,n_ranked=len(rk)))
    return dict(navs=navs,dnav=dnav,ddates=ddates,rec=rec)
def bench(series,i0,i1):
    ks=[k for k in sorted(series) if ME[i0]<=k<=ME[i1]]; b=series[ME[i0]]
    navs=[(ME[i],series[ME[i]]/b) for i in range(i0,i1+1)]
    rec=[dict(net=navs[j+1][1]/navs[j][1]-1,traded=0) for j in range(len(navs)-1)]
    return dict(navs=navs,dnav=[series[k]/b for k in ks],ddates=ks,rec=rec)
def mdd(seq):
    pk=-1e9; m=0
    for v in seq: pk=max(pk,v); m=min(m,v/pk-1)
    return m
def ddur(dn,dates):
    """longest peak-to-recovery (calendar days) and the max-DD episode peak/trough/recovery"""
    import datetime as dt
    D=lambda s: dt.date.fromisoformat(s[:10])
    pk=dn[0]; pkd=dates[0]; best=(0,None,None); mx=(0,None,None,None); cur_tr=(0,None)
    for v,d in zip(dn,dates):
        if v>=pk:
            if pkd and D(d)>D(pkd):
                L_=(D(d)-D(pkd)).days
                if L_>best[0] and cur_tr[0]<0: best=(L_,pkd,d)
            pk=v; pkd=d; cur_tr=(0,None)
        else:
            dd=v/pk-1
            if dd<cur_tr[0]: cur_tr=(dd,d)
            if dd<mx[0]: mx=(dd,pkd,d,None)
    return best,mx
def stats(res):
    import datetime as dt
    navs=res['navs']; d0=dt.date.fromisoformat(navs[0][0][:10]); d1=dt.date.fromisoformat(navs[-1][0][:10])
    yrs=(d1-d0).days/365.25; tot=navs[-1][1]
    mr=np.array([r['net'] for r in res['rec']]); dn=np.array(res['dnav']); dr=dn[1:]/dn[:-1]-1
    return dict(start=navs[0][0],end=navs[-1][0],months=len(mr),CAGR=tot**(1/yrs)-1,CAGR_months12=tot**(12/len(mr))-1,
        vol_daily_ann=float(np.std(dr,ddof=1)*math.sqrt(252)),vol_monthly_ann=float(np.std(mr,ddof=1)*math.sqrt(12)),
        MDD_daily=mdd(dn),MDD_monthly=mdd([v for _,v in navs]),sharpe_monthly=float(mr.mean()/mr.std(ddof=1)*math.sqrt(12)),
        turnover_oneway_yr=sum(r['traded'] for r in res['rec'])/2/yrs,total=tot)
