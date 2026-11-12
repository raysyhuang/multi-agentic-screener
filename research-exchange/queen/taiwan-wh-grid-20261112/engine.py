# Engine for SPEC.md (sha256 in SPEC.sha256). Loader = lines 1-151 of the prior grid_tw150.py (read-only reuse).
import os, json, datetime as dt
import numpy as np, pandas as pd
OUT='/workspace/taiwan-wh-grid-20261112'
SRC='/workspace/taiwan-winner-check-2026-10-21/short_mom/grid_tw150.py'
_code=''.join(open(SRC).readlines()[:151])
exec(_code)            # defines load, PX, EV, factor, seasoned_in, M50/M100/M150 ... ; chdir to old dir (read only)
FIRST_SNAP={'TW50':'2003-01-20','TW150':'2003-01-20','MC100':'2004-12-20'}
UNIV={'TW50':(M50,EK50),'TW150':(M150,EK150),'MC100':(M100,EK100)}
load('2330'); load('0050')
CAL=list(PX['2330'].index)
LAST=max(CAL)
ME={}
for d in CAL: ME[d[:7]]=d          # last trading day per month
MONTHS=sorted(k for k in ME if k<='2026-09')
def mkey(y,m): return f'{y}-{m:02d}'
def madd(k,n):
    y,m=int(k[:4]),int(k[5:]); t=y*12+m-1+n; return mkey(t//12,t%12+1)
ALLT=sorted(set().union(*[set().union(*M.values()) for M,_ in UNIV.values()])|{'0050'})
MISSING=[]
for t in ALLT:
    load(t)
    if PX[t] is None: MISSING.append(t)
# master calendar = union of all trading dates
MD=sorted(set().union(*[set(PX[t].index) for t in ALLT if PX[t] is not None]))
MDi={d:i for i,d in enumerate(MD)}
MDa=np.array(MD)
nD=len(MD); TI={t:j for j,t in enumerate(ALLT)}; nT=len(ALLT)
def cf_on_master(t,col):
    e=EV[t]
    if e is None or len(e)==0: return np.ones(nD)
    e=e.sort_values('date'); pos=np.searchsorted(MDa,e.date.values,side='left')  # event date<=d  => d index>=pos
    f=np.ones(nD+1)
    for p,v in zip(pos,e[col].values): f[p]*=v
    return np.cumprod(f)[:nD]
def build():
    A={}
    for col in ('fp','ftr'):
        V=np.full((nD,nT),np.nan); BO=np.full((nD,nT),np.nan)
        for t in ALLT:
            if PX[t] is None: continue
            j=TI[t]; cf=cf_on_master(t,col); idx=np.array([MDi[d] for d in PX[t].index])
            V[idx,j]=PX[t].close.values*cf[idx]; BO[idx,j]=PX[t].open.values*cf[idx]
        A[col]=(pd.DataFrame(V).ffill().values,BO)
    C=np.full((nD,nT),np.nan); LD=np.full((nD,nT),np.nan); TR=np.zeros((nD,nT),bool)
    for t in ALLT:
        if PX[t] is None: continue
        j=TI[t]; idx=np.array([MDi[d] for d in PX[t].index]); C[idx,j]=PX[t].close.values; LD[idx,j]=idx; TR[idx,j]=True
    C=pd.DataFrame(C).ffill().values; LD=pd.DataFrame(LD).ffill().values
    # next trade index strictly after row i : NX[i,j]
    NX=np.full((nD+1,nT),np.nan)
    for i in range(nD-1,-1,-1):
        NX[i]=np.where(TR[i],i,NX[i+1])
    return A,C,LD,NX[:nD]
A,CLOSE,LASTD,NXT=build()
CFP=np.column_stack([cf_on_master(t,'fp') for t in ALLT])
MDdate=np.array([dt.date.fromisoformat(d).toordinal() for d in MD])
def elig(univ,s):
    M,EK=UNIV[univ]; return sorted(seasoned_in(M,EK,s))
def feasible(univ,mk,W):
    s=ME[mk]; d0=(dt.date.fromisoformat(s)-dt.timedelta(365)).isoformat()
    return d0>=FIRST_SNAP[univ] and madd(mk,-W)>='2003-01'
RC={}; DROPS=[]
def rank(univ,mk,W):
    k=(univ,mk,W)
    if k in RC: return RC[k]
    s=ME[mk]; b=ME[madd(mk,-W)]; i1=MDi[s]; i0=MDi[b]; r=[]
    for t in elig(univ,s):
        if PX[t] is None: DROPS.append((univ,W,s,t,'no price file/rows')); continue
        j=TI[t]; l1=LASTD[i1,j]; l0=LASTD[i0,j]
        if np.isnan(l0) or np.isnan(l1) or MDdate[i1]-MDdate[int(l1)]>15 or MDdate[i0]-MDdate[int(l0)]>15:
            DROPS.append((univ,W,s,t,f"missing close within 15d: base={MD[int(l0)] if not np.isnan(l0) else None} sig={MD[int(l1)] if not np.isnan(l1) else None}")); continue
        r.append((CLOSE[i1,j]/CLOSE[i0,j]*CFP[i1,j]/CFP[i0,j]-1,t))
    r.sort(reverse=True); RC[k]=r; return r
PC={}
def period(picks,s,e,col,need_path=True):
    k=(picks,s,e,col)
    if k in PC: return PC[k]
    i0=MDi[s]; i1=MDi[e]; V,BO=A[col]; cols=[];flag=[]
    for t in picks:
        j=TI[t]; n=NXT[i0+1,j] if i0+1<nD else np.nan
        if np.isnan(n) or n>i1:
            cols.append(np.ones(i1-i0)); flag.append(f'{t}:no-trade-in-hold'); continue
        n=int(n); v=V[i0+1:i1+1,j]/BO[n,j]; v[:n-(i0+1)]=np.nan; cols.append(v)
    path=np.nanmean(np.column_stack(cols),axis=1)*0.997
    out=(path,path[-1]*0.994-1,';'.join(flag)); PC[k]=out; return out
def bench(s,e):
    j=TI['0050']; i0=MDi[s]; i1=MDi[e]; n=int(NXT[i0+1,j]); V,BO=A['ftr']
    return V[i1,j]/BO[n,j]-1
def signals(univ,W,H,M):
    anchor=None
    for mk in MONTHS:
        if int(mk[5:])==M and feasible(univ,mk,W): anchor=mk; break
    out=[]; mk=anchor
    while mk<=MONTHS[-1]: out.append(mk); mk=madd(mk,H)
    return out
def run_cell(univ,W,H,M):
    S=signals(univ,W,H,M); rows=[]
    for i,mk in enumerate(S):
        s=ME[mk]; e=ME[S[i+1]] if i+1<len(S) else LAST; cen=i+1>=len(S)
        r=rank(univ,mk,W); picks=tuple(t for _,t in r[:3])
        row=dict(univ=univ,W=W,H=H,M=M,signal=s,end=e,censored=cen,n_ranked=len(r),picks=' '.join(picks),
                 pick_rets=' '.join(f'{v*100:+.1f}' for v,_ in r[:3]))
        fl=[]
        if len(picks)<3: fl.append(f'only {len(picks)} names')
        for col,lab in (('fp','price'),('ftr','tr')):
            p,net,f=period(picks,s,e,col); row[f'{lab}_net']=net; row[f'_{lab}_path']=p
            if f and lab=='tr': fl.append(f)
        row['b0050_tr']=bench(s,e); row['flag']='; '.join(fl); rows.append(row)
    return rows
WINS=[('5y','2021-01-01','2025-12-31'),('10y','2016-01-01','2025-12-31'),('full','2005-01-01','2025-12-31')]
def summarize(rows):
    out=[]
    for w,a,b in WINS:
        q=[r for r in rows if not r['censored'] and a<=r['end']<=b]
        d=dict(univ=rows[0]['univ'],W=rows[0]['W'],H=rows[0]['H'],M=rows[0]['M'],window=w,n=len(q))
        if not q: out.append(d); continue
        yrs=(dt.date.fromisoformat(q[-1]['end'])-dt.date.fromisoformat(q[0]['signal'])).days/365.25
        d.update(first_signal=q[0]['signal'],last_end=q[-1]['end'],span_yrs=round(yrs,2))
        for lab in ('price','tr'):
            v=1;pk=1;mdd=0
            for r in q:
                z=v*r[f'_{lab}_path']  # exact chained running-peak drawdown
                run=np.maximum.accumulate(np.concatenate([[pk],z]))[1:]
                mdd=min(mdd,np.nanmin(z/run-1)); pk=max(pk,np.nanmax(z))
                v*=1+r[f'{lab}_net']; pk=max(pk,v)
            d[f'{lab}_mult']=v
            if lab=='tr': d['tr_cagr']=v**(1/yrs)-1; d['tr_maxdd']=mdd
        bm=float(np.prod([1+r['b0050_tr'] for r in q])); d['b0050_tr_mult']=bm; d['b0050_cagr']=bm**(1/yrs)-1
        d['pct_beat_0050']=np.mean([r['tr_net']>r['b0050_tr'] for r in q])
        w_=min(q,key=lambda r:r['tr_net']); d['worst_tr']=f"{w_['signal']}->{w_['end']} {w_['tr_net']*100:.1f}% ({w_['picks']})"
        d['n_flagged']=sum(bool(r['flag']) for r in q); d['beats_0050']=v>bm
        out.append(d)
    return out
