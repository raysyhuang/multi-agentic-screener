"""S&P 500 side. Loads Queen v17 strict4 read-only (exec, via no-nvda-grid build/load_v17.py), applies v18 KORS/CPRI dedupe,
and runs (1) W6H1K3P1 as in the 2026-11-09 answer key (total-return signal) and (2) the NDX post-rule signal on the same
universe/execution: MEAN of the last 6 one-month returns (each month's rank-price return, spin/special-div adjusted inside that
month exactly as rank_strict does over (B,S]). Official window 2015-01-02..2025-01-02, plus an UNAUDITED extension to 2026-10-01
(MEL list extended in memory only; corporate_actions/membership after 2025 not audited). Nothing written under the engine folder."""
import sys,pickle,numpy as np
sys.path.insert(0,'/workspace/no-nvda-grid-20261114/build')
from load_v17 import load
ns=load()
MEAN_SRC='''
def rank_mean6(S,W):
    i=MEL.index(S); Ms=MEL[i-W:i+1]; B=Ms[0]; u=seasoned(S); r=[];miss=[];cas=set()
    for t in sorted(u):
        if any(t==a and b<=S<=e for a,b,e in SKIPL): miss.append(t);continue
        if any(B<x['ex_date']<=S for x in ca_of(t) if x['type']=='unresolved'): miss.append(t);continue
        px=[mpx(m,t,S) for m in Ms]
        if not all(px) or any(p<=0 for p in px): miss.append(t);continue
        rets=[];ok=True
        for m0,m1,a,b in zip(Ms,Ms[1:],px,px[1:]):
            for ca in ca_of(t):
                if ca['type'] in('spin','special_div') and m0<ca['ex_date']<=m1:
                    pe=pex3(t,ca['ex_date'])
                    if pe is None: ok=False;break
                    ch=float(ca['ratio'])*float(ca['child_first_close'] or 0)+float(ca['cash_per_share'])
                    a=a*pe/(pe+ch);cas.add(ca['id'])
            if not ok: break
            rets.append(b/a-1)
        if ok: r.append((sum(rets)/W,t))
        else: miss.append(t)
    r.sort(key=lambda x:(-x[0],x[1])); RCAS[(S,W)]=cas; return r,miss,len(u)
'''
exec(MEAN_SRC,ns)
_rs_tot=ns['rank_strict']; _rs_mean=ns['rank_mean6']
def dedupe(f):
    def g(S,W):
        r,miss,nu=f(S,W);tk={t:x for x,t in r}
        if 'CPRI' in tk and 'KORS' in tk:
            assert abs(tk['CPRI']-tk['KORS'])<1e-12;r=[(x,t) for x,t in r if t!='KORS'];nu-=1
        return r,miss,nu
    return g
LEN=[]
def lenient(f,tag):
    def g(S,W):
        r,miss,nu=f(S,W); d=ns['SD'][ns['SDI'][S]+1]
        for t in miss: LEN.append(dict(variant=tag,signal=S,ticker=t,action='dropped_unrankable_missing'))
        bad={t for _,t in r if any(S<e<=d for e in ns['_dealdate'](t))}
        for i,(x,t) in enumerate(r):
            if t in bad: LEN.append(dict(variant=tag,signal=S,ticker=t,action='dropped_deal_effective_by_trade_date',rank=i+1))
        return [(x,t) for x,t in r if t not in bad],[],nu
    return g
def run(signal,end_override=None,len_tag=None):
    f=dedupe(_rs_tot if signal=='total' else _rs_mean)
    ns['rank_strict']=lenient(f,len_tag) if len_tag else f;ns['RS'].clear();ns['RCAS'].clear()
    if end_override: ns['_END_OVERRIDE']=end_override
    else: ns.pop('_END_OVERRIDE',None)
    row,eq,led=ns['path_strict4'](6,1,3,1)
    D0=row['entry'];END=row['end'] or end_override or ns['first_td'](2025,1)
    days=ns['SD'][ns['SDI'][D0]:ns['SDI'][END]+1]
    rb=[dict(signal=x['signal'],trade=x['date'],picks=x.get('picks')) for x in led if x.get('event')=='rebalance']
    rk={}
    for S in sorted({x['signal'] for x in rb}):
        r,miss,nu=ns['RS'][(S,6)]; rk[S]=dict(top10=[(t,float(v)) for v,t in r[:10]],n_ranked=len(r),n_missing=len(miss),n_universe=nu)
    return dict(row={k:(float(v) if isinstance(v,(np.floating,)) else v) for k,v in row.items()},eq=None if eq is None else list(map(float,eq)),days=days,rb=rb,rank=rk,
                ledger=[x for x in led if x.get('event')!='rebalance'])
out={}
out['W6H1K3P1_total_official']=run('total')
out['mean6_H1K3P1_official']=run('mean')
ns['MEL'].extend([m for m in ns['ME'] if '2025-12-31'<m<='2026-09-30'])
END2=ns['first_td'](2026,10)
out['W6H1K3P1_total_ext']=run('total',END2)
out['mean6_H1K3P1_ext']=run('mean',END2)
ns['MEL'][:]=[m for m in ns['MEL'] if m<='2025-12-31']
out['mean6_H1K3P1_official_LENIENT']=run('mean',None,'mean6_official_lenient')
out['W6H1K3P1_total_official_LENIENT']=run('total',None,'total_official_lenient')
ns['MEL'].extend([m for m in ns['ME'] if '2025-12-31'<m<='2026-09-30'])
out['mean6_H1K3P1_ext_LENIENT']=run('mean',END2,'mean6_ext_lenient')
out['W6H1K3P1_total_ext_LENIENT']=run('total',END2,'total_ext_lenient')
# no-seasoning sensitivity: universe = raw S&P PIT snapshot at S (the NDX post rule has no 365-day seasoning filter)
import bisect as _b
_seas_eng=ns['seasoned']; _md=ns['md']; _mem=ns['mem']
def raw_mem(S):
    k=_b.bisect_right(_md,S)-1; m=_mem[k]; return set(m[1] if isinstance(m,tuple) else m)
ns['seasoned']=raw_mem
ns['MEL'][:]=[m for m in ns['MEL'] if m<='2025-12-31']
out['mean6_H1K3P1_official_LENIENT_NOSEASON']=run('mean',None,'mean6_official_lenient_noseason')
ns['MEL'].extend([m for m in ns['ME'] if '2025-12-31'<m<='2026-09-30'])
out['mean6_H1K3P1_ext_LENIENT_NOSEASON']=run('mean',END2,'mean6_ext_lenient_noseason')
ns['seasoned']=_seas_eng
out['lenient_log']=LEN
for k,v in out.items():
  if 'row' in v: print(k,v['row']['status'],v['row'].get('fail_reason'),v['row'].get('fail_ticker'),v['row'].get('fail_date'),v['row']['wealth'],v['row']['CAGR'],v['row']['daily_MDD'],v['row'].get('grade_C_events'))
# S&P PIT membership snapshots and seasoned universe for overlap
md=ns['md'];mem=ns['mem']
out['sp_mem']=[(d,sorted(s)) for d,s in mem if '2013-01'<=d]
out['MEL']=list(ns['MEL']);out['ME']=list(ns['ME'])
out['seasoned']={S:sorted(ns['seasoned'](S)) for S in ns['MEL'] if S>='2014-12'}
# SPY closes (legacy env spy_daily) for benchmark
spy=ns['spy'];out['spy']=dict(zip(spy.date,map(float,spy.c)))
pickle.dump(out,open('/workspace/ndx-vs-spx-overlap-20261114/build/spx.pkl','wb'))
