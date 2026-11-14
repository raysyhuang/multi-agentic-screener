"""Claude's fixed rule (commit 5286b46 breadth_study.py 'W10-12', H=12, 12 staggered tranches) on the EXACT v18 engine, 2015-2025.
Rule: at each signal month-end S, rank = average cross-sectional position of the strict W10, W11, W12 rankings (v18 KORS/CPRI dedupe,
optional NVDA exclusion applied before positions), ties by ticker. Top N equal weight. 12 tranches, each 1/12 of capital: all 12 buy at
the first session of Jan 2015 (signal 2014-12-31); tranche `off` then rebalances at the first session of month off, off+12, ... ;
all liquidate at the first session of Jan 2025. Composite NAV = mean of tranche NAVs (no cross-tranche rebalancing, as in Claude).
Engine: path_strict of engine432_strict2 re-executed in the loaded namespace with only the schedule header replaced
(D0,rebs,END given), then the unchanged path_strict3/path_strict4 wrappers (grading, deal post-check, NotTradable).
usage: fixed_rule_1525.py mode(base|nonvda) N"""
import sys,pickle,re,numpy as np
sys.path.insert(0,'/workspace/momentum-followup-20261114/build')
from load_v17 import load
MODE=sys.argv[1];N=int(sys.argv[2]);EXCL={'NVDA'} if MODE.startswith('nonvda') else set();S1=MODE.endswith('_s1')
ns=load();_rs=ns['rank_strict']
src=open('engine432_strict2.py').read().splitlines()
i0=next(i for i,l in enumerate(src) if l.startswith('def path_strict(W,H,K,ph,ledger=None):'))
i1=next(i for i in range(i0,len(src)) if src[i].startswith("    return dict(W=W,H=H,K=K,phase=ph,entry=D0,end=END,status='FORMAL"))
body=src[i0:i1+1];assert body[1].lstrip().startswith('D0=first_td(2015,ph)') and body[5].lstrip().startswith('rebs.append(d)')
code='\n'.join(['def path_sched(W,H,K,ph,ledger=None):','    D0,rebs,END=_SCHED']+body[6:])
if S1:   # SENSITIVITY (not the engine): cash-out / skip instead of UNKNOWN; every deviation logged in _EV
    R=[("if vr is None or d not in vr[1].index: return fail(d,'corporate_action_unresolved',t+'->'+acq+'(series unverified)'),None",
        "if vr is None or d not in vr[1].index: cash+=n*last;del hold[t];_EV.append(('cashout_stock_deal_unverified',t,d));break"),
       ("elif d>s.index[-1]: return fail(d,'corporate_action_unresolved',t),None","elif d>s.index[-1]: cash+=n*last;val+=n*last;del hold[t];_EV.append(('cashout_series_end',t,d))"),
       ("else: return fail(d,'daily_gap',t),None","else: val+=n*last;_EV.append(('stale_price_gap',t,d))"),
       ("if miss: return fail(d,'rank_missing_unbounded',' '.join(miss)),None","if miss: _EV.append(('rank_missing_ignored',' '.join(miss),d))"),
       ("else series_strict(t,d,nxt[d])","else _ss_safe(t,d,nxt[d])"),
       ("if sr is None: return fail(d,'exec_price_missing',t),None","if sr is None: _EV.append(('pick_unavailable_slot_cash',t,d));continue"),
       ("T={t:Ep/K for t in sel_keys}","T={t:Ep/K for t,_ in picks}"),
       ("held_ca(t,dprev,d)","_held_safe(t,dprev,d)")]
    for a,b in R:
        assert code.count(a)==1,a
        code=code.replace(a,b)
    exec(compile("""
_EV=[]
def _ss_safe(t,d0,d1):
    try: return series_strict(t,d0,d1)
    except _NotTradable as e: _EV.append(('not_tradable_slot_cash',str(e),d0));return None
def _held_safe(t,a,b):
    out=[]
    for ca in _held0(t,a,b):
        if ca['type']=='unresolved': _EV.append(('unresolved_ca_ignored',t+':'+ca['id'],b));continue
        out.append(ca)
    return out
""",'s1','exec'),ns)
exec(compile(code,'path_sched','exec'),ns);ns['_path2']=ns['path_sched']
def rs1(S,W):
    r,miss,nu=_rs(S,W);tk={t:g for g,t in r}
    if 'CPRI' in tk and 'KORS' in tk:
        assert tk['CPRI']==tk['KORS'];r=[(g,t) for g,t in r if t!='KORS'];nu-=1
    if EXCL:
        nu-=sum(1 for g,t in r if t in EXCL)+sum(1 for t in miss if t in EXCL)
        r=[(g,t) for g,t in r if t not in EXCL];miss=[t for t in miss if t not in EXCL]
    return r,miss,nu
def rank_avg(S,W):
    assert W==1012
    rr={w:rs1(S,w) for w in (10,11,12)};pos={}
    for w,(r,_,_) in rr.items():
        for j,(_,t) in enumerate(r): pos.setdefault(t,[]).append(j)
    out=sorted((sum(v)/3,t) for t,v in pos.items() if len(v)==3)
    miss=sorted(set().union(*[set(m) for _,m,_ in rr.values()]))
    ns['RCAS'][(S,1012)]=set().union(*[ns['RCAS'].get((S,w),set()) for w in (10,11,12)])
    return [(-a,t) for a,t in out],miss,rr[12][2]
ns['rank_strict']=rank_avg;ns['RS'].clear();ns['RCAS'].clear()
ft=ns['first_td'];D0=ft(2015,1);END=ft(2025,1)
P=[];NAV={};RB=[]
for off in range(12):
    rebs=[D0]+[d for d in (ft(2015+m//12,m%12+1) for m in range(off if off else 12,120,12)) if d<END]
    ns['_SCHED']=(D0,rebs,END)
    if S1:
        ns['_EV'].clear();led=[];row,eq=ns['path_sched'](1012,12,N,off+1,ledger=led);ev=list(ns['_EV'])
        if row['status']!='UNKNOWN': row['status']='SENS_CASHOUT' if ev else 'SENS_CLEAN'
        row['fail_ticker']=(row.get('fail_ticker') or '')+(' EV:'+';'.join(f'{a}|{b}|{c}' for a,b,c in ev[:40]) if ev else '')
    else: row,eq,led=ns['path_strict4'](1012,12,N,off+1)
    tid=f'{MODE}_N{N}_T{off}'
    post=None
    for x in led:
        if x.get('event')!='rebalance': continue
        RB.append(dict(tranche=tid,signal_date=x['signal'],trade_date=x['date'],picks=x['picks']))
        if post is None and row['status']!='UNKNOWN':
            for t in x['picks'].split('|'):
                if any(x['signal']<e<=x['date'] for e in ns['_dealdate'](t)): post=f"{t}@{x['date']}";break
    P.append(dict(tranche=tid,mode=MODE,N=N,off=off,entry=D0,end=END,n_rebal=len(rebs),status=row['status'],fail_reason=row.get('fail_reason'),fail_ticker=row.get('fail_ticker'),
                  fail_date=row.get('fail_date'),final_multiple=row['wealth'],CAGR=row['CAGR'],daily_MDD=row['daily_MDD'],postcheck_all_picks_hit=post))
    if eq is not None: NAV[tid]=np.asarray(eq,dtype=float)
    print(tid,row['status'],row.get('CAGR'),row.get('fail_reason'),row.get('fail_ticker'),flush=True)
days=ns['SD'][ns['SDI'][D0]:ns['SDI'][END]+1]
pickle.dump(dict(P=P,NAV=NAV,RB=RB,days=days),open(f'/workspace/momentum-followup-20261114/scratch/fixed1525_{MODE}_N{N}.pkl','wb'));print('saved',flush=True)
