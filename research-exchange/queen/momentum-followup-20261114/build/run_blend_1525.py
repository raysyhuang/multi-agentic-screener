"""2015-2025: Queen v17 strict4 + v18 dedupe (engine read-only), optional NVDA exclusion. Runs W in Ws, H in Hs, K=N in Ns, 12 phases.
Also records the deal post-check applied to ALL picks (the v17 engine's K=1-only quirk) as an extra column.
usage: run_blend_1525.py W mode Hs Ns"""
import sys,pickle,numpy as np
sys.path.insert(0,'/workspace/momentum-followup-20261114/build')
from load_v17 import load
W=int(sys.argv[1]);MODE=sys.argv[2];Hs=[int(x) for x in sys.argv[3].split(',')];Ns=[int(x) for x in sys.argv[4].split(',')]
EXCL={'NVDA'} if MODE=='nonvda' else set()
ns=load();_rs=ns['rank_strict']
def rank_strict(S,W):
    r,miss,nu=_rs(S,W);tk={t:g for g,t in r}
    if 'CPRI' in tk and 'KORS' in tk:
        assert tk['CPRI']==tk['KORS'];r=[(g,t) for g,t in r if t!='KORS'];nu-=1
    if EXCL:
        nu-=sum(1 for g,t in r if t in EXCL)+sum(1 for t in miss if t in EXCL)
        r=[(g,t) for g,t in r if t not in EXCL];miss=[t for t in miss if t not in EXCL]
    return r,miss,nu
ns['rank_strict']=rank_strict;ns['RS'].clear();ns['RCAS'].clear()
P=[];NAV={};RB=[]
for H in Hs:
  for K in Ns:
    for ph in range(1,13):
        row,eq,led=ns['path_strict4'](W,H,K,ph);pid=f'W{W}H{H}K{K}P{ph}'
        post=None
        for x in led:
            if x.get('event')!='rebalance': continue
            RB.append(dict(path_id=pid,signal_date=x['signal'],trade_date=x['date'],picks=x['picks']))
            if post is None and K>1 and row['status']!='UNKNOWN':
                for t in x['picks'].split('|'):
                    if any(x['signal']<e<=x['date'] for e in ns['_dealdate'](t)): post=f"{t}@{x['date']}";break
        P.append(dict(path_id=pid,W=W,H=H,K=K,phase=ph,entry=row['entry'],end=row['end'],status=row['status'],fail_reason=row.get('fail_reason'),fail_ticker=row.get('fail_ticker'),
            fail_date=row.get('fail_date'),final_multiple=row['wealth'],CAGR=row['CAGR'],daily_MDD=row['daily_MDD'],postcheck_all_picks_hit=post))
        if eq is not None: NAV[pid]=np.asarray(eq,dtype=float)
    print('H',H,'K',K,'done',flush=True)
pickle.dump(dict(P=P,NAV=NAV,RB=RB),open(f'/workspace/momentum-followup-20261114/scratch/blend1525_{MODE}_W{W}_H{sys.argv[3].replace(",","_")}.pkl','wb'));print('saved',flush=True)
