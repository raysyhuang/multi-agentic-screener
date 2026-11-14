"""Queen v17 strict4 + v18 KORS/CPRI dedupe, optional NVDA exclusion from the eligible universe at every signal.
Engine loaded read-only (exec) from /workspace/second-sleeve-v17-targeted-ca-20261108/engine, nothing written there.
usage: run_grid.py <W list comma> <mode: base|nonvda>"""
import sys,pickle,numpy as np
sys.path.insert(0,'/workspace/no-nvda-grid-20261114/build')
from load_v17 import load
Ws=[int(x) for x in sys.argv[1].split(',')];MODE=sys.argv[2];EXCL={'NVDA'} if MODE=='nonvda' else set()
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
SD=ns['SD'];P=[];RB=[];NAV={}
for W in Ws:
  for H in range(1,13):
    for K in (1,2,3):
      for ph in range(1,13):
        row,eq,led=ns['path_strict4'](W,H,K,ph);pid=f'W{W}H{H}K{K}P{ph}'
        for x in led:
            if x.get('event')=='rebalance': RB.append(dict(path_id=pid,W=W,H=H,K=K,phase=ph,signal_date=x.get('signal'),trade_date=x.get('date'),picks=x.get('picks')))
        P.append(dict(path_id=pid,W=W,H=H,K=K,phase=ph,entry=row['entry'],end=row['end'],status=row['status'],fail_reason=row.get('fail_reason'),fail_ticker=row.get('fail_ticker'),
            fail_date=row.get('fail_date'),final_multiple=row['wealth'],CAGR=row['CAGR'],daily_MDD=row['daily_MDD'],grade_C_events=row.get('grade_C_events')))
        if eq is not None: NAV[pid]=np.asarray(eq,dtype=float)
  print('W',W,'done',flush=True)
pickle.dump(dict(P=P,RB=RB,NAV=NAV),open(f'/workspace/no-nvda-grid-20261114/scratch/{MODE}_{sys.argv[1].replace(",","_")}.pkl','wb'))
print('saved',len(P),flush=True)
