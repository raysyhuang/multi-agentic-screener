"""Dump engine rankings (v17 strict4 + v18 dedupe) top-25 per (S,W) for month-ends 2014-11..2025-01, W1-12. NVDA kept (filter later)."""
import sys,pickle
sys.path.insert(0,'/workspace/momentum-followup-20261114/build')
from load_v17 import load
ns=load();_rs=ns['rank_strict']
def rank_strict(S,W):
    r,miss,nu=_rs(S,W);tk={t:g for g,t in r}
    if 'CPRI' in tk and 'KORS' in tk: r=[(g,t) for g,t in r if t!='KORS'];nu-=1
    return r,miss,nu
MEL=ns['MEL'];SS=[m for m in MEL if '2014-11-01'<=m<='2025-01-31']
out={}
for S in SS:
    for W in range(1,13):
        r,miss,nu=rank_strict(S,W);out[(S,W)]=dict(top=[t for _,t in r[:25]],miss=miss,n=nu)
pickle.dump(out,open('/workspace/momentum-followup-20261114/scratch/ranks_1525.pkl','wb'));print(len(out))
