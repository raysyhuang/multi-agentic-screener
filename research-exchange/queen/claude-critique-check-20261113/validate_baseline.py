"""Check engine_ext with no switches reproduces (a) the frozen 2015-2025 answer key for K1 (all 1,728 K1 paths) and
(b) the 2005-2015 packet's 25 paths."""
import csv,sys,time;sys.path.insert(0,'.')
from engine_ext import Eng
out=open('validation_baseline.txt','w')
def P(*a): print(*a);print(*a,file=out);out.flush()
E=Eng('/workspace/momentum432-inputs-pack-full-20261109',2015)
ak={r['path_id']:r for r in csv.DictReader(open('/workspace/momentum432-answer-key-full-20261109/paths_summary.csv'))}
bad=0;n=0;t0=time.time()
for pid,r in ak.items():
    if r['K']!='1': continue
    o=E.path(int(r['W']),int(r['H']),1,int(r['phase']));n+=1
    if o['status']!=r['status'] or (o['status']!='UNKNOWN' and abs(o['CAGR']-float(r['CAGR']))>1e-9) or o['flags']:
        bad+=1;P('MISMATCH',pid,o['status'],o.get('CAGR'),r['status'],r['CAGR'],o['flags'])
P('2015-2025 K1 paths checked',n,'mismatches',bad,round(time.time()-t0,1),'s')
E2=Eng('/workspace/w10h2-w12h4-oos-2005-2015-20261112/pack',2005)
ref={r['path']:r for r in csv.DictReader(open('/workspace/w10h2-w12h4-oos-2005-2015-20261112/results/results_paths.csv'))}
bad=0
for k,r in ref.items():
    if not k.startswith('W'): continue
    o=E2.path(int(r['W']),int(r['H']),1,int(r['phase']))
    ok=o['status']==r['status'] and (o['status']=='UNKNOWN' or abs(o['CAGR']-float(r['CAGR']))<1e-9) and not o['flags']
    if not ok: bad+=1;P('MISMATCH',k,o['status'],o.get('CAGR'),r['status'],r['CAGR'],o['flags'])
P('2005-2015 packet paths checked',len([k for k in ref if k.startswith('W')]),'mismatches',bad)
