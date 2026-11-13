"""Full 432 x 12-phase grid with engine_ext (no switches). argv: window(2005|2015). Writes grid_paths_<y0>.csv"""
import csv,sys,time;sys.path.insert(0,'.')
import engine_ext as M
from engine_ext import Eng
y0=int(sys.argv[1]);tag=sys.argv[2] if len(sys.argv)>2 else 'strict'
M.APPROX_DELIST=(tag=='approx')
E=Eng('/workspace/momentum432-inputs-pack-full-20261109' if y0==2015 else '/workspace/w10h2-w12h4-oos-2005-2015-20261112/pack',y0)
rows=[];t0=time.time()
for W in range(1,13):
  for H in range(1,13):
    for K in (1,2,3):
      for ph in range(1,13):
        try: o=E.path(W,H,K,ph)
        except Exception as ex: o=dict(status='UNKNOWN',fail_reason='engine_exception:'+repr(ex)[:80],flags=set(),rebalances=[])
        rows.append(dict(W=W,H=H,K=K,phase=ph,status=o['status'],CAGR=o.get('CAGR',''),MDD=o.get('MDD',''),final=o.get('final',''),
            fail_reason=o.get('fail_reason',''),fail_ticker=o.get('fail_ticker',''),fail_date=o.get('fail_date',''),
            ext_exec=('ext_exec' in o['flags']),ext_spin=';'.join(sorted(f for f in o['flags'] if f.startswith('ext_spin'))),ext_delist=';'.join(sorted(f for f in o['flags'] if f.startswith('ext_delist'))),handover=('series_handover' in o['flags']),n_reb=len(o['rebalances'])))
  print('W',W,round(time.time()-t0,1),flush=True)
w=csv.DictWriter(open(f'grid_paths_{y0}_{tag}.csv','w',newline=''),fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('done')
