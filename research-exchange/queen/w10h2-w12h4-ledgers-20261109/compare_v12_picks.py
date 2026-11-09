"""Pick-by-pick diff between the sealed v12 ledger (pre-unblind) and this v17 export, to explain the median change."""
import gzip,json,pandas as pd,os
OUT=os.path.dirname(os.path.abspath(__file__))
v12={}
with gzip.open('/workspace/momentum432-engine-20261108/out_strict4/ledger.jsonl.gz','rt') as f:
    for l in f:
        x=json.loads(l)
        if x.get('event')=='rebalance' and x['K']==1 and (x['W'],x['H']) in((10,2),(12,4)): v12[(f"W{x['W']}H{x['H']}K1",x['phase'],x['date'])]=x['picks']
t=pd.read_csv(f'{OUT}/trades.csv');t=t[(t.event=='rebalance')&(t.action!='sell')]
v17=t.groupby(['config','phase','trade_date']).ticker.apply(lambda s:'|'.join(s)).to_dict()
rows=[dict(config=k[0],phase=k[1],trade_date=k[2],v12_pick=v12.get(k),v17_pick=v,changed=v12.get(k)!=v) for k,v in v17.items()]
d=pd.DataFrame(rows).sort_values(['config','phase','trade_date']);d.to_csv(f'{OUT}/pick_diff_v12_vs_v17.csv',index=False)
print(d.groupby('config').changed.agg(['sum','count']))
f=d[d.changed].groupby(['config','phase']).trade_date.min();print(f)
print(d[d.changed].head(12).to_string())
