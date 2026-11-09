"""12-phase v17 vs v1.2 for W12H6K1, W12H12K1, W9H2K1 (read-only frozen outputs) + pick diffs from ledgers."""
import gzip,json,pandas as pd,os
OUT=os.path.dirname(os.path.abspath(__file__))
v17=pd.read_csv('/workspace/second-sleeve-v17-targeted-ca-20261108/full_coverage/paths_status.csv')
v12=pd.read_csv('/workspace/momentum432-engine-20261108/out_strict4/paths_status.csv')
CF=[(12,6),(12,12),(9,2)];rows=[]
for W,H in CF:
    for ph in range(1,13):
        a=v17[(v17.W==W)&(v17.H==H)&(v17.K==1)&(v17.phase==ph)].iloc[0];b=v12[(v12.W==W)&(v12.H==H)&(v12.K==1)&(v12.phase==ph)].iloc[0]
        rows.append(dict(config=f'W{W}H{H}K1',phase=ph,entry=a.entry,end=a.end,v17_status=a.status,v17_fail_reason=a.fail_reason if isinstance(a.fail_reason,str) else '',
            v17_fail_ticker=a.fail_ticker if isinstance(a.fail_ticker,str) else '',v17_final_multiple=a.wealth,v17_CAGR=a.CAGR,v17_daily_MDD=a.daily_MDD,
            v17_grade_C=a.grade_C_events if isinstance(a.grade_C_events,str) else '',v12_status=b.status,v12_CAGR=b.CAGR,v12_daily_MDD=b.daily_MDD))
d=pd.DataFrame(rows)
# pick diffs v12 ledger vs v17 replay
v12p={}
with gzip.open('/workspace/momentum432-engine-20261108/out_strict4/ledger.jsonl.gz','rt') as f:
    for l in f:
        x=json.loads(l)
        if x.get('event')=='rebalance' and x['K']==1 and (x['W'],x['H']) in CF: v12p[(x['W'],x['H'],x['phase'],x['date'])]=x['picks']
from load_engine import load
ns=load();diff=[]
for W,H in CF:
    for ph in range(1,13):
        r,eq,led=ns['path_strict4'](W,H,1,ph)
        assert r['status']=='UNKNOWN' or abs(r['CAGR']-d[(d.config==f'W{W}H{H}K1')&(d.phase==ph)].v17_CAGR.iloc[0])<1e-12
        for x in led:
            if x['event']!='rebalance': continue
            k=(W,H,ph,x['date']);p12=v12p.get(k)
            if p12!=x['picks']: diff.append(dict(config=f'W{W}H{H}K1',phase=ph,trade_date=x['date'],v12_pick=p12,v17_pick=x['picks']))
dd=pd.DataFrame(diff);dd.to_csv(f'{OUT}/anchors_pick_diff_v12_vs_v17.csv',index=False)
for cfg,g in dd.groupby('config'): d.loc[d.config==cfg,'n_pick_changes_v12_to_v17_all_phases']=len(g)
d.to_csv(f'{OUT}/anchors_W12H6_W12H12_W9H2_v12_vs_v17.csv',index=False)
print(d.assign(**{c:(d[c]*100).round(2) for c in['v17_CAGR','v17_daily_MDD','v12_CAGR','v12_daily_MDD']}).drop(columns=['entry','end','v17_grade_C']).to_string())
print(d.groupby('config')[['v17_CAGR','v17_daily_MDD','v12_CAGR','v12_daily_MDD']].median()*100)
print(d.groupby('config')[['v17_CAGR','v12_CAGR']].agg(['min','max'])*100)
print(dd.groupby('config').v12_pick.value_counts().head(20)); print(dd.groupby('config').trade_date.agg(['min','max','count']))
