import pickle,pandas as pd,numpy as np
AK='/workspace/momentum432-answer-key-full-20261109/'
res=pickle.load(open('work/results.pkl','rb'))
P=pd.read_csv(AK+'paths_summary.csv').set_index('path_id')
R=pd.read_csv(AK+'rebalances.csv.gz')
navs={}
rows=[];bad=0
for (W,H,ph,v),o in res.items():
    if v!='none': continue
    pid=f'W{W}H{H}K1P{ph}';a=P.loc[pid]
    if W not in navs: navs[W]=pd.read_csv(AK+f'nav_daily/nav_daily_W{W:02d}.csv.gz',index_col=0)
    nv=navs[W][pid].dropna().values
    rb=R[R.path_id==pid];mine=[x for x in o['log'] if x['event']=='rebalance']
    picks_ok=list(rb.picks)==[x['pick'] for x in mine] and list(rb.trade_date)==[x['date'] for x in mine]
    navd=float(np.max(np.abs(o['nav']/nv-1))) if len(nv)==len(o['nav']) else np.inf
    r=dict(path_id=pid,status_ok=o['status']==a.status,final_rel=abs(o['final']/a.final_multiple-1),CAGR_diff=abs(o['CAGR']-a.CAGR),MDD_diff=abs(o['MDD']-a.daily_MDD),
           nreb_ok=o['n_rebalances']==a.n_rebalances_executed,picks_ok=picks_ok,nav_maxrel=navd)
    r['match']=r['status_ok'] and r['final_rel']<1e-9 and r['CAGR_diff']<1e-9 and r['MDD_diff']<1e-9 and r['nreb_ok'] and picks_ok and navd<1e-9
    rows.append(r)
df=pd.DataFrame(rows);df.to_csv('baseline_reproduction_check.csv',index=False)
print(len(df),'match',df.match.sum());print(df.describe())
print(df[~df.match])
