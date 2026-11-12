import pickle,pandas as pd,numpy as np,json
res=pickle.load(open('work/results.pkl','rb'))
AK='/workspace/momentum432-answer-key-full-20261109/'
P=pd.read_csv(AK+'paths_summary.csv').set_index('path_id')
VAR=['none','A','B','A_next','B_next']
rows=[];ev=[]
for (W,H,ph,v),o in res.items():
    pid=f'W{W}H{H}K1P{ph}'
    r=dict(path_id=pid,W=W,H=H,K=1,phase=ph,variant=v,entry=o['entry'],end=o['end'],status=o['status'],final_multiple=o.get('final'),CAGR=o.get('CAGR'),daily_MDD=o.get('MDD'),
           n_stops=len(o['stops']),n_cash_stops=sum(1 for s in o['stops'] if not s['replacement']),n_scheduled_rebalances=o.get('n_rebalances'),
           first_stop_date=o['stops'][0]['date'] if o['stops'] else None,
           days_to_first_stop=(pd.Timestamp(o['stops'][0]['date'])-pd.Timestamp(o['entry'])).days if o['stops'] else None,
           n_superseded_pending=len(o.get('superseded',[])),fail_reason=o.get('fail_reason'))
    rows.append(r)
    for i,s in enumerate(o['stops']):
        ev.append(dict(path_id=pid,variant=v,stop_no=i+1,trigger_date=s['date'],exec_date=s['exec_date'],stopped=s['stopped'],held_kind=s['held_kind'],position_entry=s['entry_date'],
            peak_date=s['peak_date'],peak_close=s['peak'],trigger_close=s['stop_close'],dd_at_trigger=s['dd'],exec_close=s['exec_close'],signal_used=s['signal'],rank_W=s['W_rank'],
            top1=s['top1'],top2=s['top2'],replacement=s['replacement'],replacement_series=s['replacement_series'],cash_reason=s['cash_reason'] or None,
            equity_before=s['E_before'],equity_after=s['E_after'],cost_factor=s['E_after']/s['E_before'],next_rebalance=s['next_rebalance'],
            days_in_position=(pd.Timestamp(s['date'])-pd.Timestamp(s['entry_date'])).days))
L=pd.DataFrame(rows);E=pd.DataFrame(ev)
E.to_csv('stop_events.csv',index=False)
# wide per-path
w=L.pivot_table(index=['path_id','W','H','phase'],columns='variant',values=['CAGR','daily_MDD','final_multiple','n_stops','n_scheduled_rebalances','status','n_cash_stops'],aggfunc='first')
w.columns=[f'{a}_{b}' for a,b in w.columns];w=w.reset_index()
w['official_CAGR']=[P.loc[p,'CAGR'] for p in w.path_id];w['official_MDD']=[P.loc[p,'daily_MDD'] for p in w.path_id];w['official_status']=[P.loc[p,'status'] for p in w.path_id]
for v in ['A','B','A_next','B_next']:
    w[f'dCAGR_{v}_pp']=(w[f'CAGR_{v}']-w['CAGR_none'])*100;w[f'dMDD_{v}_pp']=(w[f'daily_MDD_{v}']-w['daily_MDD_none'])*100
cols=['path_id','W','H','phase','official_status','official_CAGR','official_MDD']+[f'{m}_{v}' for v in VAR for m in ['status','CAGR','daily_MDD','final_multiple','n_stops','n_cash_stops','n_scheduled_rebalances']]+[f'{d}_{v}_pp' for v in ['A','B','A_next','B_next'] for d in ['dCAGR','dMDD']]
w=w[cols].sort_values(['W','H','phase'])
band=w[w.H<=4];band.to_csv('per_path_W10-12_H2-4_K1.csv',index=False)
w[w.H==12].to_csv('side_annual_W12H12K1P1.csv',index=False)
L.to_csv('per_path_long.csv',index=False)
# summaries
def summ(df,label):
    out=[]
    for v in VAR:
        c=df[f'CAGR_{v}'];m=df[f'daily_MDD_{v}'];n=df[f'n_stops_{v}'] if v!='none' else pd.Series([0]*len(df))
        out.append(dict(group=label,variant=v,n_paths=len(df),median_CAGR=c.median(),mean_CAGR=c.mean(),min_CAGR=c.min(),max_CAGR=c.max(),median_MDD=m.median(),worst_MDD=m.min(),
            paths_with_stop=int((n>0).sum()),total_stops=int(n.sum()),
            n_better_CAGR=int((c>df.CAGR_none+1e-12).sum()) if v!='none' else None,n_worse_CAGR=int((c<df.CAGR_none-1e-12).sum()) if v!='none' else None,
            median_dCAGR_pp=float(((c-df.CAGR_none)*100).median()) if v!='none' else None,median_dMDD_pp=float(((m-df.daily_MDD_none)*100).median()) if v!='none' else None))
    return out
S=[]
S+=summ(band,'W10-12xH2-4 (108)')
for (W,H),g in band.groupby(['W','H']): S+=summ(g,f'W{W}H{H} (12 phases)')
S+=summ(band[band.phase==1],'phase1 only (9)')
S=pd.DataFrame(S);S.to_csv('summary_by_group.csv',index=False)
pd.set_option('display.width',250);pd.set_option('display.max_columns',30)
print(S[S.group.str.startswith(('W10-12','phase1'))].to_string())
print(S[S.variant.isin(['none','A','B'])].pivot(index='group',columns='variant',values='median_CAGR').to_string())
for p in ['W10H2K1P1','W12H4K1P1']:
    print(w[w.path_id==p][['path_id']+[f'{m}_{v}' for v in VAR for m in ['CAGR','daily_MDD','n_stops']]].T.to_string())
print(w[w.H==12][['path_id']+[f'{m}_{v}' for v in VAR for m in ['CAGR','daily_MDD','n_stops','final_multiple']]].T.to_string())
for v in ['A','B']:
    b=band.sort_values(f'dCAGR_{v}_pp');print(v,'worst');print(b[['path_id','CAGR_none',f'CAGR_{v}',f'dCAGR_{v}_pp',f'dMDD_{v}_pp',f'n_stops_{v}']].head(5).to_string());print(v,'best');print(b[['path_id','CAGR_none',f'CAGR_{v}',f'dCAGR_{v}_pp',f'dMDD_{v}_pp',f'n_stops_{v}']].tail(5).to_string())
Eb=E[E.path_id.isin(band.path_id)]
for v in ['A','B','A_next','B_next']:
    e=Eb[Eb.variant==v];f=band[f'n_stops_{v}']>0
    tf=L[(L.variant==v)&(L.H<=4)].days_to_first_stop
    print(v,'events',len(e),'paths w stop',f.sum(),'/108','cash',e.replacement.isna().sum(),'time_to_first median/mean days',tf.median(),round(tf.mean(),1),'days_in_pos med',e.days_in_position.median(),
          'repl counts',e.replacement.value_counts().head(12).to_dict(),'top2 used',(e.replacement==e.top2).sum(),'stopped names',e.stopped.value_counts().head(10).to_dict(),'uniq triggers',e[['trigger_date','stopped']].drop_duplicates().shape[0])
json.dump({'unknown':L[L.status=='UNKNOWN'][['path_id','variant','fail_reason']].to_dict('records'),'superseded':int(L.n_superseded_pending.sum())},open('work/flags.json','w'))
print(open('work/flags.json').read())
