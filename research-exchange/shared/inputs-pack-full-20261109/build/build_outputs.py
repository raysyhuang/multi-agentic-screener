"""Merge run_part outputs into folder A (inputs) and folder B (answer key). Engine loaded read-only for event rules."""
import sys,glob,pickle,json,os,shutil,collections,numpy as np,pandas as pd
sys.path.insert(0,'/workspace/momentum432-answer-key-full-20261109/build')
from load_v17 import load
A='/workspace/momentum432-inputs-pack-full-20261109/';B='/workspace/momentum432-answer-key-full-20261109/';OLD='/workspace/momentum432-inputs-pack-20261109/'
parts=[pickle.load(open(f,'rb')) for f in sorted(glob.glob('/workspace/m432full-scratch/part_*.pkl'))]
ns=load();SD=ns['SD'];MEL=ns['MEL']
for f in ['membership_intervals.csv','eligibility_by_signal.csv.gz','rank_prices_month_end.csv.gz','rank_price_overrides.csv','corporate_actions_applied.csv','trading_sessions.csv']:
    shutil.copy2(OLD+f,A+f)
CL=pd.concat([p['CL'] for p in parts if p['CL'] is not None]).drop_duplicates()
g=CL.groupby(['price_series','execution_key','date']).close
spread=(g.max()/g.min()-1).max();assert spread<1e-12,spread
CL=CL.drop_duplicates(['price_series','execution_key','date']).sort_values(['price_series','execution_key','date'])
CL['close']=CL.close.map(lambda v:repr(float(v)))
CL.to_csv(A+'execution_daily_closes.csv.gz',index=False,compression='gzip')
keys=set();SL={};SF={}
for p in parts: keys|=set(p['SPELLKEYS']);SL.update(p['SERLAST']);SF.update(p['SERFIRST'])
pd.DataFrame([dict(price_series=k[0],execution_key=k[1],series_first_date=SF[k],series_last_date=SL[k]) for k in sorted(keys)]).to_csv(A+'series_info.csv',index=False)
_F={}
def fmp_oc(t,d):
    if t not in _F:
        p=f'/workspace/nh-matrix-bt-v2/bars/{t}.px.json';_F[t]={r['date']:(r['open'],r['close']) for r in json.load(open(p))} if os.path.exists(p) else {}
    return _F[t].get(d,(None,None))
EX={}
for p in parts:
    for k,v in p['EXA'].items(): EX[k]=(v,p['EXAC'][k])
rows=[]
for (d0,t,d1),((st,nm),cl) in sorted(EX.items()):
    o=None;note=''
    if st=='ok':
        base=nm.split(':',1)[1].split('(')[0].split('+')[0].split('#')[0] if nm.startswith('fmp:') else None
        if base:
            oo,fc=fmp_oc(base.replace('.','-'),d0)
            if fc is not None and cl and abs(fc/cl-1)<1e-9: o=oo;note='open from same FMP file/scale as close'
        if o is None: note='UNKNOWN: open not available on the same price scale (informational only; engine trades at close)'
    rows.append(dict(trade_date=d0,engine_ticker=t,window_end=d1,status=st,price_series=nm if st=='ok' else '',detail=nm if st!='ok' else '',
        exec_close=repr(cl) if cl is not None else '',exec_open=o if o is not None else '',open_note=note))
EXA=pd.DataFrame(rows);EXA.to_csv(A+'execution_availability.csv.gz',index=False,compression='gzip')
TR=ns['TICKER_REUSE'];GR={r['id']:r.get('evidence_grade','') for r in ns['CAL']}
EV=[]
for nm,t in sorted(keys):
    for ca in ns['ca_of'](t):
        ty=ca['type'];ex=ca['ex_date'];dproc=next((d for d in SD if d>=ex),None)
        r=dict(execution_key=t,price_series=nm,event_id=ca['id'],type=ty,ex_date=ex,process_date=dproc,ratio=ca['ratio'],cash_per_share=ca['cash_per_share'],
               child_first_close=ca['child_first_close'],child_or_acquirer=ca['child_or_acquirer'],evidence_grade=GR.get(ca['id'],''),acquirer_series='',acquirer_verified='')
        if ty in('spin','special_div'): r.update(action='apply',reason='spin/special: child value (ratio x child_first_close, fee c) + cash to cash')
        elif ty in('cash','stock') and SL[(nm,t)]>=ex and ca['id'] not in TR: r.update(action='skip',reason='held series continues past ex (already successor/renamed series)')
        elif ty=='cash': r.update(action='apply',reason='cash deal: n x cash_per_share to cash, holding ends')
        elif ty=='stock':
            acq=ca['child_or_acquirer'];i=[k for k,m in enumerate(MEL) if m>=dproc][0] if dproc else None
            vr=ns['verify_post'](acq,dproc,MEL[i:i+4]) if dproc else None
            ok=vr is not None and dproc in vr[1].index
            r.update(action='apply' if ok else 'fail',reason=('stock deal: ratio x acquirer shares (+cash); follow acq:'+acq) if ok else 'acquirer series unverified -> path UNKNOWN corporate_action_unresolved',
                     acquirer_series='acq:'+acq,acquirer_verified=ok)
        elif ty=='unresolved': r.update(action='fail',reason='unresolved corporate action while held -> path UNKNOWN corporate_action_unresolved')
        else: r.update(action='ignore',reason=f'type {ty} has no holding-side leg in the engine')
        EV.append(r)
pd.DataFrame(EV).to_csv(A+'holding_events.csv',index=False)
el=pd.read_csv(A+'eligibility_by_signal.csv.gz');U=sorted(el[el.eligible_final].engine_ticker.unique())
pd.DataFrame([dict(engine_ticker=t,deal_date=e) for t in U for e in ns['_dealdate'](t)]).to_csv(A+'untradable_deal_dates.csv',index=False)
P=pd.DataFrame([r for p in parts for r in p['PATHS']]).sort_values(['W','H','K','phase']);P.to_csv(B+'paths_summary.csv',index=False)
pd.DataFrame([r for p in parts for r in p['REB']]).to_csv(B+'rebalances.csv.gz',index=False,compression='gzip')
TRD=pd.DataFrame([r for p in parts for r in p['TRD']])
# fill the price of a stock-deal recipient received and sold on the same rebalance day (never in an end-of-day snapshot)
CLD={(a,b,d):float(v) for a,b,d,v in CL[CL.price_series.str.startswith('acq:')][['price_series','execution_key','date','close']].itertuples(index=False)}
m=TRD.event.eq('rebalance')&TRD.price.isna()
for i in TRD.index[m]:
    k=('acq:'+TRD.at[i,'execution_key'],TRD.at[i,'execution_key'],TRD.at[i,'trade_date'])
    if k in CLD:
        p=CLD[k];TRD.at[i,'price']=p;dv=(TRD.at[i,'shares_after']-TRD.at[i,'shares_before'])*p;TRD.at[i,'trade_value']=dv;TRD.at[i,'fee']=ns['c']*abs(dv)
assert TRD[m].price.notna().all(),TRD[m&TRD.price.isna()].head()
R=pd.DataFrame([r for p in parts for r in p['REB']]);fs=TRD[TRD.event.eq('rebalance')].groupby(['path_id','trade_date']).fee.sum()
chk=R.set_index(['path_id','trade_date']).fee;dd=(fs.reindex(chk.index)-chk).abs()/R.set_index(['path_id','trade_date']).equity_before
print('trade-ledger fee sum vs engine fee: max rel diff',float(dd.max()))
TRD.to_csv(B+'trades.csv.gz',index=False,compression='gzip')
def cell(x):
    k=x[x.status!='UNKNOWN']
    return pd.Series(dict(n_phases=len(x),n_known=len(k),n_unknown=len(x)-len(k),median_CAGR=k.CAGR.median(),median_MDD=k.daily_MDD.median(),min_CAGR=k.CAGR.min(),max_CAGR=k.CAGR.max(),worst_MDD=k.daily_MDD.min(),all_12_known=len(k)==12))
P.groupby(['W','H','K']).apply(cell).reset_index().to_csv(B+'cells_12phase_medians.csv',index=False)
NAV={k:v for p in parts for k,v in p['NAV'].items()}
pickle.dump(NAV,open('/workspace/m432full-scratch/NAV.pkl','wb'))
print('A closes',len(CL),'exa',len(EXA),'events',len(EV),'B paths',len(P),P.status.value_counts().to_dict(),'nav paths',len(NAV))
