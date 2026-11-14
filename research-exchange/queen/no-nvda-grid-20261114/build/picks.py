import pickle,glob,pandas as pd
P=[];RB=[];NAV={}
for f in sorted(glob.glob('scratch/nonvda_*.pkl')):
    d=pickle.load(open(f,'rb'));P+=d['P'];RB+=d['RB'];NAV.update(d['NAV'])
P=pd.DataFrame(P).set_index('path_id');RB=pd.DataFrame(RB);S=pd.read_csv('/workspace/momentum432-inputs-pack-full-20261109/trading_sessions.csv').session.tolist()
out=[]
for pid in ['W12H12K1P1','W11H9K1P9','W10H9K1P8','W12H12K2P1','W11H10K1P1']:
    r=RB[RB.path_id==pid].copy();e=P.loc[pid];eq=NAV[pid];i0=S.index(e.entry)
    v=lambda d:eq[S.index(d)-i0]
    ds=list(r.trade_date)+[e.end]
    r['equity_at_trade_close']=[v(d) for d in r.trade_date];r['next_date']=ds[1:]
    r['period_growth_pct']=[(v(b)/v(a)-1)*100 for a,b in zip(ds[:-1],ds[1:])]
    out.append(r)
    print(pid,e.CAGR,e.daily_MDD);print(r[['signal_date','trade_date','next_date','picks','period_growth_pct']].round(1).to_string(index=False))
pd.concat(out).to_csv('rebalance_picks_best_no_nvda.csv',index=False)
