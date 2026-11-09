"""Export reconciliation tables for W10H2K1 and W12H4K1, 12 phases, from the frozen v17 strict4 engine (read-only).
No engine logic is changed; a snapshot hook only observes state. Research only."""
import bisect,json,os,numpy as np,pandas as pd
from load_engine import load,V17
OUT=os.path.dirname(os.path.abspath(__file__))
ns=load();c=ns['c']
CFG=[(10,2),(12,4)]
def members_at(S):
    mem,md=ns['mem'],ns['md'];k=bisect.bisect_right(md,S)-1;m=mem[k]
    return set(m[1]) if isinstance(m,tuple) else set(m)
def miss_reason(t,S,W):
    B=ns['MEL'][ns['MEL'].index(S)-W]
    if any(t==a and b<=S<=e for a,b,e in ns['SKIPL']): return 'legacy_data_error_exclusion(SKIPL)'
    ur=[x['id'] for x in ns['ca_of'](t) if x['type']=='unresolved' and B<x['ex_date']<=S]
    if ur: return 'unresolved_corporate_action_in_lookback:'+' '.join(ur)
    a,b=ns['mpx'](B,t,S),ns['mpx'](S,t,S)
    if not a: return f'no_verified_price_at_lookback_start {B}'
    if not b: return f'no_verified_price_at_signal {S}'
    return 'spin_factor_price_unavailable_or_nonpositive'
RK=[];TR=[];EQ=[];SM=[];DAILY=[]
for W,H in CFG:
    cfg=f'W{W}H{H}K1'
    for ph in range(1,13):
        snaps=[];SER={}
        def hook(w,h,k,p,d,val,cash,hold,terminal):
            for t,(n,nm,s,last) in hold.items(): SER[t]=s
            snaps.append(dict(date=d,equity=float(val),cash=float(cash) if not terminal else float(val),
                              shares={t:float(v[0]) for t,v in hold.items()},close={t:float(v[3]) for t,v in hold.items()},terminal=terminal))
        ns['_snapshot_hook']=hook
        row,eq,led=ns['path_strict4'](W,H,1,ph)
        ns.pop('_snapshot_hook')
        SM.append(dict(config=cfg,phase=ph,entry=row['entry'],end=row['end'],status=row['status'],unknown_reason=row.get('fail_reason') or '',
            unknown_ticker=row.get('fail_ticker') or '',unknown_date=row.get('fail_date') or '',final_multiple=row['wealth'],CAGR=row['CAGR'],daily_MDD=row['daily_MDD'],
            n_rebalance_dates=row['n_rebal'],grade_C_events_touched=row.get('grade_C_events',''),n_events=row.get('n_events')))
        if row['status']=='UNKNOWN': continue
        bydate={s['date']:i for i,s in enumerate(snaps)}
        for s in snaps:
            EQ.append(dict(config=cfg,phase=ph,date=s['date'],equity=s['equity'],cash=s['cash'],
                holdings=';'.join(f'{t}:{n:.10g}@{s["close"][t]}' for t,n in s['shares'].items()) if not s['terminal'] else 'LIQUIDATED'))
        # ---- trades
        for x in led:
            d=x['date'];i=bydate[d];post=snaps[i];pre=snaps[i-1] if i>0 else dict(shares={},close={},equity=1.0,cash=1.0)
            base=dict(config=cfg,phase=ph,trade_date=d,event=x['event'],signal_date=x.get('signal',''))
            if x['event']=='rebalance':
                Eafter=x['E_after'];fees=0.0
                for t in dict.fromkeys(list(pre['shares'])+list(post['shares'])):
                    n0=pre['shares'].get(t,0.0);n1=post['shares'].get(t,0.0) if t in post['shares'] else 0.0
                    px=float(SER[t][d]);dv=(n1-n0)*px
                    if t not in pre['shares']: act='buy'
                    elif t not in post['shares']: act='sell'
                    elif abs(n1-n0)<=1e-12*max(n0,1e-30): act='hold'
                    else: act='rebalance_buy' if n1>n0 else 'rebalance_sell'
                    fee=c*abs(dv) if act!='hold' else 0.0;fees+=fee
                    TR.append(dict(base,ticker=t,action=act,weight_after=(n1*px/Eafter) if t in post['shares'] else 0.0,price=px,
                        shares_traded=n1-n0,shares_after=n1,trade_value=dv,fee=fee,equity_before=x['E_before'],equity_after=Eafter,ca_id='',note='close execution'))
                assert abs(fees-x['fee'])<=1e-9*max(1,x['E_before']),(cfg,ph,d,fees,x['fee'])
            elif x['event']=='terminal_liquidation':
                for t,n in pre['shares'].items() if False else post['shares'].items():
                    px=post['close'][t];TR.append(dict(base,ticker=t,action='sell',weight_after=0.0,price=px,shares_traded=-n,shares_after=0.0,
                        trade_value=-n*px,fee=c*n*px,equity_before=x['turnover']+x['cash_before'],equity_after=x['E_after'],ca_id='',note='terminal liquidation at close'))
            elif x['event']=='cash_deal':
                n=pre['shares'].get(x['ticker']);TR.append(dict(base,ticker=x['ticker'],action='corporate_action_cash_out',weight_after=0.0,price=x['cash']/n if n else None,
                    shares_traded=-n if n else None,shares_after=0.0,trade_value=-x['cash'],fee=0.0,equity_before=None,equity_after=post['equity'],ca_id=x['ca'],note='issuer cash consideration, no fee; evidence '+x['evidence']))
            elif x['event']=='spin':
                TR.append(dict(base,ticker=x['ticker']+'->child',action='spin_child_sold',weight_after=0.0,price=None,shares_traded=None,shares_after=0.0,
                    trade_value=-x['child_gross'],fee=x['fee'],equity_before=None,equity_after=post['equity'],ca_id=x['ca'],note='child valued at first regular-way close and sold, 25bps; net cash kept until next rebalance; evidence '+x['evidence']))
            elif x['event']=='stock_deal':
                TR.append(dict(base,ticker=x['ticker'],action='corporate_action_exchange_out',weight_after=0.0,price=None,shares_traded=None,shares_after=0.0,trade_value=None,fee=0.0,
                    equity_before=None,equity_after=post['equity'],ca_id=x['ca'],note=f"exchanged into {x['to']}, no fee"))
                TR.append(dict(base,ticker=x['to'],action='corporate_action_exchange_in',weight_after=None,price=post['close'].get(x['to']),shares_traded=x['shares'],
                    shares_after=post['shares'].get(x['to']),trade_value=None,fee=0.0,equity_before=None,equity_after=post['equity'],ca_id=x['ca'],note='received shares; cash '+str(x['cash'])))
            else: raise RuntimeError(x)
        # ---- rankings for every executed signal of this path
        for x in led:
            if x['event']!='rebalance': continue
            S=x['signal'];r,miss,nu=ns['RS'][(S,W)];sel=x['execution_keys'].split('|')
            cut=r[min(9,len(r)-1)][0] if r else None
            for i,(g,t) in enumerate(r[:max(10,1)]):
                RK.append(dict(config=cfg,phase=ph,signal_date=S,trade_date=x['date'],rank=i+1,ticker=t,lookback_return=g,eligible=True,ranked=True,
                    selected=t in x['picks'].split('|'),exclusion_reason=''))
            for t in miss:
                RK.append(dict(config=cfg,phase=ph,signal_date=S,trade_date=x['date'],rank=None,ticker=t,lookback_return=None,eligible=True,ranked=False,selected=False,
                    exclusion_reason='unrankable: '+miss_reason(t,S,W)))
            B=ns['MEL'][ns['MEL'].index(S)-W]
            for t in sorted(members_at(S)-ns['seasoned'](S)):
                try: a,b=ns['mpx'](B,t,S),ns['mpx'](S,t,S)
                except Exception: a=b=None
                g=(b/a-1) if a and b and a>0 else None
                if g is not None and cut is not None and g<cut: continue   # only non-eligible members that would have entered the top 10
                RK.append(dict(config=cfg,phase=ph,signal_date=S,trade_date=x['date'],rank=None,ticker=t,lookback_return=g,eligible=False,ranked=False,selected=False,
                    exclusion_reason='index member but not continuous 365 days at signal date'+('' if g is not None else '; lookback return not computable (no verified price at lookback start)')))
        print(cfg,ph,row['status'],round(row['CAGR'],6),flush=True)
sm=pd.DataFrame(SM)
# compare with frozen outputs
v17=pd.read_csv(V17/'full_coverage/paths_status.csv');v12=pd.read_csv('/workspace/momentum432-engine-20261108/out_strict4/paths_status.csv')
def look(df,W,H,ph,col):
    z=df[(df.W==W)&(df.H==H)&(df.K==1)&(df.phase==ph)];return float(z[col].iloc[0]) if len(z) else None
for i,r in sm.iterrows():
    W,H=(10,2) if r.config.startswith('W10') else (12,4)
    sm.loc[i,'v17_frozen_CAGR']=look(v17,W,H,r.phase,'CAGR');sm.loc[i,'v17_frozen_MDD']=look(v17,W,H,r.phase,'daily_MDD')
    sm.loc[i,'v12_sealed_CAGR']=look(v12,W,H,r.phase,'CAGR');sm.loc[i,'v12_sealed_MDD']=look(v12,W,H,r.phase,'daily_MDD')
sm['abs_diff_vs_v17_CAGR']=(sm.CAGR-sm.v17_frozen_CAGR).abs();sm['abs_diff_vs_v17_MDD']=(sm.daily_MDD-sm.v17_frozen_MDD).abs()
sm.to_csv(f'{OUT}/summary.csv',index=False)
pd.DataFrame(RK).to_csv(f'{OUT}/rankings.csv',index=False)
pd.DataFrame(TR).to_csv(f'{OUT}/trades.csv',index=False)
pd.DataFrame(EQ).to_csv(f'{OUT}/daily_equity.csv',index=False)
med=[]
for cfg,g in sm.groupby('config'):
    med.append(dict(config=cfg,n_phases=len(g),n_unknown=int((g.status=='UNKNOWN').sum()),v17_median_CAGR=g.CAGR.median(),v17_min_CAGR=g.CAGR.min(),v17_max_CAGR=g.CAGR.max(),
        v17_median_MDD=g.daily_MDD.median(),v17_worst_MDD=g.daily_MDD.min(),v12_median_CAGR=g.v12_sealed_CAGR.median(),v12_median_MDD=g.v12_sealed_MDD.median(),
        max_abs_diff_vs_v17_frozen_CAGR=g.abs_diff_vs_v17_CAGR.max()))
pd.DataFrame(med).to_csv(f'{OUT}/medians_v12_vs_v17.csv',index=False);print(pd.DataFrame(med).T)
