"""Per-cell match/mismatch: Claude 5286b46 README/breadth_results (THIRD PARTY) vs Queen exact engine (strict if 12/12 tranches known,
else S1/S1t sensitivity, labelled). CAGR tolerance: |diff|<=1.0pp MATCH, <=3.0pp CLOSE, else MISMATCH. MDD: Claude month-end vs Queen month-sampled."""
import pandas as pd
B='/workspace/momentum-followup-20261114/'
C=pd.read_csv(B+'claude_5286b46/breadth_results.csv');C=C[(C.H==12)]
Q=pd.read_csv(B+'results/fixed_rule/fixed_rule_summary.csv')
rows=[]
def pick(period,univ,N):
    q=Q[(Q.period==period)&(Q.universe==univ)&(Q.N==N)]
    s=q[(q.variant=='exact_engine_strict')&(q.known_tranches==12)]
    if len(s): return s.iloc[0],'exact_engine_strict'
    v='s1t_sensitivity_tail_cashout' if period=='2005-2015' else 's1_sensitivity_cashout'
    return q[q.variant==v].iloc[0],v
for _,c in C.iterrows():
    period,univ=c.decade,c.universe
    if c.signal=='EW universe':
        rows.append(dict(period=period,universe=univ,cell='EW',claude_CAGR=c.CAGR,queen_CAGR=Q[(Q.period==period)&(Q.universe==univ)].EW_proxy_CAGR.dropna().iloc[0],queen_source='Claude proxy code rerun (EW not run on exact engine)'));continue
    if c.signal!='W10-12': continue
    N=int(c.K);q,src=pick(period,univ,N)
    rows.append(dict(period=period,universe=univ,cell=f'Top{N}',claude_CAGR=c.CAGR,queen_CAGR=q.CAGR,claude_MDD_monthend=c.monthend_MDD,queen_MDD_monthly=q.monthly_MDD,queen_MDD_daily=q.daily_MDD,
        claude_active_pp=c.active_vs_EW_per_year*100,queen_active_pp=q.excess_vs_EW_mean_pp_yr,claude_t=c.active_t_stat,queen_t_simple=q.t_simple_vs_EW,queen_t_NW6=q.t_NW6_vs_EW,queen_boot_p=q.boot_p_vs_EW,
        claude_tranche_range=f'{c.tranche_min_CAGR*100:.1f}..{c.tranche_max_CAGR*100:.1f}',queen_tranche_range=f'{q.tranche_min_CAGR*100:.1f}..{q.tranche_max_CAGR*100:.1f}',queen_source=src))
D=pd.DataFrame(rows);d=(D.queen_CAGR-D.claude_CAGR).abs()*100
D['CAGR_diff_pp']=(D.queen_CAGR-D.claude_CAGR)*100;D['verdict']=['MATCH' if x<=1 else ('CLOSE' if x<=3 else 'MISMATCH') for x in d]
D.to_csv(B+'results/fixed_rule/claude_match.csv',index=False)
pd.set_option('display.width',250);print(D[['period','universe','cell','claude_CAGR','queen_CAGR','CAGR_diff_pp','verdict','claude_MDD_monthend','queen_MDD_monthly','queen_MDD_daily','claude_active_pp','queen_active_pp','claude_t','queen_t_NW6','queen_source']].round(3).to_string())
