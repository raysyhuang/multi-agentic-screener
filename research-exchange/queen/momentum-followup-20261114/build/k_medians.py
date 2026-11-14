"""Median CAGR of known paths by K: answer key (with NVDA) and no-NVDA grid. SPY price-only 2015-2025 = 11.02% for the below-SPY share."""
import pandas as pd
rows=[]
for name,f in (('with NVDA','/workspace/momentum432-answer-key-full-20261109/paths_summary.csv'),('no NVDA','/workspace/no-nvda-grid-20261114/paths_summary_no_nvda.csv')):
    D=pd.read_csv(f);D=D[D.status!='UNKNOWN']
    for k,g in [('K1',D[D.K==1]),('K2',D[D.K==2]),('K3',D[D.K==3]),('all',D)]:
        cm=g.groupby(['W','H','K']).CAGR.median()
        rows.append(dict(set=name,K=k,known_paths=len(g),median_CAGR=g.CAGR.median(),mean_CAGR=g.CAGR.mean(),share_below_SPY_px_11_02=(g.CAGR<0.1102).mean(),median_of_cell_medians=cm.median()))
R=pd.DataFrame(rows);R.to_csv('/workspace/momentum-followup-20261114/results/k_medians.csv',index=False);print(R.round(4).to_string())
