"""2005-2015 full grid summary (exact pack engine, PROVISIONAL membership) + consistency vs the frozen 25-path packet."""
import pandas as pd,numpy as np
B='/workspace/momentum-followup-20261114/';O=B+'results/r0515/'
D=pd.read_csv(O+'paths_0515_all.csv');D['known']=D.status!='UNKNOWN'
G=D[D.K<=3]
print('grid paths',len(G),G.status.value_counts().to_dict())
print('fail reasons',G[~G.known].fail_reason.value_counts().to_dict())
print('top fail tickers',G[~G.known].fail_ticker.astype(str).str.split().str[0].value_counts().head(15).to_dict())
def cells(X):
    g=X.groupby(['W','H','K'])
    c=g.apply(lambda x:pd.Series(dict(known=int(x.known.sum()),median_CAGR=x[x.known].CAGR.median(),min_CAGR=x[x.known].CAGR.min(),max_CAGR=x[x.known].CAGR.max(),
        median_MDD=x[x.known].MDD.median(),worst_MDD=x[x.known].MDD.min(),
        median_low_bound=np.median(np.r_[x[x.known].CAGR.values,[-1]*int((~x.known).sum())]) if x.known.any() else np.nan,
        median_high_bound=np.median(np.r_[x[x.known].CAGR.values,[9]*int((~x.known).sum())]) if x.known.any() else np.nan)),include_groups=False).reset_index()
    return c
C=cells(G);C.to_csv(O+'cells_12phase_0515.csv',index=False)
print('cells',len(C),'cells 12/12 known',int((C.known==12).sum()),'cells 0 known',int((C.known==0).sum()))
full=C[C.known==12].sort_values('median_CAGR',ascending=False)
print('TOP 15 cells (12/12 known):\n',full.head(15).round(4).to_string())
print('TOP 10 cells (>=9/12 known):\n',C[C.known>=9].sort_values('median_CAGR',ascending=False).head(10).round(4).to_string())
print('median of 12/12 cell medians',full.median_CAGR.median(),'share of known paths CAGR>0',(G[G.known].CAGR>0).mean(),'median known path',G[G.known].CAGR.median())
print('by K median known path',G[G.known].groupby('K').CAGR.median().round(4).to_dict())
W=C[C.W.isin([10,11,12])&C.H.isin([6,12])];W.to_csv(O+'cells_W10_12_H6_H12_0515.csv',index=False);print('W10-12 x H6/H12:\n',W.round(4).to_string())
# consistency vs frozen 25 paths
F=pd.read_csv('/workspace/w10h2-w12h4-oos-2005-2015-20261112/results/results_paths.csv')
M=F.merge(D,on=['W','H','K','phase'],suffixes=('_frozen','_new'))
M['dCAGR']=M.CAGR_new-M.CAGR_frozen
M[['path','status_frozen','status_new','CAGR_frozen','CAGR_new','dCAGR','fail_reason_frozen','fail_reason_new','fail_ticker_new']].to_csv(O+'consistency_vs_frozen_25paths.csv',index=False)
print('consistency: n',len(M),'status equal',int((M.status_frozen==M.status_new).sum()),'max |dCAGR|',M.dCAGR.abs().max())
print(M[['path','status_frozen','status_new','CAGR_frozen','CAGR_new','fail_ticker_new']].to_string())
