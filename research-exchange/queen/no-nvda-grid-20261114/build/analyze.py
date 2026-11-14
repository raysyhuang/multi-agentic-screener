import pickle,glob,numpy as np,pandas as pd
def ld(m):
    P=[];RB=[]
    for f in sorted(glob.glob(f'scratch/{m}_*.pkl')):
        d=pickle.load(open(f,'rb'));P+=d['P'];RB+=d['RB']
    return pd.DataFrame(P),pd.DataFrame(RB)
P,RB=ld('nonvda');B,BR=ld('base')
st={'DATA_COMPLETE_PROVISIONAL_C':'PROVISIONAL_C','DATA_COMPLETE_CONDITIONAL':'CONDITIONAL','UNKNOWN':'UNKNOWN'}
P['status']=P.status.map(st);B['status']=B.status.map(st)
P=P.sort_values(['W','H','K','phase']);P.to_csv('paths_summary_no_nvda.csv',index=False)
RB.to_csv('rebalances_no_nvda.csv.gz',index=False,compression='gzip')
def cell(x):
    k=x[x.status!='UNKNOWN']
    return pd.Series(dict(n_phases=len(x),n_known=len(k),n_unknown=len(x)-len(k),median_CAGR=k.CAGR.median(),median_MDD=k.daily_MDD.median(),min_CAGR=k.CAGR.min(),max_CAGR=k.CAGR.max(),worst_MDD=k.daily_MDD.min(),all_12_known=len(k)==12))
C=P.groupby(['W','H','K']).apply(cell).reset_index().sort_values('median_CAGR',ascending=False);C.to_csv('cells_12phase_medians_no_nvda.csv',index=False)
pd.set_option('display.width',250);pd.set_option('display.max_columns',30)
f=lambda d:d.assign(**{c:(d[c]*100).round(2) for c in ['median_CAGR','median_MDD','min_CAGR','max_CAGR','worst_MDD'] if c in d})
print('TOP10 cells (known phases)');print(f(C.head(10)).to_string(index=False))
print('TOP10 cells all 12 known');print(f(C[C.all_12_known].head(10)).to_string(index=False))
T=P[P.status!='UNKNOWN'].sort_values('CAGR',ascending=False).head(10)
print('TOP10 paths');print(T[['path_id','CAGR','daily_MDD','final_multiple','status']].assign(CAGR=lambda d:(d.CAGR*100).round(2),daily_MDD=lambda d:(d.daily_MDD*100).round(2)).to_string(index=False))
Pi=P.set_index('path_id')
print('anchor W12H12K1P1',Pi.loc['W12H12K1P1',['status','CAGR','daily_MDD','final_multiple']].to_dict())
for c in ('W10H2K1','W12H4K1','W12H12K1'):
    x=P[P.path_id.str.startswith(c+'P')];k=x[x.status!='UNKNOWN'];print(c,'median',k.CAGR.median(),'known',len(k),'medMDD',k.daily_MDD.median(),'min',k.CAGR.min(),'max',k.CAGR.max())
print('status',P.status.value_counts().to_dict());print('fail reasons',P.fail_reason.value_counts().to_dict())
Bi=B.set_index('path_id');newU=Pi[(Pi.status=='UNKNOWN')&(Bi.loc[Pi.index,'status']!='UNKNOWN')];goneU=Pi[(Pi.status!='UNKNOWN')&(Bi.loc[Pi.index,'status']=='UNKNOWN')]
print('new UNKNOWN',len(newU));print(newU.groupby(['fail_reason','fail_ticker','fail_date']).size().to_string())
print('UNKNOWN->known',len(goneU));print(goneU.index.tolist()[:40])
newU.reset_index().to_csv('new_unknowns_vs_base.csv',index=False)
# paths unchanged (never picked NVDA)
br=BR.groupby('path_id').picks.apply(lambda s:any('NVDA' in str(p).split('|') for p in s))
print('base paths that ever held NVDA',int(br.sum()),'of',len(br))
