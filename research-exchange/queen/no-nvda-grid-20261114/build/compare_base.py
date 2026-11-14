import pickle,glob,gzip,csv,collections,numpy as np,pandas as pd
B='/workspace/momentum432-answer-key-full-20261109/'
def ld(m):
    P=[];RB=[];NAV={}
    for f in sorted(glob.glob(f'scratch/{m}_*.pkl')):
        d=pickle.load(open(f,'rb'));P+=d['P'];RB+=d['RB'];NAV.update(d['NAV'])
    return pd.DataFrame(P).set_index('path_id'),pd.DataFrame(RB),NAV
P,RB,NAV=ld('base');R=pd.read_csv(B+'paths_summary.csv').set_index('path_id')
P=P.loc[R.index];print(len(P),(P.status==R.status).all())
k=R.status!='UNKNOWN'
for a,b in (('final_multiple','final_multiple'),('CAGR','CAGR'),('daily_MDD','daily_MDD')):
    print(a,float(np.nanmax(np.abs(P.loc[k,a].astype(float)-R.loc[k,b]))))
u=~k;print('unknown fail match',((P.loc[u,'fail_reason']==R.loc[u,'fail_reason'])&(P.loc[u,'fail_ticker'].astype(str)==R.loc[u,'fail_ticker'].astype(str))&(P.loc[u,'fail_date'].astype(str)==R.loc[u,'fail_date'].astype(str))).all())
rr=collections.defaultdict(list)
for r in csv.DictReader(gzip.open(B+'rebalances.csv.gz','rt')): rr[r['path_id']].append((r['signal_date'],r['trade_date'],r['picks']))
mine=collections.defaultdict(list)
for r in RB.itertuples(): mine[r.path_id].append((r.signal_date,r.trade_date,str(r.picks)))
bad=[p for p in rr if mine[p][:len(rr[p])]!=rr[p]];print('rebalance rows',sum(map(len,rr.values())),'path pick mismatches',len(bad),bad[:5])
print(P.status.value_counts().to_dict())
print('W12H12K1P1',P.loc['W12H12K1P1','CAGR'])
for c in ('W10H2K1','W12H4K1','W12H12K1'):
    x=P[P.index.str.startswith(c+'P')&(P.status!='UNKNOWN')];print(c,x.CAGR.median(),len(x))
