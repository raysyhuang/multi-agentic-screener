"""Claim 3: cross-decade stability of the 432-grid ranking (K1 primary).
A) engine cells: 2015-2025 = official answer key; 2005-2015 = engine_ext grid (strict / approx_delist), median over known phases.
B) month-end proxy (no daily data needed, full coverage): hold top-1 from signal month-end S to the next signal month-end
   at rank prices (spin F applied), 25 bp per side on changes; CAGR over the same 10-year phase windows. Used only as a
   robustness check of the rank correlation; validated against the official 2015-2025 cells."""
import sys,csv,math,statistics as st,collections,datetime as dt;sys.path.insert(0,'.')
import pandas as pd
from scipy.stats import spearmanr
from engine_ext import Eng
ak=pd.read_csv('/workspace/momentum432-answer-key-full-20261109/cells_12phase_medians.csv')
def cells(g):
    k=g[g.status!='UNKNOWN'].groupby(['W','H','K']).agg(n_known=('CAGR','size'),median_CAGR=('CAGR','median'),median_MDD=('MDD','median')).reset_index()
    return k
def proxy(E):
    rows=[]
    for W in range(1,13):
      for H in range(1,13):
        for ph in range(1,13):
            D0=E.first_td(E.y0,ph);END=E.first_td(E.y0+10,ph);rebs=[];j=0
            while True:
                mm=(ph-1)+j*H;d=E.first_td(E.y0+mm//12,mm%12+1)
                if d>=END: break
                rebs.append(d);j+=1
            sig=[E.SES[E.SI[d]-1] for d in rebs]+[E.SES[E.SI[END]-1]]
            nav=1.0;prev=None;ok=True;why='';nflag=[0]
            for i in range(len(rebs)):
                S,S2=sig[i],sig[i+1];r,miss,_=E.rank(S,W)
                if miss or not r: ok=False;why='rank_missing';break
                t=r[0][1]
                if t!=prev: nav*=(1-0.0025)*(1-0.0025 if prev else 1)  # sell old (if any) + buy new
                a=E.px(t,S,S);b=E.px(t,S2,S2)
                if a and not b:   # proxy approximation: pick stopped pricing (deal/delisting) -> cash at its last month-end rank price
                    ms=[m for m in E.ME if S<m<S2 and E.px(t,m,m)]
                    if ms: b=E.px(t,ms[-1],ms[-1]);nflag[0]+=1
                if not a or not b: ok=False;why='price_missing:'+t+'@'+S2;break
                for e in E.CA[t]:
                    if e['kind'] in('spin','special_div') and S<e['ex_date']<=S2:
                        if not e['factor_F']: ok=False;why='F_missing';break
                        b/=float(e['factor_F'])
                if not ok: break
                nav*=b/a;prev=t
            if ok:
                nav*=(1-0.0025);yrs=(dt.date.fromisoformat(END)-dt.date.fromisoformat(D0)).days/365.25
                rows.append(dict(W=W,H=H,K=1,phase=ph,CAGR=nav**(1/yrs)-1,status='OK',n_lastprice_cashouts=nflag[0]))
            else: rows.append(dict(W=W,H=H,K=1,phase=ph,CAGR=None,status='UNKNOWN',why=why))
    return pd.DataFrame(rows)
out=[];detail=[]
A15=ak[ak.K==1][['W','H','K','n_known','median_CAGR','median_MDD']]
def compare(lbl,a15,a05,minn):
    m=a15.merge(a05,on=['W','H','K'],suffixes=('_15','_05'))
    m=m[(m.n_known_05>=minn)&(m.n_known_15>=minn)].copy()
    rho,p=spearmanr(m.median_CAGR_15,m.median_CAGR_05)
    m['rank15']=m.median_CAGR_15.rank(ascending=False,method='min').astype(int);m['rank05']=m.median_CAGR_05.rank(ascending=False,method='min').astype(int)
    n=len(m);top10=m.nsmallest(10,'rank15')
    def at(W,H):
        x=m[(m.W==W)&(m.H==H)]
        return (int(x.rank15.iloc[0]),int(x.rank05.iloc[0]),float(x.median_CAGR_05.iloc[0])) if len(x) else None
    out.append(dict(comparison=lbl,min_known_phases=minn,n_cells=n,spearman_rho=rho,p_value=p,
        W10H2K1=at(10,2),W12H4K1=at(12,4),W12H12K1=at(12,12),
        top10_2015_rank05=' '.join(f'W{r.W}H{r.H}:{r.rank05}' for r in top10.itertuples()),
        top10_2015_median_rank05=float(top10.rank05.median()),top10_in_bottom_half_05=int((top10.rank05>n/2).sum()),
        top10_2015_median_CAGR05=float(top10.median_CAGR_05.median())))
    m.insert(0,'comparison',lbl+f'|min{minn}');detail.append(m)
for tag in ('strict','approx'):
    g=pd.read_csv(f'grid_paths_2005_{tag}.csv');c=cells(g[g.K==1])
    for minn in (6,12): compare(f'engine K1: official 2015-25 vs 2005-15 {tag}',A15,c,minn)
E15=Eng('/workspace/momentum432-inputs-pack-full-20261109',2015);E05=Eng('/workspace/w10h2-w12h4-oos-2005-2015-20261112/pack',2005)
p15=proxy(E15);p05=proxy(E05);p15.to_csv('claim3_proxy_paths_2015.csv',index=False);p05.to_csv('claim3_proxy_paths_2005.csv',index=False)
pc15=cells(p15.rename(columns={'CAGR':'CAGR'}).assign(MDD=0.0));pc05=cells(p05.assign(MDD=0.0))
# proxy validity in 2015: proxy cell medians vs official
v=A15.merge(pc15,on=['W','H','K'],suffixes=('_off','_proxy'));rv,_=spearmanr(v.median_CAGR_off,v.median_CAGR_proxy,nan_policy='omit')
print('proxy-vs-official 2015 K1 cell spearman',rv,len(v))
for minn in (6,12): compare('month-end proxy K1: 2015-25 vs 2005-15',pc15,pc05,minn)
compare('official 2015-25 vs month-end proxy 2005-15',A15,pc05,12)
res=pd.DataFrame(out);res['proxy_vs_official_2015_rho']=rv;res.to_csv('claim3_rank_correlation.csv',index=False)
pd.concat(detail).to_csv('claim3_cell_ranks_detail.csv',index=False)
# all-K table where available (strict & approx)
for tag in ('strict','approx'):
    g=pd.read_csv(f'grid_paths_2005_{tag}.csv');c=cells(g)
    m=ak[['W','H','K','n_known','median_CAGR']].merge(c,on=['W','H','K'],suffixes=('_15','_05'))
    for minn in (6,12):
        x=m[(m.n_known_05>=minn)&(m.n_known_15>=minn)]
        print(tag,'allK min',minn,'n',len(x),'rho',spearmanr(x.median_CAGR_15,x.median_CAGR_05)[0] if len(x)>2 else None, x.K.value_counts().to_dict())
pd.set_option('display.width',250);pd.set_option('display.max_colwidth',300)
print(res.T.to_string())
