#!/usr/bin/env python3
"""Independent recompute of third-party claims on US momentum band W10-12 x H2-4.
Source of truth (read-only): Queen v18 answer key /workspace/momentum432-answer-key-full-20261109/
SPY: spy_daily.csv (FMP dividend-adjusted close, saved once; no network here).
Conventions:
  * CAGR years = calendar days / 365.25 (same as official paths_summary; verified).
  * NAV before entry = 1.0 (NAV on entry date already includes buy fee). Engine trades at close.
  * Day t log return ln(NAV_t/NAV_{t-1}) belongs to the holding carried over (t-1, t].
    On a rebalance day: market part ln(equity_before/NAV_{t-1}) -> outgoing holding; fee part
    ln(equity_after/equity_before) split pro-rata between sell-leg fees (outgoing) and buy-leg fees
    (incoming), using trades.csv. Terminal-liquidation fee -> held name.
  * Calendar year Y return = NAV(last session <= Y-12-31) / NAV(last session <= (Y-1)-12-31) - 1;
    if the path starts in Y, base = 1.0 at entry close; if it ends in Y, end = final NAV. SPY over the same dates.
"""
import numpy as np, pandas as pd, os
AK='/workspace/momentum432-answer-key-full-20261109/'
OUT=os.path.dirname(os.path.abspath(__file__))+'/'
P=pd.read_csv(AK+'paths_summary.csv')
P['days']=(pd.to_datetime(P.end)-pd.to_datetime(P.entry)).dt.days
spy=pd.read_csv(OUT+'spy_daily.csv').set_index('date').adjClose
def spy_on(d):  # last SPY close on/before date
    return spy.loc[:d].iloc[-1]
def cagr(mult,days): return mult**(365.25/days)-1

band=lambda Ws,K: P[P.W.isin(Ws)&P.H.isin([2,3,4])&(P.K==K)]
B1,B2,B3=band([10,11,12],1),band([10,11,12],2),band([10,11,12],3)
W9={k:band([9],k) for k in (1,2,3)}
rows=[]  # comparison rows
def add(claim,item,theirs,ours,match,note=''): rows.append(dict(claim=claim,item=item,their_number=theirs,queen_recompute=ours,match=match,convention_note=note))
pc=lambda x:f'{x*100:.2f}%'
def m1(ours,theirs,tol=0.0005+1e-9):  # one-decimal rounding match on percent
    return 'MATCH' if abs(round(ours*100,1)-theirs)<1e-9 else ('NEAR (±0.1pp)' if abs(ours*100-theirs)<=0.15 else 'MISMATCH')

# ---------- NAV + holdings for K1 W9..12 H2-4 ----------
need=pd.concat([B1,W9[1]]).path_id.tolist()
NAV={}
for w in (9,10,11,12):
    n=pd.read_csv(AK+f'nav_daily/nav_daily_W{w:02d}.csv.gz',index_col=0)
    for c in n.columns:
        if c in need: NAV[c]=n[c].dropna()
R=pd.read_csv(AK+'rebalances.csv.gz'); R=R[R.path_id.isin(need)]
T=pd.read_csv(AK+'trades.csv.gz',usecols=['path_id','trade_date','event','execution_key','action','fee']); T=T[T.path_id.isin(need)]
ca=T[T.event.isin(['cash_deal','stock_deal','spin'])]
ca[['path_id','trade_date','event','execution_key']].drop_duplicates().to_csv(OUT+'k1_band_corporate_action_events.csv',index=False)
T['side']=np.where(T.action.isin(['buy','rebalance_buy']),'buy','sell')
FEE=T.groupby(['path_id','trade_date','side']).fee.sum().unstack(fill_value=0)

def daily_attr(pid):
    nav=NAV[pid]; r=R[R.path_id==pid].sort_values('trade_date')
    dates=nav.index.tolist(); prev=np.r_[1.0,nav.values[:-1]]
    lr=np.log(nav.values/prev)
    rb={d:(pk,eb,ea) for d,pk,eb,ea in zip(r.trade_date,r.picks,r.equity_before,r.equity_after)}
    held=None; rec=[]  # (date, ticker, logret, kind)
    for i,d in enumerate(dates):
        if d in rb:
            pk,eb,ea=rb[d]
            assert abs(ea/nav.values[i]-1)<1e-9, (pid,d)
            if held is None:  # entry day: whole lr is buy fee of new pick
                rec.append((d,pk,lr[i],'fee',prev[i])); held=pk; continue
            mkt=np.log(eb/prev[i]); fl=lr[i]-mkt
            f=FEE.loc[(pid,d)] if (pid,d) in FEE.index else pd.Series({'buy':0,'sell':0})
            tot=f.get('buy',0)+f.get('sell',0)
            fb=fl*f.get('buy',0)/tot if tot>0 else 0.0
            rec.append((d,held,mkt,'mkt',prev[i])); rec.append((d,held,fl-fb,'fee',prev[i])); rec.append((d,pk,fb,'fee',prev[i]))
            held=pk
        else:
            rec.append((d,held,lr[i],'mkt',prev[i]))
    df=pd.DataFrame(rec,columns=['date','ticker','lr','kind','nav_prev']); df['path_id']=pid
    assert abs(df.lr.sum()-np.log(nav.iloc[-1]))<1e-9
    return df
A=pd.concat([daily_attr(p) for p in need],ignore_index=True)
A['year']=A.date.str[:4].astype(int)

# ---------- path-level stats ----------
def year_rets(pid,entry,end,final):
    nav=NAV[pid]; out={}
    for Y in range(2015,2026):
        s=f'{Y-1}-12-31'; e=f'{Y}-12-31'
        if end<=s: continue
        if entry>s: b=1.0; bd=entry
        else: bd=nav.loc[:s].index[-1]; b=nav.loc[bd]
        ed=nav.loc[:e].index[-1]; v=nav.loc[ed]
        x=nav.loc[bd:ed] if entry<=s else pd.concat([pd.Series([1.0]),nav.loc[:ed]])
        mdd=(x/x.cummax()-1).min()
        out[Y]=dict(ret=v/b-1,spy=spy_on(ed)/spy_on(bd)-1,mdd_in_year=mdd,start=bd,stop=ed,full=(entry<=s and end>e))
    return out
PL=[];YR=[]
for _,p in pd.concat([B1,W9[1]]).iterrows():
    pid=p.path_id; nav=NAV[pid]; a=A[A.path_id==pid]
    nv=a[a.ticker=='NVDA']; nvm=nv[nv.kind=='mkt']
    ndays=(a.kind=='mkt').sum()
    split=nav.loc[:'2019-12-31'].index[-1]
    d1=(pd.Timestamp(split)-pd.Timestamp(p.entry)).days; d2=(pd.Timestamp(p.end)-pd.Timestamp(split)).days
    # alt halves: full calendar years 2016-2019 and 2020-2024 (common to every path)
    b15=nav.loc[:'2015-12-31'].iloc[-1]; e19=nav.loc[:'2019-12-31'].iloc[-1]; e24=nav.loc[:'2024-12-31'].iloc[-1]
    dd15=nav.loc[:'2015-12-31'].index[-1]; dd24=nav.loc[:'2024-12-31'].index[-1]
    tot=np.log(p.final_multiple)
    cf1=np.exp(tot-nvm.lr.sum()); cf2=np.exp(tot-nv.lr.sum())
    spyc=cagr(spy_on(p.end)/spy_on(p.entry),p.days)
    PL.append(dict(path_id=pid,W=p.W,H=p.H,K=p.K,phase=p.phase,entry=p.entry,end=p.end,status=p.status,final_multiple=p.final_multiple,
        CAGR=p.CAGR,daily_MDD=p.daily_MDD,CAGR_check=cagr(p.final_multiple,p.days),SPY_CAGR_same_window=spyc,beat_SPY=p.CAGR>spyc,
        nvda_time_share_days=(nv.kind=='mkt').sum()/ndays,
        nvda_time_share_equity=nvm.nav_prev.sum()/a[a.kind=='mkt'].nav_prev.sum(),
        total_log=tot,nvda_log_mkt=nvm.lr.sum(),nvda_log_incl_fees=nv.lr.sum(),
        nvda_log_share=nv.lr.sum()/tot,nvda_log_share_mkt=nvm.lr.sum()/tot,
        cashflat_CAGR_keepfees=cagr(cf1,p.days),cashflat_CAGR_nvdaflat=cagr(cf2,p.days),
        split_date=split,half1_CAGR=cagr(nav.loc[split],d1),half2_CAGR=cagr(p.final_multiple/nav.loc[split],d2),half1_days=d1,half2_days=d2,
        alt_2016_2019_CAGR=cagr(e19/b15,(pd.Timestamp(split)-pd.Timestamp(dd15)).days),
        alt_2020_2024_CAGR=cagr(e24/e19,(pd.Timestamp(dd24)-pd.Timestamp(split)).days)))
    for Y,v in year_rets(pid,p.entry,p.end,p.final_multiple).items():
        YR.append(dict(path_id=pid,W=p.W,H=p.H,phase=p.phase,year=Y,**v,beat=v['ret']>v['spy']))
PL=pd.DataFrame(PL); YR=pd.DataFrame(YR)
PL[PL.W>=10].to_csv(OUT+'top1_108_path_stats.csv',index=False)
PL[PL.W==9].to_csv(OUT+'w9_k1_36_path_stats.csv',index=False)
YR.to_csv(OUT+'year_by_year_path_returns.csv',index=False)
P1=PL[PL.W>=10]; Y1=YR[YR.W>=10]; Y9=YR[YR.W==9]

# ---------- claim 1 ----------
add(1,'Top1 band paths computable','108/108',f'{(B1.status!="UNKNOWN").sum()}/108','MATCH')
for nm,v,t in [('Top1 min CAGR',B1.CAGR.min(),20.5),('Top1 median CAGR',B1.CAGR.median(),40.1),('Top1 max CAGR',B1.CAGR.max(),65.0)]:
    add(1,nm,f'{t}%',pc(v),m1(v,t),'paths_summary CAGR, 108 K1 paths')
# ---------- claim 2 ----------
G=P[P.W.isin([10,11,12])&(P.K==1)&(P.status!='UNKNOWN')].groupby(['W','H']).CAGR.median().unstack()
G.to_csv(OUT+'w10_12_h1_12_k1_median_cagr.csv')
C=pd.read_csv(AK+'cells_12phase_medians.csv')
cc=C[C.W.isin([10,11,12])&(C.K==1)].set_index(['W','H']).median_CAGR.unstack()
assert np.allclose(cc.values,G.values)
nk=P[P.W.isin([10,11,12])&(P.K==1)].groupby(['W','H']).status.apply(lambda s:(s=='UNKNOWN').sum()).sum()
inr=((G>=0.30)&(G<=0.45)).values.sum()
add(2,'W10-12 x H1-12 K1 cell medians all in 30%-45%','36/36 in band',f'{inr}/36 in band; min {pc(G.values.min())} (W{G.stack().idxmin()[0]}H{G.stack().idxmin()[1]}), max {pc(G.values.max())} (W{G.stack().idxmax()[0]}H{G.stack().idxmax()[1]}); UNKNOWN paths in these 432: {nk}',
    'MATCH' if inr==36 else 'MISMATCH','median over known phases; equals cells_12phase_medians.csv')
h24=G[[2,3,4]].values.mean(); oth=G[[1]+list(range(5,13))].values.mean()
add(2,'H2-4 not special',"H2-4 not special",f'mean of H2-4 cell medians {pc(h24)} vs other H {pc(oth)}; rank of H2-4 cells among 36: '+
    ','.join(str(int(x)) for x in G.stack().rank(ascending=False).loc[[(w,h) for w in (10,11,12) for h in (2,3,4)]]),'see table','qualitative')
# ---------- claim 3 ----------
spy_y={}
yt=[]
for Y in range(2015,2026):
    y=Y1[Y1.year==Y]; y9=Y9[Y9.year==Y]
    full=y[y.full]
    yt.append(dict(year=Y,n_paths=len(y),n_full_year=len(full),median_ret_W10_12=y.ret.median(),n_beat_SPY=int(y.beat.sum()),share_beat=y.beat.mean(),
        SPY_full_calendar=(spy_on(f'{Y}-12-31')/spy_on(f'{Y-1}-12-31')-1),median_ret_W9=y9.ret.median(),W9_share_beat=y9.beat.mean(),
        median_in_year_MDD_W10_12=y.mdd_in_year.median(),worst_in_year_MDD_W10_12=y.mdd_in_year.min()))
YT=pd.DataFrame(yt); YT.to_csv(OUT+'year_by_year_vs_spy.csv',index=False)
f=YT[(YT.year>=2016)&(YT.year<=2024)]
maj=f[f.share_beat>0.5].year.tolist(); lag=f[f.share_beat<=0.5].year.tolist()
add(3,'Years 2016-2024 where majority of 108 beat SPY','7 of 9; lag 2018, 2023',f'{len(maj)} of 9; lag {lag}',
    'MATCH' if lag==[2018,2023] else 'MISMATCH','calendar year last-close to last-close; SPY FMP dividend-adjusted, same dates')
# ---------- claim 4 ----------
common=cagr(spy_on('2025-01-02')/spy_on('2015-01-02'),(pd.Timestamp('2025-01-02')-pd.Timestamp('2015-01-02')).days)
add(4,'Top2 median CAGR','29.5%',pc(B2.CAGR.median()),m1(B2.CAGR.median(),29.5),f'108 K2 paths, all known={ (B2.status!="UNKNOWN").all()}')
add(4,'Top3 median CAGR','21.4%',pc(B3.CAGR.median()),m1(B3.CAGR.median(),21.4),f'108 K3 paths, all known={ (B3.status!="UNKNOWN").all()}')
add(4,'Contemporaneous SPY CAGR','~13% (from memory)',f'{pc(common)} (2015-01-02->2025-01-02 close-to-close); per-path same-window median {pc(P1.SPY_CAGR_same_window.median())}, range {pc(P1.SPY_CAGR_same_window.min())}-{pc(P1.SPY_CAGR_same_window.max())}',
    'MATCH' if abs(common-0.13)<0.006 else 'MISMATCH','FMP SPY dividend-adjusted (total-return proxy), no fees')
# ---------- claim 5 ----------
pool_time=(A[(A.path_id.isin(P1.path_id))&(A.kind=='mkt')].ticker=='NVDA').mean()
pool_share=P1.nvda_log_incl_fees.sum()/P1.total_log.sum(); pool_share_m=P1.nvda_log_mkt.sum()/P1.total_log.sum()
add(5,'NVDA share of Top1 total log return','77%',f'pooled {pc(pool_share)} (incl. own fees) / {pc(pool_share_m)} (market only); median per-path {pc(P1.nvda_log_share.median())}',
    m1(pool_share,77) if abs(pool_share*100-77)<0.5 else ('NEAR' if abs(pool_share*100-77)<3 else 'MISMATCH'),'daily attribution, holding over (t-1,t]; fees split by leg')
add(5,'NVDA holding time share','36%',f'pooled day-weighted {pc(pool_time)}; median per-path {pc(P1.nvda_time_share_days.median())}; equity-weighted median {pc(P1.nvda_time_share_equity.median())}',
    'MATCH' if abs(pool_time*100-36)<0.5 else ('NEAR' if abs(pool_time*100-36)<3 else 'MISMATCH'),'trading sessions held / sessions in path')
add(5,'Median CAGR with NVDA periods as cash','~8%',f'{pc(P1.cashflat_CAGR_keepfees.median())} (keep all fees) / {pc(P1.cashflat_CAGR_nvdaflat.median())} (NVDA-leg fees also removed)',
    'MATCH' if abs(P1.cashflat_CAGR_keepfees.median()-0.08)<0.01 else ('NEAR' if abs(P1.cashflat_CAGR_keepfees.median()-0.08)<0.02 else 'MISMATCH'),'NAV flat on NVDA-held sessions; other returns unchanged (no reinvestment into another pick)')
NV=P1[['path_id','W','H','phase','total_log','nvda_log_mkt','nvda_log_incl_fees','nvda_log_share','nvda_time_share_days','nvda_time_share_equity','CAGR','cashflat_CAGR_keepfees','cashflat_CAGR_nvdaflat']]
NV.to_csv(OUT+'nvda_attribution_per_path.csv',index=False)
# ---------- claim 6 ----------
w9m=P[(P.W==9)&(P.K==1)&P.H.isin([2,3,4])].groupby('H').CAGR.median()
add(6,'W9 H2-4 K1 cell medians','17%-28%',', '.join(f'H{h} {pc(v)}' for h,v in w9m.items()),
    'MATCH' if abs(round(w9m.min()*100)-17)<1 and abs(round(w9m.max()*100)-28)<1 else 'MISMATCH','12 phases each, all known')
for Y,t9,t10 in [(2016,11,86),(2022,-21,34)]:
    a=YT.set_index('year').loc[Y]
    add(6,f'{Y} median calendar return W9 vs W10-12',f'W9 ~{t9:+d}% vs W10-12 ~{t10:+d}%',f'W9 {pc(a.median_ret_W9)} vs W10-12 {pc(a.median_ret_W10_12)}',
        'MATCH' if abs(a.median_ret_W9*100-t9)<1.5 and abs(a.median_ret_W10_12*100-t10)<1.5 else 'MISMATCH','calendar year, median over 36 / 108 paths')
# per-ticker log contribution in 2016 and 2022 (mean per path), W9 vs W10-12
tk=[]
for Y in (2016,2022):
    for g,ids in [('W9',W9[1].path_id),('W10-12',P1.path_id)]:
        a=A[(A.year==Y)&A.path_id.isin(ids)]
        s=a.groupby('ticker').lr.sum()/len(ids); d=a[a.kind=='mkt'].groupby('ticker').size()/len(ids)
        for t in s.index: tk.append(dict(year=Y,group=g,ticker=t,mean_log_contrib_per_path=s[t],mean_sessions_held_per_path=d.get(t,0)))
TK=pd.DataFrame(tk); TK.to_csv(OUT+'w9_vs_w10_12_ticker_contrib_2016_2022.csv',index=False)
W9T=pd.DataFrame([dict(K=k,H=h,median_CAGR=g.CAGR.median(),min_CAGR=g.CAGR.min(),max_CAGR=g.CAGR.max(),median_MDD=g.daily_MDD.median()) for k in (1,2,3) for h,g in W9[k].groupby('H')]+
    [dict(K=k,H='all',median_CAGR=W9[k].CAGR.median(),min_CAGR=W9[k].CAGR.min(),max_CAGR=W9[k].CAGR.max(),median_MDD=W9[k].daily_MDD.median()) for k in (1,2,3)])
W9T.to_csv(OUT+'w9_contrast.csv',index=False)
# ---------- claim 7 ----------
add(7,'Top1 -> Top3 median','~40% -> ~21%',f'{pc(B1.CAGR.median())} -> {pc(B2.CAGR.median())} -> {pc(B3.CAGR.median())}','MATCH' if m1(B3.CAGR.median(),21.4)=='MATCH' and m1(B1.CAGR.median(),40.1)=='MATCH' else 'see claims 1/4','')
# ---------- claim 8 ----------
add(8,'Top1 median CAGR entry->2019-12-31','30.9%',f'{pc(P1.half1_CAGR.median())} (alt full-yr 2016-2019: {pc(P1.alt_2016_2019_CAGR.median())})',m1(P1.half1_CAGR.median(),30.9),'split at last session <=2019-12-31; base 1.0 at entry')
add(8,'Top1 worst path entry->2019-12-31','1.0%',f'{pc(P1.half1_CAGR.min())} (alt 2016-2019: {pc(P1.alt_2016_2019_CAGR.min())})',m1(P1.half1_CAGR.min(),1.0),'')
add(8,'Top1 median CAGR 2020->end','50.6%',f'{pc(P1.half2_CAGR.median())} (alt 2020-2024: {pc(P1.alt_2020_2024_CAGR.median())})',m1(P1.half2_CAGR.median(),50.6),'split->path end')
P1[['path_id','W','H','phase','entry','end','split_date','half1_days','half2_days','half1_CAGR','half2_CAGR','alt_2016_2019_CAGR','alt_2020_2024_CAGR']].to_csv(OUT+'half_period_cagr.csv',index=False)
# ---------- claim 9 ----------
add(9,'Top1 median max DD','-54%',pc(B1.daily_MDD.median()),m1(B1.daily_MDD.median(),-54.0) if True else '','paths_summary daily_MDD; their figure 0-decimal')
add(9,'Top1 worst max DD','-69%',pc(B1.daily_MDD.min()),'','')
y18=YT.set_index('year').loc[2018]
add(9,'2018 median calendar return (Top1)','-30%',f'{pc(y18.median_ret_W10_12)}; median in-2018 MDD {pc(y18.median_in_year_MDD_W10_12)}','',' calendar 2018')
add(9,'SPY 2018','~-4%',pc(y18.SPY_full_calendar),'','FMP dividend-adjusted, 2017-12-29 -> 2018-12-31')
add(10,'Post-hoc selection from 432 cells','qualitative','not recomputable','N/A','band chosen after seeing results')
CMP=pd.DataFrame(rows)
# integer-precision matches for 0-decimal claims
def m0(v,t): return 'MATCH' if round(v*100)==t else ('NEAR' if abs(v*100-t)<1.5 else 'MISMATCH')
CMP.loc[CMP.item=='Top1 median max DD','match']=m0(B1.daily_MDD.median(),-54)
CMP.loc[CMP.item=='Top1 worst max DD','match']=m0(B1.daily_MDD.min(),-69)
CMP.loc[CMP.item=='2018 median calendar return (Top1)','match']=m0(y18.median_ret_W10_12,-30)
CMP.loc[CMP.item=='SPY 2018','match']=m0(y18.SPY_full_calendar,-4)
# ---------- convention checks for near-misses ----------
def mid(x): v=np.sort(np.asarray(x)); return v[len(v)//2-1],v[len(v)//2]
lo,hi=mid(B1.CAGR)
CMP.loc[CMP.item=='Top1 median CAGR',['match','convention_note']]=['NEAR (median convention)',f'standard median of 108 = {pc(np.median(B1.CAGR))}; 54th/55th sorted values = {pc(lo)}/{pc(hi)} -> 40.1% equals the lower-middle value']
lo,hi=mid(P1.half2_CAGR)
CMP.loc[CMP.item=='Top1 median CAGR 2020->end',['match','convention_note']]=['NEAR (median convention)',f'standard median {pc(P1.half2_CAGR.median())}; 54th/55th sorted = {pc(lo)}/{pc(hi)} -> 50.6% equals the upper-middle value']
CMP.loc[CMP.claim==2,'match']=CMP.loc[CMP.claim==2,'match'].replace('MISMATCH','NEAR (W10H2 45.24% just above 45%)')
CMP.loc[CMP.claim==7,'match']='MATCH (direction); levels per claims 1/4'
# off-by-one alternative NVDA attribution: rebalance-day move credited to the new pick
alt=[]
for pid in P1.path_id:
    nav=NAV[pid]; r=R[R.path_id==pid].set_index('trade_date').picks
    hold=r.reindex(nav.index).ffill(); lr=np.log(nav/nav.shift(1).fillna(1.0))
    alt.append((lr[hold=='NVDA'].sum(),lr.sum()))
alt=np.array(alt); d=P1.set_index('path_id').loc[:, 'final_multiple']
cfa=[np.exp(t-n)**(365.25/dd)-1 for (n,t),dd in zip(alt,P1.merge(P[['path_id','days']]).days)]
CMP.loc[CMP.item=='NVDA share of Top1 total log return','convention_note']+=f'; alt (rebalance-day move to new pick): pooled {pc(alt[:,0].sum()/alt[:,1].sum())}, median {pc(np.median(alt[:,0]/alt[:,1]))}; mean per-path {pc(P1.nvda_log_share.mean())}'
CMP.loc[CMP.item=='Median CAGR with NVDA periods as cash','convention_note']+=f'; alt off-by-one attribution: {pc(np.median(cfa))}'
CMP.to_csv(OUT+'comparison.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',120)
print(CMP.to_string()); print(G.round(4)); print(YT.round(4).to_string()); print(W9T.round(4).to_string())
for Y in (2016,2022):
    for g in ('W9','W10-12'):
        print(Y,g,TK[(TK.year==Y)&(TK.group==g)].sort_values('mean_log_contrib_per_path',key=abs,ascending=False).head(6).round(4).to_string(index=False))
print('CAGR_check maxdiff',(P1.CAGR-P1.CAGR_check).abs().max(),'CA events in K1 band rows',len(ca), ca.execution_key.unique())
