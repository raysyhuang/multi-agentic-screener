"""Blended W10/W11/W12 sleeves (each its own engine path with K=N, own fees; 1/3 capital each). SECONDARY study.
drift: blend NAV = mean of the three sleeve NAVs (no cross-sleeve rebalancing). reset: at each common rebalance date the three sleeves
are reset to 1/3 each, charging 25 bp on sum|E_i - E/3| (conservative transfer cost). Any sleeve UNKNOWN -> blend UNKNOWN for that phase."""
import pickle,glob,numpy as np,pandas as pd,datetime as dt
B='/workspace/momentum-followup-20261114/';c=0.0025
spy=pd.read_csv('/workspace/momentum-etf-check-20261113/raw/SPY_px_daily.csv').date.tolist()
def mdd(v): v=np.asarray(v);return float((v/np.maximum.accumulate(v)-1).min())
def sess(a,b,cal): return [d for d in cal if a<=d<=b]
def blend(navs,days,rebs):
    drift=np.mean(navs,0)
    ix=[days.index(d) for d in rebs if d in days]+[len(days)-1];ix=sorted(set(ix))
    E=1.0;out=np.empty(len(days));out[:ix[0]+1]=np.mean([n[:ix[0]+1] for n in navs],0)   # first segment from inception
    E=out[ix[0]]
    for a,b in zip(ix,ix[1:]):
        g=np.array([n[a:b+1]/n[a] for n in navs]);seg=E*g.mean(0);out[a:b+1]=seg
        Ei=E/3*g[:,-1];Et=Ei.sum();fee=c*np.abs(Ei-Et/3).sum() if b!=len(days)-1 else 0.0
        E=Et-fee;out[b]=E
    return drift,out
rows=[]
def run(period,mode,P,NAV,RB,cal,y0):
    for H in sorted({p['H'] for p in P}):
        for K in sorted({p['K'] for p in P}):
            for ph in range(1,13):
                ps=[p for p in P if p['H']==H and p['K']==K and p['phase']==ph]
                if len(ps)!=3: continue
                st=[p['status'] for p in ps];unk=[p for p in ps if p['status']=='UNKNOWN']
                r=dict(period=period,mode=mode,H=H,N=K,phase=ph,sleeves_known=3-len(unk),fail=';'.join(f"W{p['W']}:{p.get('fail_reason')}:{p.get('fail_ticker')}" for p in unk))
                if not unk:
                    navs=[np.asarray(NAV[(p['W'],H,K,ph)]) for p in ps];days=sess(ps[0]['entry'],ps[0]['end'],cal)
                    assert all(len(n)==len(days) for n in navs),(len(days),[len(n) for n in navs])
                    rebs=sorted({x for x in RB.get((ps[0]['W'],H,K,ph),[])})
                    d,rs=blend(navs,days,rebs);yrs=(dt.date.fromisoformat(days[-1])-dt.date.fromisoformat(days[0])).days/365.25
                    r.update(drift_CAGR=d[-1]**(1/yrs)-1,drift_MDD=mdd(d),reset_CAGR=rs[-1]**(1/yrs)-1,reset_MDD=mdd(rs),
                             postcheck_all_picks_hit=';'.join(str(p.get('postcheck_all_picks_hit')) for p in ps if p.get('postcheck_all_picks_hit')))
                rows.append(r)
# 2015-2025
for mode in ('base','nonvda'):
    P=[];NAV={};RB={}
    for f in glob.glob(B+f'scratch/blend1525_{mode}_W*_H*.pkl'):
        d=pickle.load(open(f,'rb'))
        for p in d['P']:
            P.append(p)
            if p['path_id'] in d['NAV']: NAV[(p['W'],p['H'],p['K'],p['phase'])]=d['NAV'][p['path_id']]
        for x in d['RB']:
            import re;m=re.match(r'W(\d+)H(\d+)K(\d+)P(\d+)',x['path_id']);RB.setdefault(tuple(map(int,m.groups())),[]).append(x['trade_date'])
    run('2015-2025',mode,P,NAV,RB,spy,2015)
# 2005-2015 (pack engine grid run; K1/K3 from the grid, K5/10/20 extra jobs)
E=pickle.load(open(B+'scratch/nav0515_blend.pkl','rb'));G=pd.read_csv(B+'results/r0515/paths_0515_all.csv')
P=[];NAV={};RB={}
for _,g in G[G.W.isin([10,11,12])&G.H.isin([1,3,6,12])&G.K.isin([1,3,5,10,20])].iterrows():
    k=(g.W,g.H,g.K,g.phase);P.append(dict(W=g.W,H=g.H,K=g.K,phase=g.phase,entry=g.entry,end=g.end,status=g.status,fail_reason=g.fail_reason,fail_ticker=g.fail_ticker))
    if k in E and E[k]['nav']: NAV[k]=E[k]['nav'];RB[k]=[d for _,d,_ in E[k]['rebalances']]
cal=sorted(set(d for v in E.values() if v['dates'] for d in v['dates']))
run('2005-2015','base',P,NAV,RB,cal,2005)
D=pd.DataFrame(rows);D.to_csv(B+'results/blend_paths.csv',index=False)
def agg(x):
    k=x[x.sleeves_known==3]
    return pd.Series(dict(phases_known=len(k),drift_median=k.drift_CAGR.median(),drift_min=k.drift_CAGR.min(),drift_max=k.drift_CAGR.max(),drift_medMDD=k.drift_MDD.median(),drift_worstMDD=k.drift_MDD.min(),
                          reset_median=k.reset_CAGR.median(),reset_worstMDD=k.reset_MDD.min()))
S=D.groupby(['period','mode','H','N']).apply(agg,include_groups=False).reset_index();S.to_csv(B+'results/blend_cells.csv',index=False)
pd.set_option('display.width',220);print(S.round(4).to_string())
