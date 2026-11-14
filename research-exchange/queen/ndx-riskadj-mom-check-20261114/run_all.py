from riskadj import *
P,TR=E.qqq()
WIN={'A_their_2008-01_to_2026-08':('2008-01-31','2026-08-31'),'B_2015-01_to_2026-11-13':('2014-12-31','2026-11-13'),'C_2008-01_to_2026-11-13':('2008-01-31','2026-11-13')}
VAR=[('N0_12-1',('mom',12,1,0),()),('N0.5_12-1',('mom',12,1,0.5),()),('N1_12-1',('mom',12,1,1),()),('N1.5_12-1',('mom',12,1,1.5),()),('N2_12-1',('mom',12,1,2),()),
     ('N0_6-1',('mom',6,1,0),()),('N1_6-1',('mom',6,1,1),()),('N0_9-1',('mom',9,1,0),()),('N1_9-1',('mom',9,1,1),()),
     ('mean6_postrule',('mean6',),()),('N0_12-1_noNVDA',('mom',12,1,0),('NVDA',)),('N1_12-1_noNVDA',('mom',12,1,1),('NVDA',)),('mean6_postrule_noNVDA',('mean6',),('NVDA',))]
EXECS=[('next_open',1,0.0015),('month_end_close',0,0.0015),('month_end_close',0,0.0025)]
rows=[]; RES={}
cols=['window','execution','cost_side','variant','start','end','months','CAGR','CAGR_months12','vol_daily_ann','vol_monthly_ann','MDD_daily','MDD_monthly','sharpe_monthly','turnover_oneway_yr','total']
def add(w,ex,c,v,res):
    s=stats(res); s.update(window=w,execution=ex,cost_side=c,variant=v); rows.append(s); RES[(w,ex,c,v)]=res
    print('%-26s %-15s %.2f%% %-22s CAGR %6.2f%% vol %5.1f%% MDDd %6.1f%% MDDm %6.1f%% Sh %.3f x%.1f'%(w,ex,c*100,v,s['CAGR']*100,s['vol_daily_ann']*100,s['MDD_daily']*100,s['MDD_monthly']*100,s['sharpe_monthly'],s['total']))
for w,(a,b) in WIN.items():
    i0,i1=IDX[a],IDX[b]
    for ex,lag,c in EXECS:
        for v,spec,exc in VAR: add(w,ex,c,v,run(spec,3,i0,i1,c,lag,exc))
    for v,ser in (('QQQ_price',P),('QQQ_total_return',TR)): add(w,'buy_hold_close',0,v,bench(ser,i0,i1))
with open(OUT+'/results_main.csv','w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore'); wr.writeheader(); wr.writerows(rows)
# cost sweep and execution-lag robustness (window A and B)
rows2=[]
for w in ('A_their_2008-01_to_2026-08','B_2015-01_to_2026-11-13'):
    i0,i1=IDX[WIN[w][0]],IDX[WIN[w][1]]
    for v,spec in (('N1_12-1',('mom',12,1,1)),('N0_12-1',('mom',12,1,0)),('mean6_postrule',('mean6',))):
        for c in (0,0.0015,0.0025,0.005):
            s=stats(run(spec,3,i0,i1,c,1)); s.update(window=w,execution='next_open',cost_side=c,variant=v,test='cost_sweep'); rows2.append(s)
        for lag in (0,1,2,5,10):
            s=stats(run(spec,3,i0,i1,0.0015,lag)); s.update(window=w,execution='lag%d'%lag+('_close' if lag==0 else '_open'),cost_side=0.0015,variant=v,test='exec_lag'); rows2.append(s)
with open(OUT+'/results_cost_and_lag.csv','w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=['test']+cols,extrasaction='ignore'); wr.writeheader(); wr.writerows(rows2)
print('\n== cost sweep / exec lag ==')
for s in rows2: print('%-9s %-26s %-11s %.2f%% %-15s CAGR %6.2f%% MDDd %6.1f%% Sh %.3f'%(s['test'],s['window'],s['execution'],s['cost_side']*100,s['variant'],s['CAGR']*100,s['MDD_daily']*100,s['sharpe_monthly']))
# pairwise: daily corr, overlap, bootstrap Sharpe diff
def sharpe(x): return x.mean()/x.std(ddof=1)*math.sqrt(12)
def cbb(a,b,block,reps=10000,seed=20261114):
    rng=np.random.default_rng(seed); n=len(a); nb=-(-n//block); out=np.empty(reps)
    for k in range(reps):
        st=rng.integers(0,n,nb); idx=(st[:,None]+np.arange(block)[None,:]).ravel()[:n]%n
        out[k]=sharpe(a[idx])-sharpe(b[idx])
    return np.percentile(out,[2.5,97.5]),float((out<=0).mean())
rows3=[]
print('\n== pairwise ==')
for w in WIN:
  for ex,lag,c in EXECS[:2]:
    for x,y in (('N1_12-1','N0_12-1'),('N1_12-1','mean6_postrule'),('mean6_postrule','N0_12-1')):
        A=RES[(w,ex,c,x)]; B=RES[(w,ex,c,y)]
        ma=np.array([r['net'] for r in A['rec']]); mb=np.array([r['net'] for r in B['rec']])
        da=np.array(A['dnav']); db=np.array(B['dnav']); assert A['ddates']==B['ddates']
        corr=float(np.corrcoef(da[1:]/da[:-1]-1,db[1:]/db[:-1]-1)[0,1])
        ov=np.mean([len(set(p['picks'].split())&set(q['picks'].split()))/3 for p,q in zip(A['rec'],B['rec'])])
        r=dict(window=w,execution=ex,cost_side=c,a=x,b=y,months=len(ma),sharpe_a=sharpe(ma),sharpe_b=sharpe(mb),sharpe_diff=sharpe(ma)-sharpe(mb),daily_corr=corr,avg_top3_overlap=ov)
        for bl in (6,12):
            ci,p=cbb(ma,mb,bl); r['ci95_lo_block%d'%bl]=ci[0]; r['ci95_hi_block%d'%bl]=ci[1]; r['share_le0_block%d'%bl]=p
        rows3.append(r); print({k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items()})
with open(OUT+'/results_pairwise_bootstrap.csv','w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=list(rows3[0])); wr.writeheader(); wr.writerows(rows3)
# drawdown episodes and 2025Q1 (window A, next open)
print('\n== drawdown episodes / 2025Q1 (next_open 0.15%) ==')
rows4=[]
def win_stats(dn,dd,a,b):
    xs=[(d[:10],v) for d,v in zip(dd,dn)]
    base=[v for d,v in xs if d<=a][-1]; seg=[base]+[v for d,v in xs if a<d<=b]
    return seg[-1]/seg[0]-1, mdd(seg)
for w in ('A_their_2008-01_to_2026-08','C_2008-01_to_2026-11-13'):
    for v in ('N1_12-1','N0_12-1','mean6_postrule'):
        R_=RES[(w,'next_open',0.0015,v)]; (L_,p0,p1),(m,pk,tr,_)=ddur(R_['dnav'],R_['ddates'])
        q,qm=win_stats(R_['dnav'],R_['ddates'],'2024-12-31','2025-03-31')
        r=dict(window=w,variant=v,maxdd=m,maxdd_peak=pk,maxdd_trough=tr,longest_recovery_days=L_,longest_peak=p0,longest_recovered=p1,q1_2025_ret=q,q1_2025_mdd=qm)
        # recovery date of max-DD episode
        dn=R_['dnav']; dd=R_['ddates']; j=dd.index(pk); pkv=dn[j]; rec=[d for d,x in zip(dd[j:],dn[j:]) if x>=pkv and d!=pk]
        r['maxdd_recovered']=rec[0] if rec else 'not_recovered'; rows4.append(r); print(r)
    for v,ser in (('QQQ_price',P),('QQQ_total_return',TR)):
        R_=RES[(w,'buy_hold_close',0,v)]; q,qm=win_stats(R_['dnav'],R_['ddates'],'2024-12-31','2025-03-31')
        r=dict(window=w,variant=v,q1_2025_ret=q,q1_2025_mdd=qm,maxdd=mdd(R_['dnav'])); rows4.append(r); print(r)
with open(OUT+'/results_drawdown_episodes.csv','w',newline='') as f:
    ks=[]; [ks.append(k) for r in rows4 for k in r if k not in ks]
    wr=csv.DictWriter(f,fieldnames=ks); wr.writeheader(); wr.writerows(rows4)
# picks
with open(OUT+'/picks_next_open_015.csv','w',newline='') as f:
    wr=None
    for v,_,_ in VAR:
        for r in RES[('C_2008-01_to_2026-11-13','next_open',0.0015,v)]['rec']:
            if wr is None: wr=csv.DictWriter(f,fieldnames=['variant']+list(r)); wr.writeheader()
            wr.writerow(dict(variant=v,**r))
# 2026-09-30 example
print('\n== 2026-09-30 signal example ==')
i=IDX['2026-09-30']; r0=ranking(i,('mom',12,1,0)); r1=ranking(i,('mom',12,1,1))
pos0={t:k+1 for k,(s,t,*_ ) in enumerate(r0)}; rows5=[]
for k,(s,t,key,ret,vol) in enumerate(r1):
    rows5.append(dict(adj_rank=k+1,raw_rank=pos0.get(t),ticker=t,ret_12_1=ret,vol_ann=vol,score_N1=s))
    if k<15 or t in ('FTNT','INTC'): print(rows5[-1])
with open(OUT+'/example_2026-09-30_ranking.csv','w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=list(rows5[0])); wr.writeheader(); wr.writerows(rows5)
print('N=1 top3 2026-09-30:',[x[1] for x in r1[:3]],' N=0 top3:',[x[1] for x in r0[:3]],' n_ranked',len(r1))
print('window 12-1 = ME',ME[i-12],'->',ME[i-1])
print('FALLBACK counts (cumulative over all runs):',FALLBACK)
