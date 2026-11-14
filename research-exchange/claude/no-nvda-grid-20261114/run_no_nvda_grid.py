#!/usr/bin/env python3
"""Claude: the full 5,184-path grid (W1-12 x H1-12 x K1-3 x 12 start months) with NVDA removed from the candidate list.
Exact engine (Claude's own executor, ../input-audit-20261112/stoploss_claude.py with no stop) on the frozen 2015-2025 pack.
When NVDA would have been picked, the next-ranked name moves up. A replacement the frozen execution data has no
record for makes the path UNKNOWN (the pack only holds prices for names that were held in the original grid).
Usage: python3 run_no_nvda_grid.py  -> paths_no_nvda_5184.csv, cells_no_nvda_432.csv, stdout."""
import os,sys,csv,subprocess,importlib.util,statistics as st,collections
HERE=os.path.dirname(os.path.abspath(__file__)); A=os.path.join(HERE,'..','input-audit-20261112'); R=os.path.join(HERE,'..','..')
PACK=os.path.join(R,'shared','inputs-pack-full-20261109'); pk=os.path.join(HERE,'rank_no_nvda.pkl')
env=dict(os.environ,EXCLUDE_TICKERS='NVDA',RANK_OUT=pk); subprocess.check_call([sys.executable,os.path.join(A,'independent_ranker_claude.py'),PACK],env=env,stdout=subprocess.DEVNULL)
os.environ['RANK_PKL']=pk; sys.argv=['x',PACK]
spec=importlib.util.spec_from_file_location('eng',os.path.join(A,'stoploss_claude.py')); eng=importlib.util.module_from_spec(spec); spec.loader.exec_module(eng)
base={r['path_id']:r for r in csv.DictReader(open(os.path.join(R,'queen','answer-key-full-20261109','paths_summary.csv')))}
rows=[]; c=collections.Counter()
for W in range(1,13):
  for H in range(1,13):
    for K in (1,2,3):
      for ph in range(1,13):
        pid=f'W{W}H{H}K{K}P{ph}'; b=base[pid]
        try:
            f,cg,md,_,fb=eng.run(W,H,K,ph,None); stt,why='KNOWN',''
        except eng.Fail as e: f=cg=md='';stt,why='UNKNOWN',str(e).split(' ')[0]
        c[(K,stt)]+=1
        rows.append(dict(path_id=pid,W=W,H=H,K=K,start_month=ph,status=stt,fail_reason=why,final_multiple=f,CAGR=cg,daily_MDD=md,with_nvda_status=b['status'],with_nvda_CAGR=b['CAGR'],with_nvda_MDD=b['daily_MDD']))
w=csv.DictWriter(open(os.path.join(HERE,'paths_no_nvda_5184.csv'),'w',newline=''),fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('paths by K and status:',dict(sorted(c.items())))
print('fail reasons:',dict(collections.Counter(r['fail_reason'] for r in rows if r['status']=='UNKNOWN')))
cells=[]
for W in range(1,13):
  for H in range(1,13):
    for K in (1,2,3):
        z=[r for r in rows if (r['W'],r['H'],r['K'])==(W,H,K)]; k=[r for r in z if r['status']=='KNOWN']
        wb=[float(r['with_nvda_CAGR']) for r in z if r['with_nvda_status']!='UNKNOWN']
        if not k: cells.append(dict(W=W,H=H,K=K,known=0,median_CAGR='',min_CAGR='',max_CAGR='',median_MDD='',worst_MDD='',with_nvda_known=len(wb),with_nvda_median_CAGR=st.median(wb) if wb else '')); continue
        cg=sorted(r['CAGR'] for r in k); md=[r['daily_MDD'] for r in k]
        cells.append(dict(W=W,H=H,K=K,known=len(k),median_CAGR=st.median(cg),min_CAGR=cg[0],max_CAGR=cg[-1],median_MDD=st.median(md),worst_MDD=min(md),with_nvda_known=len(wb),with_nvda_median_CAGR=st.median(wb) if wb else ''))
w=csv.DictWriter(open(os.path.join(HERE,'cells_no_nvda_432.csv'),'w',newline=''),fieldnames=list(cells[0])); w.writeheader(); w.writerows(cells)
# ---- month-end proxy (same simplifications as ../literature-review-response-20261113/topk_proxy.py), NVDA excluded:
# covers every path, including the Top3 paths the exact engine cannot price. Validated below against the exact results.
import gzip
def proxy(P,first_signal,excl=('NVDA',),C=0.0025):
    rd=lambda f:list(csv.DictReader(gzip.open(os.path.join(P,f),'rt') if f.endswith('.gz') else open(os.path.join(P,f))))
    px=collections.defaultdict(dict)
    for r in rd('rank_prices_month_end.csv.gz'):
        if r['rank_price']: px[r['engine_ticker']][r['month_end']]=float(r['rank_price'])
    el=collections.defaultdict(list)
    for r in rd('eligibility_by_signal.csv.gz'):
        if r['eligible_final']=='True' and r['engine_ticker'] not in excl: el[r['signal_date']].append(r['engine_ticker'])
    ca=collections.defaultdict(list)
    for r in rd('corporate_actions_applied.csv'):
        if r['kind'] in('spin','special_div') and r['factor_F']: ca[r['engine_ticker']].append((r['ex_date'],float(r['factor_F'])))
    ME=sorted({d for v in px.values() for d in v}); mi={d:i for i,d in enumerate(ME)}; cache={}
    def F(t,a,b):
        f=1.0
        for e,x in ca.get(t,()):
            if a<e<=b: f*=x
        return f
    def top(S,W):
        if (S,W) not in cache:
            B=ME[mi[S]-W]; cache[(S,W)]=[t for _,t in sorted((-(px[t][S]/(px[t][B]*F(t,B,S))-1),t) for t in el[S] if S in px[t] and B in px[t])[:3]]
        return cache[(S,W)]
    out={}; i0=mi[first_signal]
    for W in range(1,13):
      for H in range(1,13):
        for K in (1,2,3):
            res=[]
            for ph in range(12):
                s=i0+ph; e=s+120
                if e>=len(ME) or ME[s+(119//H)*H] not in el: continue
                hold={}; cash=1.0; k=s; pk=0; mdd=0
                while k<e:
                    names=top(ME[k],W)[:K]; E0=cash+sum(hold.values()); Ep=E0
                    for _ in range(100):
                        T=Ep/K; new=E0-C*(sum(abs(T-hold.get(t,0.0)) for t in names)+sum(v for t,v in hold.items() if t not in names))
                        if abs(new-Ep)<1e-13: break
                        Ep=new
                    hold={t:Ep/K for t in names}; cash=0.0
                    for m in range(k+1,min(k+H,e)+1):
                        a,b=ME[m-1],ME[m]
                        for t in list(hold):
                            if b in px[t] and a in px[t]: hold[t]*=px[t][b]/(px[t][a]*F(t,a,b))
                            elif a in px[t]: cash+=hold.pop(t)
                        v=cash+sum(hold.values()); pk=max(pk,v); mdd=min(mdd,v/pk-1)
                    k=min(k+H,e)
                res.append(((cash+sum(hold.values())*(1-C))**0.1-1,mdd))
            out[(W,H,K)]=res
    return out
PX=proxy(PACK,'2014-12-31')
def spearman(x,y):
    def rk(v):
        o=sorted(range(len(v)),key=lambda i:v[i]); r=[0]*len(v)
        for j,i in enumerate(o): r[i]=j
        return r
    a,b=rk(x),rk(y); n=len(x); return 1-6*sum((p-q)**2 for p,q in zip(a,b))/(n*(n*n-1))
for x in cells:
    v=PX[(x['W'],x['H'],x['K'])]; x['proxy_median_CAGR']=st.median(a for a,_ in v); x['proxy_min_CAGR']=min(a for a,_ in v); x['proxy_max_CAGR']=max(a for a,_ in v); x['proxy_median_monthend_MDD']=st.median(b for _,b in v)
for K in (1,2):
    ck=[x for x in cells if x['K']==K and x['known']==12]
    print('proxy validation Top%d (cells with all 12 exact paths, n=%d): Spearman %.3f, mean abs diff %.2f pp'%(K,len(ck),spearman([x['median_CAGR'] for x in ck],[x['proxy_median_CAGR'] for x in ck]),st.mean(abs(x['median_CAGR']-x['proxy_median_CAGR']) for x in ck)*100))
w=csv.DictWriter(open(os.path.join(HERE,'cells_no_nvda_432.csv'),'w',newline=''),fieldnames=list(cells[0])); w.writeheader(); w.writerows(cells)
p05={}
f05=os.path.join(HERE,'..','literature-review-response-20261113','grid_persistence_proxy.csv')
if os.path.exists(f05):
    for r in csv.DictReader(open(f05)): p05[(int(r['W']),int(r['H']))]=float(r['proxy_median_CAGR_2005_2015'])
for K in (1,2,3):
    exact=K<3; key='median_CAGR' if exact else 'proxy_median_CAGR'
    ck=[x for x in cells if x['K']==K and (x['known']>=9 if exact else True)]
    print(f'--- Top{K} ({"exact engine" if exact else "month-end PROXY (exact engine can price only 244 of 1,728 Top3 paths)"}): cells ranked {len(ck)} of 144; median of cell medians {st.median(x[key] for x in ck)*100:.1f}%; cells whose median beats 13%: {sum(1 for x in ck if x[key]>0.13)}')
    for x in sorted(ck,key=lambda x:-x[key])[:6]:
        if exact: print('   W%dH%d  median %.1f%%  range %.1f%%..%.1f%%  median MDD %.0f%% worst %.0f%%  known %d/12 | with NVDA median %.1f%% | proxy %.1f%%%s'%(x['W'],x['H'],x['median_CAGR']*100,x['min_CAGR']*100,x['max_CAGR']*100,x['median_MDD']*100,x['worst_MDD']*100,x['known'],x['with_nvda_median_CAGR']*100,x['proxy_median_CAGR']*100,(' | 2005-15 Top1 proxy %.1f%%'%(p05[(x['W'],x['H'])]*100)) if K==1 and (x['W'],x['H']) in p05 else ''))
        else: print('   W%dH%d  proxy median %.1f%%  range %.1f%%..%.1f%%  month-end MDD median %.0f%% | with NVDA exact median %s'%(x['W'],x['H'],x[key]*100,x['proxy_min_CAGR']*100,x['proxy_max_CAGR']*100,x['proxy_median_monthend_MDD']*100,('%.1f%%'%(x['with_nvda_median_CAGR']*100)) if x['with_nvda_median_CAGR']!='' else 'n/a'))
    kp=[r for r in rows if r['K']==K and r['status']=='KNOWN']
    if exact:
        best=max(kp,key=lambda r:r['CAGR']); print('   best single path:',best['path_id'],'CAGR %.1f%% MDD %.0f%%'%(best['CAGR']*100,best['daily_MDD']*100),'| all %d known paths: median %.1f%%, share below 0: %.0f%%, share above 13%%: %.0f%%'%(len(kp),st.median(r['CAGR'] for r in kp)*100,100*sum(1 for r in kp if r['CAGR']<0)/len(kp),100*sum(1 for r in kp if r['CAGR']>0.13)/len(kp)))
    else:
        allp=[a for (W,H,KK),v in PX.items() if KK==3 for a,_ in v]; print('   all %d proxy paths: median %.1f%%, share below 0: %.0f%%, share above 13%%: %.0f%%, best %.1f%%'%(len(allp),st.median(allp)*100,100*sum(1 for a in allp if a<0)/len(allp),100*sum(1 for a in allp if a>0.13)/len(allp),max(allp)*100))
for c0 in ((12,12,1),(10,2,1),(12,4,1)):
    x=[y for y in cells if (y['W'],y['H'],y['K'])==c0][0]; p1=[r for r in rows if r['path_id']=='W%dH%dK%dP1'%c0][0]
    rk=sorted([y for y in cells if y['K']==1 and y['known']>=9],key=lambda y:-y['median_CAGR']).index(x)+1 if x['known']>=9 else None
    print('anchor W%dH%dK%d: no-NVDA median %.1f%% (known %d, rank %s of 138) vs with NVDA %.1f%%; January path %s'%(c0+(x['median_CAGR']*100,x['known'],rk,x['with_nvda_median_CAGR']*100,('%.1f%% MDD %.0f%%'%(p1['CAGR']*100,p1['daily_MDD']*100)) if p1['status']=='KNOWN' else p1['fail_reason'])))
os.remove(pk)
