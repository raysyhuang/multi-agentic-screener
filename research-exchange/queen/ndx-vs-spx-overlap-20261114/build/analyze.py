import pickle,bisect,csv,collections,numpy as np
O='/workspace/ndx-vs-spx-overlap-20261114/'
n=pickle.load(open(O+'build/ndx.pkl','rb'));s=pickle.load(open(O+'build/spx.pkl','rb'))
CAN={'FB':'META','PCLN':'BKNG','YHOO':'AABA','KORS':'CPRI'}
cn=lambda t:CAN.get(t,t)
md=[d for d,_ in s['sp_mem']];sm=[set(x) for _,x in s['sp_mem']]
sp_at=lambda d:sm[bisect.bisect_right(md,d)-1]
def W(path,rows,cols=None):
    cols=cols or list(rows[0].keys())
    with open(O+path,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
# 1 membership overlap
mo=[]
for d in sorted(n['mem']):
    if not('2015-01'<=d<'2025-01'):continue
    N=[cn(t) for t in n['mem'][d]];S=sp_at(d);Ss=set(s['seasoned'].get(d,[]))
    ins=[t for t in N if t in S]
    mo.append(dict(month_end=d,ndx_members=len(N),in_sp500=len(ins),share_in_sp500=round(len(ins)/len(N),4),
       in_sp500_seasoned_universe=sum(t in Ss for t in N),share_in_seasoned=round(sum(t in Ss for t in N)/len(N),4),
       sp_snapshot=md[bisect.bisect_right(md,d)-1],not_in_sp500=' '.join(sorted(t for t in N if t not in S))))
W('membership_overlap_by_month.csv',mo)
sh=[r['share_in_sp500'] for r in mo];shs=[r['share_in_seasoned'] for r in mo]
print('membership share in SP500: mean %.3f min %.3f max %.3f; first %.3f last %.3f; seasoned mean %.3f'%(np.mean(sh),min(sh),max(sh),sh[0],sh[-1],np.mean(shs)))
# 2 picks
def sp_picks(key):
    return {x['signal']:[cn(t) for t in (x['picks'] or '').split('|') if t] for x in s[key]['rb']}
SPW=sp_picks('W6H1K3P1_total_ext_LENIENT');SPM=sp_picks('mean6_H1K3P1_ext_LENIENT');SPMo=sp_picks('mean6_H1K3P1_official_LENIENT');SPWo=sp_picks('W6H1K3P1_total_official')
# sanity: ext picks identical to official in overlap
for a,b in ((SPW,SPWo),(SPM,SPMo)):
    assert all(a[k]==b[k] for k in b), 'ext/official pick mismatch'
rank_m={S:[cn(t) for t,_ in v['top10']] for S,v in s['mean6_H1K3P1_ext_LENIENT']['rank'].items()}
pk=[];contrib=collections.defaultdict(float);cnt=collections.Counter()
for r in n['top3']['rec']:
    S=r['signal']
    if S>='2026-10': continue
    N=[cn(t) for t in r['picks'].split()];rets=[float(x) for x in r['stock_rets'].split()]
    SP=sp_at(S);m=SPM.get(S,[]);w=SPW.get(S,[])
    for t,x in zip(N,rets):
        if t not in SP: contrib[(t,'2015-2024' if S<'2024-12-31' else '2025-2026')]+=x/3;cnt[(t,'2015-2024' if S<'2024-12-31' else '2025-2026')]+=1
    pk.append(dict(signal=S,ndx_top3=' '.join(N),ndx_stock_rets=r['stock_rets'],ndx_net=round(r['net'],6),
        ndx_picks_in_sp500=sum(t in SP for t in N),sp_mean6_top3=' '.join(m),overlap_with_sp_mean6=len(set(N)&set(m)),
        sp_W6total_top3=' '.join(w),overlap_with_sp_W6total=len(set(N)&set(w)),
        ndx_picks_in_sp_mean6_top10=sum(t in rank_m.get(S,[]) for t in N),
        in_sp_seasoned_universe=sum(t in set(s['seasoned'].get(S,[])) for t in N)))
W('pick_overlap_by_month.csv',pk)
def summ(rows,lab):
    T=3*len(rows)
    print(lab,'months',len(rows),'NDX picks in SP500 %.1f%%'%(100*sum(r['ndx_picks_in_sp500'] for r in rows)/T),
      'in SP seasoned univ %.1f%%'%(100*sum(r['in_sp_seasoned_universe'] for r in rows)/T),
      'also SP mean6 top3 %.1f%%'%(100*sum(r['overlap_with_sp_mean6'] for r in rows)/T),
      'also SP W6total top3 %.1f%%'%(100*sum(r['overlap_with_sp_W6total'] for r in rows)/T),
      'in SP mean6 top10 %.1f%%'%(100*sum(r['ndx_picks_in_sp_mean6_top10'] for r in rows)/T),
      'months identical mean6 %d'%sum(r['overlap_with_sp_mean6']==3 for r in rows),'months zero overlap mean6 %d'%sum(r['overlap_with_sp_mean6']==0 for r in rows))
    return dict(window=lab,months=len(rows),ndx_pick_slots=T,share_in_sp500=round(sum(r['ndx_picks_in_sp500'] for r in rows)/T,4),
      share_in_sp_seasoned_universe=round(sum(r['in_sp_seasoned_universe'] for r in rows)/T,4),
      share_also_sp_mean6_top3=round(sum(r['overlap_with_sp_mean6'] for r in rows)/T,4),share_also_sp_W6total_top3=round(sum(r['overlap_with_sp_W6total'] for r in rows)/T,4),
      share_in_sp_mean6_top10=round(sum(r['ndx_picks_in_sp_mean6_top10'] for r in rows)/T,4),
      months_all3_same_mean6=sum(r['overlap_with_sp_mean6']==3 for r in rows),months_zero_overlap_mean6=sum(r['overlap_with_sp_mean6']==0 for r in rows))
ps=[summ([r for r in pk if r['signal']<'2024-12-31'],'2015-2024'),summ([r for r in pk if r['signal']>='2024-12-31'],'2025-01..2026-09'),summ(pk,'2015-01..2026-09')]
W('pick_overlap_summary.csv',ps)
cr=[dict(ticker=t,window=wd,months_held_while_not_in_sp500=cnt[(t,wd)],sum_contribution_simple=round(v,4)) for (t,wd),v in sorted(contrib.items(),key=lambda x:-x[1])]
W('non_sp500_ndx_picks_contribution.csv',cr)
print('top non-SP contributors',[(r['ticker'],r['window'],r['months_held_while_not_in_sp500'],r['sum_contribution_simple']) for r in cr[:12]]); print('worst',[(r['ticker'],r['window'],r['sum_contribution_simple']) for r in cr[-5:]])
# 3 performance
def mdd(v):
    v=np.asarray(v);return float((v/np.maximum.accumulate(v)-1).min())
def st(lab,dates,vals,dd,dv,start,end,note=''):
    yrs=(np.datetime64(end)-np.datetime64(start)).astype(int)/365.25
    return dict(series=lab,start=start,end=end,ending_multiple=round(vals,4),CAGR=round(vals**(1/yrs)-1,4),daily_MDD=round(mdd(dv),4),note=note)
def ndx_seg(key,end):
    navs=n[key]['navs'];dn=n[key]['dnav'];days=n['days']
    j=[d for d,_ in navs].index(end);k=days.index(end)
    m=navs[j][1];yrs=j/12
    return dict(series='NDX post rule Top3 (6m mean, ME close, 25bp)' if key=='top3' else 'NDX post rule Top3, universe = NDX∩S&P500 PIT',start='2014-12-31',end=end,ending_multiple=round(m,4),CAGR=round(m**(1/yrs)-1,4),daily_MDD=round(mdd(dn[:k+1]),4),note='month-count CAGR (%d months)'%j)
def sp_seg(key,lab):
    x=s[key];eq=[1.0]+list(x['eq']);days=x['days']; m=eq[-1]  # engine wealth vs initial capital 1 (x['eq'][0] is after entry cost)
    yrs=(np.datetime64(days[-1])-np.datetime64(days[0])).astype(int)/365.25
    r=x['row']
    return dict(series=lab,start=days[0],end=days[-1],ending_multiple=round(m,4),CAGR=round(m**(1/yrs)-1,4),daily_MDD=round(mdd(eq),4),
                note='engine status %s; engine CAGR %s'%(r['status'],None if r['CAGR'] is None else round(r['CAGR'],4)))
def bench(lab,ser,a,b):
    ks=[k for k in sorted(ser) if a<=k<=b];v=[ser[k]/ser[ks[0]] for k in ks]
    yrs=(np.datetime64(ks[-1])-np.datetime64(ks[0])).astype(int)/365.25
    return dict(series=lab,start=ks[0],end=ks[-1],ending_multiple=round(v[-1],4),CAGR=round(v[-1]**(1/yrs)-1,4),daily_MDD=round(mdd(v),4),note='price only')
perf=[]
for wl,ndx_end,sp_suffix,spy_a,spy_b in (('2015-2024','2024-12-31','official','2014-12-31','2024-12-31'),('2015-01..2026-09','2026-09-30','ext','2014-12-31','2026-09-30')):
    rows=[ndx_seg('top3',ndx_end),ndx_seg('top3_spx_members_only',ndx_end)]
    if sp_suffix=='official':
        rows+= [sp_seg('mean6_H1K3P1_official_LENIENT','S&P500 same signal Top3 (6m mean, next first-session close, 25bp) LENIENT: ANDV 2018-09-28 not tradable -> next rank'),
                sp_seg('W6H1K3P1_total_official','S&P500 W6H1K3P1 (6m total return) official answer-key path'),
                sp_seg('mean6_H1K3P1_official_LENIENT_NOSEASON','S&P500 same signal Top3 SENSITIVITY no 365-day seasoning (raw PIT snapshot; names without 6m rank prices dropped)')]
    else:
        rows+= [sp_seg('mean6_H1K3P1_ext_LENIENT','S&P500 same signal Top3 EXTENDED UNAUDITED LENIENT'),sp_seg('W6H1K3P1_total_ext_LENIENT','S&P500 W6H1K3P1 EXTENDED UNAUDITED LENIENT')]
    rows+=[bench('QQQ price',n['qqq'],spy_a,spy_b),bench('SPY price',s['spy'],spy_a,spy_b)]
    for r in rows: r['window']=wl
    perf+=rows
for r in perf: r['status']='PROVISIONAL'
W('performance_side_by_side.csv',perf,['window','series','start','end','ending_multiple','CAGR','daily_MDD','note','status'])
for r in perf: print(r['window'],'|',r['series'][:60],'|',r['start'],r['end'],'| x%.2f CAGR %.2f%% MDD %.1f%%'%(r['ending_multiple'],100*r['CAGR'],100*r['daily_MDD']))
# monthly series + correlation (aligned by signal month-end; S&P period = trade day after signal to next trade day)
def sp_monthly(key):
    x=s[key];eq=x['eq'];days=x['days'];idx={d:i for i,d in enumerate(days)};rb=x['rb'];out={}
    tds=[r['trade'] for r in rb]+[days[-1]]
    for r,a,b in zip(rb,tds,tds[1:]): out[r['signal']]=eq[idx[b]]/eq[idx[a]]-1
    return out
SMm=sp_monthly('mean6_H1K3P1_ext_LENIENT');SMw=sp_monthly('W6H1K3P1_total_ext_LENIENT')
q=n['qqq'];spy=s['spy'];ME=n['ME']
mr=[]
for r,r2 in zip(n['top3']['rec'],n['top3_spx_members_only']['rec']):
    S=r['signal']
    if S not in SMm: continue
    e=r['end']
    mr.append(dict(signal=S,end=e,ndx_top3=round(r['net'],6),ndx_top3_spx_members_only=round(r2['net'],6),sp_mean6_top3=round(SMm[S],6),sp_W6total_top3=round(SMw[S],6),
        qqq=round(q[e]/q[S]-1,6),spy=round(spy[e]/spy[S]-1,6) if e in spy and S in spy else ''))
W('monthly_returns_aligned.csv',mr)
cors=[]
for lab,f in (('2015-2024',lambda r:r['signal']<'2024-12-31'),('2015-01..2026-09',lambda r:True)):
    R=[r for r in mr if f(r)]
    for a,b in (('ndx_top3','sp_mean6_top3'),('ndx_top3','sp_W6total_top3'),('ndx_top3','qqq'),('sp_mean6_top3','spy'),('ndx_top3','ndx_top3_spx_members_only')):
        c=float(np.corrcoef([r[a] for r in R],[r[b] for r in R])[0,1]);cors.append(dict(window=lab,a=a,b=b,months=len(R),corr=round(c,4)))
        print(lab,a,b,len(R),round(c,3))
W('monthly_correlation.csv',cors)
