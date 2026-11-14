"""Re-run the post rule (2015-2024) on Claude's partial universe (members with a price in Claude's S&P pack), using Queen's Massive prices."""
import engine
from engine import *
from claude_mask import build_mask
mask,_,_=build_mask()
i0,i1=IDX['2014-12-31'],IDX['2024-12-31']
def yearly(res):
    nv=dict(res['navs']); ys=[]
    for y in range(2015,2025):
        a=max(d for d in nv if d<'%d-01-01'%y); b=max(d for d in nv if d<'%d-01-01'%(y+1)); ys.append(nv[b]/nv[a]-1)
    return ys
out=[]
for label,M in (('claude_partial_universe',mask),('full_universe',None)):
    engine.MASK=M
    for name,K,ex,c in (('Top1',1,(),.0025),('Top3',3,(),.0025),('Top5',5,(),.0025),('Top10',10,(),.0025),('Top3_noNVDA',3,('NVDA',),.0025),('Top3_zero_cost',3,(),0)):
        res=run(K,i0,i1,ex,c); s=stats(res); y=yearly(res)
        out.append(dict(universe=label,variant=name,CAGR=s['CAGR'],MDD_monthly=s['MDD_monthly'],MDD_daily=s['MDD_daily'],worst_month=s['worst_month'],yearly=' '.join('%+.0f%%'%(v*100) for v in y)))
        print(label,name,'%.1f%% MDDm %.0f%% | %s'%(s['CAGR']*100,s['MDD_monthly']*100,out[-1]['yearly']))
        if name=='Top3': globals()['picks_'+label]=[r['picks'].split() for r in res['rec']]; globals()['rec_'+label]=res['rec']
    r=ew(i0,i1); s=stats(r); out.append(dict(universe=label,variant='EW_members',CAGR=s['CAGR'],MDD_monthly=s['MDD_monthly'],MDD_daily=s['MDD_daily'],worst_month=s['worst_month'],yearly=' '.join('%+.0f%%'%(v*100) for v in yearly(r))))
    print(label,'EW','%.1f%%'%(s['CAGR']*100),out[-1]['yearly'])
engine.MASK=None
with open('out/results_claude_universe_vs_full_2015_2024.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
# month-by-month diff of Top3 picks
rows=[]
for a,b in zip(rec_claude_partial_universe,rec_full_universe):
    if a['picks']!=b['picks']:
        rows.append(dict(signal=a['signal'],claude_univ_picks=a['picks'],claude_univ_net=a['net'],full_picks=b['picks'],full_net=b['net'],
                         new_names=' '.join(sorted(set(b['picks'].split())-set(a['picks'].split()))),diff_net=b['net']-a['net']))
with open('out/top3_pick_diffs_claude_vs_full.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('months differing',len(rows),'of 120')
from collections import defaultdict
# v1.1: when several newly-added names enter the same month, split that month's
# net-return difference equally by count. v1.0 credited the full monthly diff to each name
# (BIDU -22pp / PDD -14pp); equal-split is the fair attribution (Claude review 4aac8b3).
agg_full=defaultdict(lambda:[0,0.0]); agg_split=defaultdict(lambda:[0,0.0])
for r in rows:
    names=[t for t in r['new_names'].split() if t]
    if not names: continue
    share=r['diff_net']/len(names)
    for t in names:
        agg_full[t][0]+=1; agg_full[t][1]+=r['diff_net']
        agg_split[t][0]+=1; agg_split[t][1]+=share
attr=[]
for t in sorted(set(agg_split)|set(agg_full)):
    n,vs=agg_split[t]; _,vf=agg_full[t]
    attr.append(dict(ticker=t,months=n,equal_split_sum=round(vs,6),full_credit_sum_v10=round(vf,6)))
attr.sort(key=lambda x:x['equal_split_sum'])
with open('out/new_name_attribution.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['ticker','months','equal_split_sum','full_credit_sum_v10']); w.writeheader(); w.writerows(attr)
print('sum of monthly net diff',sum(r['diff_net'] for r in rows))
print('equal-split (v1.1) worst:')
for r in attr[:15]: print(r['ticker'],r['months'],'%+.3f'%r['equal_split_sum'],'(v1.0 full-credit %+0.3f)'%r['full_credit_sum_v10'])
print('equal-split (v1.1) best:')
for r in sorted(attr,key=lambda x:-x['equal_split_sum'])[:10]:
    print(r['ticker'],r['months'],'%+.3f'%r['equal_split_sum'],'(v1.0 full-credit %+0.3f)'%r['full_credit_sum_v10'])
