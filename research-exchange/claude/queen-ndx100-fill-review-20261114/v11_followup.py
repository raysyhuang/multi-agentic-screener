#!/usr/bin/env python3
"""Claude: follow-up on Queen's v1.1 (commit 909cd93): re-run my checks on the v1.1 prices, publish my month-by-month NAV for the
2007-2014 Top10 path and locate the months behind the 1.5pp drawdown difference.  Usage: python3 v11_followup.py"""
import os,csv,collections
H=os.path.dirname(os.path.abspath(__file__)); V=os.path.join(H,'..','..','queen','ndx100-fill-20261114-v1.1')
src=open(os.path.join(H,'verify_queen_ndx.py')).read().split("P=lambda x:")[0].replace("Q=os.path.join(H,'..','..','queen','ndx100-fill-20261114')","Q=V")
exec(src)
R=Run(); i15=IDX['2014-12-31']; i07=IDX['2006-12-29']
print('BIIB 2017-01→02 (v1.1):','%+.2f%%'%((px['BIIB']['2017-02-28']/px['BIIB']['2017-01-31']-1)*100))
for lab,i0,n in (('2015-2024',i15,120),('2007-2014',i07,96)):
    for K in (1,3,5,10):
        e,c,m=R.path('M6',K,1,i0,n); print(f'{lab} Top{K}: {c*100:.1f}% / 月末回撤 {m*100:.1f}%')
lg=[]; e,c,m=R.path('M6',10,1,i07,96,lg)
qr=[r for r in csv.DictReader(open(os.path.join(V,'out','picks_by_month.csv'))) if r['period']=='2007-2014' and r['variant']=='Top10']
w=csv.writer(open(os.path.join(H,'claude_nav_2007_2014_top10.csv'),'w',newline='')); w.writerow(['signal','end','picks','stock_rets','claude_month_ret_after_cost','queen_net','diff','claude_nav_after_costs','picks_same'])
dif=[]
for j,((a,b,pr,g),q) in enumerate(zip(lg,qr)):
    same=set(pr)==set(q['picks'].split()); d=g-float(q['net'])
    w.writerow([a,b,' '.join(pr),' '.join('NA' if v is None else '%+.4f'%v for v in pr.values()),'%.6f'%g,q['net'],'%.6f'%d,'%.6f'%e[j+1],same])
    if abs(d)>0.002 or not same: dif.append((a,same,round(d*100,2),[t for t,v in pr.items() if v is None],sorted(set(pr)^set(q['picks'].split()))))
print('months compared',len(qr),'| months with different picks or |净收益差|>0.2pp:',len(dif))
for x in dif: print('  ',x)
pk=0;tr=None
for j,v in enumerate(e):
    if v>pk: pk=v;pj=j
    if tr is None or v/pk-1<tr[0]: tr=(v/pk-1,pj,j)
print('my drawdown: peak after signal',ME[i07+tr[1]],'trough',ME[i07+tr[2]],'%.1f%%'%(tr[0]*100))
