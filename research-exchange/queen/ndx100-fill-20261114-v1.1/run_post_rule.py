from engine import *
P,TR=qqq()
periods={'2007-2014':('2006-12-29','2014-12-31'),'2015-2024':('2014-12-31','2024-12-31'),'2007-2024':('2006-12-29','2024-12-31'),
         '2025-01_to_2026-11-13':('2024-12-31','2026-11-13')}
rows=[]; picks=[]; contribs=[]
variants=[('Top1',1,(),0.0025),('Top3',3,(),0.0025),('Top5',5,(),0.0025),('Top10',10,(),0.0025),
          ('Top3_noNVDA',3,('NVDA',),0.0025),('Top1_noNVDA',1,('NVDA',),0.0025),('Top5_noNVDA',5,('NVDA',),0.0025),('Top10_noNVDA',10,('NVDA',),0.0025),
          ('Top3_zero_cost',3,(),0.0)]
for pname,(a,b) in periods.items():
    i0,i1=IDX[a],IDX[b]
    for name,K,ex,c in variants:
        res=run(K,i0,i1,ex,c); s=stats(res); s.update(period=pname,variant=name); rows.append(s)
        for r in res['rec']: picks.append(dict(period=pname,variant=name,**r))
        top=sorted(res['contrib'].items(),key=lambda x:-x[1])[:10]
        contribs.append(dict(period=pname,variant=name,top_contributors=' '.join('%s:%+.2f(%dm)'%(t,v,res['held'][t]) for t,v in top),
                             worst=' '.join('%s:%+.2f'%(t,v) for t,v in sorted(res['contrib'].items(),key=lambda x:x[1])[:5])))
    for name,ex in (('EW_members',()),('EW_members_noNVDA',('NVDA',))):
        res=ew(i0,i1,ex); s=stats(res); s.update(period=pname,variant=name); rows.append(s)
    for name,ser in (('QQQ_price',P),('QQQ_total_return',TR)):
        res=bench(ser,i0,i1); s=stats(res); s.update(period=pname,variant=name); rows.append(s)
cols=['period','variant','months','CAGR','MDD_monthly','MDD_daily','worst_month','worst_month_date','best_month','turnover_oneway_per_year','total']
with open('out/results_post_rule.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
with open('out/picks_by_month.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(picks[0])); w.writeheader(); w.writerows(picks)
with open('out/top_contributors.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(contribs[0])); w.writeheader(); w.writerows(contribs)
for r in rows: print('%-22s %-18s CAGR %6.1f%%  MDDm %5.0f%%  MDDd %5.0f%%  worst %+5.0f%% %s  TO %4.1fx'%(r['period'],r['variant'],r['CAGR']*100,r['MDD_monthly']*100,(r['MDD_daily'] or 0)*100,r['worst_month']*100,r['worst_month_date'],r['turnover_oneway_per_year']))
for c in contribs:
    if c['variant'] in ('Top3','Top3_noNVDA'): print(c)
