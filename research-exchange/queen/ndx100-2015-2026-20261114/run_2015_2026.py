from engine import *
P,TR=qqq()
rows=[]
segs={'2015-01_to_2026-11-13':('2014-12-31','2026-11-13'),'2025-01_to_2026-11-13':('2024-12-31','2026-11-13')}
for pname,(a,b) in segs.items():
    i0,i1=IDX[a],IDX[b]
    for name,K,ex in [('Top1',1,()),('Top3',3,()),('Top5',5,()),('Top10',10,()),('Top3_noNVDA',3,('NVDA',)),('Top5_noNVDA',5,('NVDA',))]:
        for c in (0.0,0.0025):
            s=stats(run(K,i0,i1,ex,c)); s.update(period=pname,variant=name,cost_bps_side=c*1e4); rows.append(s)
    for name,ser in (('QQQ_price',P),('QQQ_total_return',TR)):
        s=stats(bench(ser,i0,i1)); s.update(period=pname,variant=name,cost_bps_side=0); rows.append(s)
cols=['period','variant','cost_bps_side','months','CAGR','MDD_monthly','MDD_daily','worst_month','worst_month_date','total']
with open('out/p2015_2026/results_2015_2026.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
for r in rows: print('%-24s %-12s %4.0fbp m%3d CAGR %6.2f%% total %7.1f%% MDDm %6.1f%% MDDd %6.1f%%'%(r['period'],r['variant'],r['cost_bps_side'],r['months'],r['CAGR']*100,r['total']*100,r['MDD_monthly']*100,(r['MDD_daily'] or 0)*100))
