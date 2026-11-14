"""run the 25 Queen v18 paths (W10H2K1 x12, W12H4K1 x12, W12H12K1P1) on this pack; writes ../results/"""
import sys,csv,json,statistics,datetime as dt
sys.argv=[sys.argv[0]]
import rebuild_2005_2015 as R
OUT='../results/'
rows=[];navs={};logs={}
for W,H,ph in [(10,2,p) for p in range(1,13)]+[(12,4,p) for p in range(1,13)]+[(12,12,1)]:
    o=R.path(W,H,1,ph,want_nav=True); k=f'W{W}H{H}K1P{ph}'
    logs[k]=o['rebalances']
    if o.get('nav'): navs[k]=o['nav']
    rows.append(dict(path=k,W=W,H=H,K=1,phase=ph,status=o['status'],entry=o.get('entry',''),end=o.get('end',''),
        final_equity=o.get('final',''),CAGR=o.get('CAGR',''),MDD=o.get('MDD',''),n_rebalances=len(o['rebalances']),n_trades=o['n_trades'],
        fail_reason=o.get('fail_reason',''),fail_ticker=o.get('fail_ticker',''),fail_date=o.get('fail_date','')))
    print(rows[-1])
json.dump({'logs':logs},open(OUT+'rebalance_logs.json','w'),indent=0)
json.dump(navs,open(OUT+'_navs.json','w'))
w=csv.DictWriter(open(OUT+'results_paths.csv','w',newline=''),fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
