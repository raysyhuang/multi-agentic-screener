"""Extended check (not needed to use the pack): (1) rebuild all 1,452 ranking frames (121 signals x W1..12) from the pack
inputs with reference_rebuild.rank() and compare to reference_engine_rankings_v18.csv.gz (order, returns, unrankable sets);
(2) rebuild all 36 K=1 paths (W10H2, W12H4, W12H12 x 12 phases) and compare picks/CAGR/MDD to reference_engine_paths_K1.csv."""
import csv,gzip,collections,reference_rebuild as R
E=collections.defaultdict(lambda:([],set()))
for r in csv.DictReader(gzip.open('reference_engine_rankings_v18.csv.gz','rt')):
    k=(r['signal_date'],int(r['W']))
    if r['rank']: E[k][0].append((float(r['lookback_return']),r['engine_ticker']))
    else: E[k][1].add(r['engine_ticker'])
bad=0;maxd=0;n=0
for (S,W),(er,em) in sorted(E.items()):
    r,m=R.rank(S,W);n+=1
    if [t for _,t in r]!=[t for _,t in er] or set(m)!=em: bad+=1;print('MISMATCH',S,W);continue
    maxd=max([maxd]+[abs(a[0]-b[0]) for a,b in zip(r,er)])
print(f'rank frames checked={n} mismatched={bad} max|return diff|={maxd:.2e}')
REF={(r['config'],r['phase']):r for r in csv.DictReader(open('reference_engine_paths_K1.csv'))}
ENG=collections.defaultdict(list)
for r in csv.DictReader(open('execution_prices_picks.csv')): ENG[(r['config'],r['phase'])].append((r['signal_date'],r['engine_ticker']))
ok=0
for W,H in [(12,12),(10,2),(12,4)]:
    for ph in range(1,13):
        cfg=f'W{W}H{H}K1';o=R.path(W,H,ph);ref=REF[(cfg,str(ph))]
        pk=[(s,t) for s,t,_,_ in o['picks']]==ENG[(cfg,str(ph))]
        if ref['status']=='UNKNOWN': good=o['status']!='OK'
        else: good=pk and o['status']=='OK' and abs(o['CAGR']-float(ref['CAGR']))<1e-10 and abs(o['MDD']-float(ref['daily_MDD']))<1e-10
        ok+=good
        print(cfg,ph,o['status'],'picks_match',pk,'CAGR %.4f%%'%(o['CAGR']*100) if 'CAGR' in o else '', 'engine %s'%(ref['CAGR'] and '%.4f%%'%(float(ref['CAGR'])*100)),'OK' if good else 'FAIL')
print(f'K1 paths reproduced {ok}/36')
import statistics
for cfg in ['W12H12K1','W10H2K1','W12H4K1']:
    W,H=int(cfg[1:cfg.index('H')]),int(cfg[cfg.index('H')+1:-2]);xs=[R.path(W,H,p) for p in range(1,13)]
    if all('CAGR' in x for x in xs): print(cfg,'12-phase median CAGR %.4f%%'%(statistics.median(x['CAGR'] for x in xs)*100),'median MDD %.4f%%'%(statistics.median(x['MDD'] for x in xs)*100))
