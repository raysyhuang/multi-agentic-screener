"""Rebuild paths from folder A alone (rebuild_full.py) and diff against the answer key folder B.
Usage: python3 verify_full.py [B_DIR] [all|sample]   (default B_DIR=../momentum432-answer-key-full-20261109, all)
sample = all K2/K3 of W10H2, W12H4, W12H12 + 200 random other paths (seed 20261109)."""
import sys,csv,gzip,random,collections,multiprocessing as mp,json
import rebuild_full as R
B=sys.argv[1] if len(sys.argv)>1 else '../momentum432-answer-key-full-20261109';MODE=sys.argv[2] if len(sys.argv)>2 else 'all'
ALL=[(W,H,K,p) for W in range(1,13) for H in range(1,13) for K in (1,2,3) for p in range(1,13)]
if MODE=='sample':
    core=[x for x in ALL if (x[0],x[1]) in((10,2),(12,4),(12,12)) and x[2] in(2,3)]
    rest=[x for x in ALL if x not in set(core)];random.seed(20261109);JOBS=core+random.sample(rest,200)
else: JOBS=ALL
def run(x):
    o=R.path(*x,want_nav=True);nav=o.pop('nav',None)
    if nav: o['nav_last']=nav[-1];o['nav']=nav
    return x,o
if __name__=='__main__':
    with mp.get_context('fork').Pool(8) as pool: OUT=dict(pool.map(run,JOBS,chunksize=8))
    ref={r['path_id']:r for r in csv.DictReader(open(B+'/paths_summary.csv'))}
    rb=collections.defaultdict(list)
    for r in csv.DictReader(gzip.open(B+'/rebalances.csv.gz','rt')): rb[r['path_id']].append((r['signal_date'],r['trade_date'],r['picks']))
    navf={}
    for W in range(1,13):
        rows=list(csv.reader(gzip.open(f'{B}/nav_daily/nav_daily_W{W:02d}.csv.gz','rt')));hdr=rows[0]
        for j,pid in enumerate(hdr[1:],1): navf[pid]=[float(r[j]) for r in rows[1:] if r[j]!='']
    cnt=collections.Counter();mx=collections.defaultdict(float);bad=[]
    for (W,H,K,p),o in OUT.items():
        pid=f'W{W}H{H}K{K}P{p}';r=ref[pid];ok=True
        if o['status']!=r['status']: ok=False
        if r['status']=='UNKNOWN':
            if o.get('fail_reason')!=r['fail_reason'] or str(o.get('fail_ticker') or '')!=r['fail_ticker'] or str(o.get('fail_date') or '')!=r['fail_date']: ok=False
        else:
            for k,rk in(('final','final_multiple'),('CAGR','CAGR'),('MDD','daily_MDD')):
                d=abs(o[k]-float(r[rk]))/(abs(float(r[rk])) if k=='final' else 1);mx[k]=max(mx[k],d);ok&=d<1e-9
            nv=navf[pid];d=max(abs(a/b-1) for a,b in zip(o['nav'],nv)) if len(nv)==len(o['nav']) else 9;mx['nav_daily_rel']=max(mx['nav_daily_rel'],d);ok&=d<1e-9
        if r['status']!='UNKNOWN' and o['n_trades']!=int(r['n_trades']): ok=False;cnt['n_trades_diff']+=1
        if rb[pid] and o['rebalances'][:len(rb[pid])]!=rb[pid]: ok=False;cnt['picks_diff']+=1   # B has no ledger for paths the engine aborted by exception
        cnt['rebalance_rows_compared']+=len(rb[pid])
        cnt['match' if ok else 'mismatch']+=1;cnt[r['status']]+=1
        if not ok: bad.append((pid,r['status'],r['fail_reason'],o['status'],o.get('fail_reason')))
    print(json.dumps(dict(mode=MODE,paths=len(OUT),counts=cnt,max_diff={k:float(v) for k,v in mx.items()},n_rebalance_rows_compared=sum(len(o['rebalances']) for o in OUT.values())),indent=1))
    for b in bad[:30]: print('MISMATCH',b)
