"""2005-2015 full grid (W1-12 x H1-12 x K1-3 x 12 phases = 5,184) + blended sleeves (W10-12 x H{1,3,6,12} x K{5,10,20} x 12)
with the exact pack engine pack0515/rebuild_2005_2015.py (stdlib copy of frozen rebuild_full.py, years -> 2005/2015).
Only change: a rebalance whose signal month-end is outside the pack's eligibility (130 signals, 2004-12-31..2015-09-30)
is UNKNOWN 'signal_outside_pack' instead of silently going to cash."""
import os,sys,csv,pickle,multiprocessing as mp
PK='/workspace/momentum-followup-20261114/pack0515/';OUT='/workspace/momentum-followup-20261114/'
R=None
def init():
    global R
    os.chdir(PK);sys.path.insert(0,PK);sys.argv=[sys.argv[0]]
    import rebuild_2005_2015 as M;R=M
    PACK=set(k for k,v in R.ELIG.items() if v);orig=R.rank
    def rank(S,W):
        if S not in PACK: return ([],['__SIGNAL_OUTSIDE_PACK__'],set())
        return orig(S,W)
    R.rank=rank
BL={(W,H,K) for W in (10,11,12) for H in (1,3,6,12) for K in (1,3,5,10,20)}
def job(j):
    W,H,K,ph=j;keep=(W,H,K) in BL
    o=R.path(W,H,K,ph,want_nav=keep)
    if o['status']=='UNKNOWN' and o.get('fail_ticker')=='__SIGNAL_OUTSIDE_PACK__': o['fail_reason']='signal_outside_pack';o['fail_ticker']=''
    D0=R.first_td(2005,ph);END=R.first_td(2015,ph)
    row=dict(W=W,H=H,K=K,phase=ph,entry=D0,end=END,status=o['status'],CAGR=o.get('CAGR',''),MDD=o.get('MDD',''),final=o.get('final',''),
             n_trades=o['n_trades'],n_rebalances=len(o['rebalances']),fail_reason=o.get('fail_reason',''),fail_ticker=o.get('fail_ticker',''),fail_date=o.get('fail_date',''))
    extra=None
    if keep:
        dates=R.SES[R.SI[D0]:R.SI[END]+1]
        extra=dict(nav=o.get('nav'),dates=dates if o.get('nav') else None,rebalances=o['rebalances'])
    return j,row,extra
if __name__=='__main__':
    jobs=[(W,H,K,ph) for W in range(1,13) for H in range(1,13) for K in (1,2,3) for ph in range(1,13)]
    jobs+=[(W,H,K,ph) for W in (10,11,12) for H in (1,3,6,12) for K in (5,10,20) for ph in range(1,13)]
    rows=[];ext={}
    with mp.Pool(7,initializer=init) as p:
        for i,(j,row,extra) in enumerate(p.imap_unordered(job,jobs,chunksize=8)):
            rows.append(row)
            if extra: ext[j]=extra
            if i%500==0: print(i,flush=True)
    rows.sort(key=lambda r:(r['W'],r['H'],r['K'],r['phase']))
    with open(OUT+'results/r0515/paths_0515_all.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    pickle.dump(ext,open(OUT+'scratch/nav0515_blend.pkl','wb'))
    print('done',len(rows))
