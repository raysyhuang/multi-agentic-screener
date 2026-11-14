"""2026 live track of the post rule: signals 2026-04-30 .. 2026-10-30, last holding period ends 2026-11-13 (partial month)."""
from engine import *
P,TR=qqq()
i0,i1=IDX['2026-04-30'],IDX['2026-11-13']
rows=[]
NDX={r['month']:float(r['close']) for r in csv.DictReader(open('raw/ndx_index_month_massive.csv'))}
for lab,c in (('post_style_no_cost',0.0),('queen_25bp',0.0025)):
    res=run(3,i0,i1,(),c,daily=False); nav=1; q=1
    for r in res['rec']:
        qr=P[r['end']]/P[r['signal']]-1; nav*=1+r['net']; q*=1+qr
        rows.append(dict(version=lab,signal=r['signal'],end=r['end'],picks=r['picks'],signal_scores=r['scores'],stock_rets=r['stock_rets'],
                         port_ret=round(r['net'],6),cost=round(r['cost'],6),QQQ_price_ret=round(qr,6),NDX_index_ret=round(NDX[r['end'][:7]]/NDX[r['signal'][:7]]-1,6),cum_port=round(nav,6),cum_QQQ=round(q,6)))
with open('out/live_track_2026.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(r['version'],r['signal'],'->',r['end'],r['picks'],r['stock_rets'],'port %+.2f%% QQQ %+.2f%% NDX %+.2f%% cum %.3f vs %.3f'%(r['port_ret']*100,r['QQQ_price_ret']*100,r['NDX_index_ret']*100,r['cum_port'],r['cum_QQQ']))
# rankings at 2026-04-30 and 2026-05-29 (top 10)
out=[]
for d in ('2026-04-30','2026-05-29'):
    for n,(s,t,k) in enumerate(ranking(IDX[d])[:10],1): out.append(dict(signal=d,rank=n,ticker=t,mean6=round(s,6)))
with open('out/post_rankings_2026.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
