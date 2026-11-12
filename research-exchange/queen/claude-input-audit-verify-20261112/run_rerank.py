import csv,json,collections,sys
sys.path.insert(0,'.')
from queen_rerank import *
# ---------- corrections accepted from Massive evidence ----------
F_GOOG=571.50/(571.50+569.74)      # Massive raw 2014-04-03 GOOGL / (GOOGL+GOOG)
F_DISCA=40.975/(40.975+39.78)      # Massive 2014-08-07 DISCA / (DISCA+DISCK)
TT_DIV=(100.55+0.8824*32.80)/100.55  # Massive 2020-03-02 TT + 0.8824 x new IR, over TT
IRW=[S for S in SIG if '2020-03-31'<=S<='2021-02-26']
C9={'scale':[('GOOGL','2014-03-31',F_GOOG),('GOOG','2014-03-31',F_GOOG),('TMK','2014-06-30',2/3),('DISCA','2014-07-31',F_DISCA),
     ('BF.B','2016-07-29',0.5),('BF.B','2018-02-28',0.8),('BLL','2017-04-28',0.5),('BKNG','2018-01-31',1/25)],
    'replace_series':{'TT':('IR','2020-02-28',TT_DIV)},
    'elig_remove':{(S,'IR') for S in IRW},'elig_add':{(S,'TT') for S in IRW}}
NEW={'scale':[('EQT','2018-10-31',1837/1000),('ARNC','2016-10-31',1/0.801),('HWM','2016-10-31',1/0.801)]}
C11={k:(v if not isinstance(v,list) else list(v)) for k,v in C9.items()}; C11['scale']=C9['scale']+NEW['scale']
variants={'pack':{},'claude9':C9,'claude9_plus_EQT_ARNC':C11,'EQT_ARNC_only':NEW}
res={k:all_tables(v) for k,v in variants.items()}
print('factors: GOOG',F_GOOG,'DISCA',F_DISCA,'TT divisor',TT_DIV)
# validate pack baseline against answer key picks (K=3 path rebalances list Top3 in rank order)
ak=collections.defaultdict(set)
import gzip
for r in csv.DictReader(gzip.open('/workspace/momentum432-answer-key-full-20261109/rebalances.csv.gz','rt')):
    ak[(r['signal_date'],int(r['W']),int(r['K']))].add(r['picks'])
mm=0;n=0
for (S,W,K),ps in ak.items():
    tab,bad=res['pack'][(S,W)]
    if tab is None: continue
    n+=1
    mine='|'.join(t for t,_ in tab[:K])
    if ps!={mine}: mm+=1; print('AK mismatch',S,W,K,ps,mine) if mm<5 else None
print('answer-key (S,W,K) checks',n,'mismatches',mm)
unk=[k for k,v in res['pack'].items() if v[0] is None]
print('UNKNOWN tables (pack):',len(unk),sorted(set(s for s,_ in unk)))
rows=[]
for var in ('claude9','claude9_plus_EQT_ARNC','EQT_ARNC_only'):
    cnt=collections.Counter()
    for key in sorted(res['pack']):
        a,_=res['pack'][key];b,_=res[var][key]
        if a is None or b is None:
            if (a is None)!=(b is None): cnt['status_change']+=1
            continue
        ra={t:i+1 for i,(t,_) in enumerate(a)};rb={t:i+1 for i,(t,_) in enumerate(b)}
        ch={}
        for k in (1,2,3,5,10):
            A=[t for t,_ in a[:k]];Bk=[t for t,_ in b[:k]]
            if A!=Bk: cnt[f'top{k}_order_changed']+=1
            if set(A)!=set(Bk): cnt[f'top{k}_set_changed']+=1
            ch[k]=(A!=Bk)
        if any(ch.values()):
            rows.append(dict(variant=var,signal_date=key[0],W=key[1],pack_top10='|'.join(t for t,_ in a[:10]),corr_top10='|'.join(t for t,_ in b[:10]),
               top1_changed=ch[1],top2_changed=ch[2],top3_changed=ch[3],top5_changed=ch[5],top10_changed=ch[10],
               pack_ret3=round(a[2][1],6),corr_ret3=round(b[2][1],6)))
    print(var,dict(cnt),'tables compared',sum(1 for k in res['pack'] if res['pack'][k][0] is not None))
w=csv.DictWriter(open('rerank_changes.csv','w',newline=''),fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# detail for specific cases
def show(var,S,W,names):
    tab,_=res[var][(S,W)];r={t:(i+1,ret) for i,(t,ret) in enumerate(tab)}
    return {n:(r[n][0],round(r[n][1]*100,2)) if n in r else None for n in names}|{'top3_ret%':round(tab[2][1]*100,2)}
for S,W,n in [('2018-04-30',4,['BKNG']),('2020-07-31',1,['TT']),('2020-10-30',4,['TT'])]:
    print(S,W,'pack',show('pack',S,W,n),'claude9',show('claude9',S,W,n))
# closest misses to Top3 among corrected names
names9={'GOOGL','GOOG','TMK','DISCA','BF.B','BLL','BKNG','TT','IR'}
best=[]
for key,(tab,_) in res['claude9'].items():
    if tab is None: continue
    for i,(t,ret) in enumerate(tab):
        if t in names9 and i>=3:
            best.append((round((tab[2][1]-ret)*100,2),key,t,i+1))
best.sort();print('closest misses claude9:',best[:6])
import pickle;pickle.dump(res,open('/tmp/rerank_res.pkl','wb'))
