"""List every (label, window) the 2005-2015 grid (+ skip-month W10H2/W12H4 K1) would execute that has no pack row and no
valid extension series. Output needs_2005_missing.csv (label, symbols, first/last window)."""
import sys,csv,collections;sys.path.insert(0,'.')
import engine_ext as M
E=M.Eng('/workspace/w10h2-w12h4-oos-2005-2015-20261112/pack',2005)
need=set()
def sched(H,ph):
    D0=E.first_td(2005,ph);END=E.first_td(2015,ph);rebs=[];j=0
    while True:
        mm=(ph-1)+j*H;d=E.first_td(2005+mm//12,mm%12+1)
        if d>=END: break
        rebs.append(d);j+=1
    return rebs,END
jobs=[(W,H,K,ph,False) for W in range(1,13) for H in range(1,13) for K in (1,2,3) for ph in range(1,13)]+[(W,H,1,ph,True) for W,H in((10,2),(12,4),(12,12)) for ph in range(1,13)]
for W,H,K,ph,sk in jobs:
    rebs,END=sched(H,ph);nx=dict(zip(rebs,rebs[1:]+[END]))
    for d in rebs:
        S=E.SES[E.SI[d]-1];r,miss,_=E.rank(S,W,frozenset(),sk)
        if miss: continue
        for _,t in r[:K]:
            if (d,t,nx[d]) not in E.EXA: need.add((t,d,min(nx[d],END)))
print('needed ext windows',len(need))
E.END='9999';E.ext('AAPL','AAPL','2005-02-01','2005-03-01')  # init selector
X=E._sel;bad=collections.defaultdict(list)
for t,d,e in sorted(need):
    n,_,res=X.select(t,d,e)
    if n is None: bad[t].append((d,e))
w=csv.writer(open('needs_2005_missing.csv','w'));w.writerow(['label','symbols','n_windows','first','last'])
for t,v in sorted(bad.items()):
    syms=sorted({X.sym(t,m) for (a,b) in v for m in X.mes_in(a,b)})
    w.writerow([t,' '.join(syms),len(v),min(a for a,b in v),max(b for a,b in v)])
print('labels missing',len(bad),'windows',sum(map(len,bad.values())))
