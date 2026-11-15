# sanity: adjusted open vs previous adjusted close for every held name/month in 2015-2026 (flags |gap|>25%)
from sim import *
bad=[];n=0
for i in range(i0,i1):
    for s,t,key in ranking(i)[:K]:
        if key not in PX.k: continue
        dd=days_between(ME[i],ME[i+1]); prevd=ME[i]
        for d in dd:
            if PX.D[key].get(d) is None: continue
            o=adj_open(key,d); pc=PX.D[key].get(prevd)
            if pc and o:
                n+=1; g=o/pc-1
                if abs(g)>0.25: bad.append((t,d,round(g,4)))
            prevd=d
print('checked',n,'open gaps; missing opens',OPEN_MISS[0]); print('|gap|>25%:',bad)
