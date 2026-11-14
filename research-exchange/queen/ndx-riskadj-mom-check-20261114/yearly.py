from riskadj import *
P,TR=E.qqq()
i0,i1=IDX['2008-01-31'],IDX['2026-08-31']
V=[('N0_12-1',('mom',12,1,0)),('N0.5_12-1',('mom',12,1,0.5)),('N1_12-1',('mom',12,1,1)),('mean6_postrule',('mean6',))]
R={v:run(s,3,i0,i1,0.0015,1) for v,s in V}; R['QQQ_TR']=bench(TR,i0,i1)
def yr(res):
    out={}; prev=None
    for d,v in res['navs']:
        y=d[:4]
        if prev is None: prev=(y,v); base=v; continue
        out.setdefault(y,[None,None]); 
        if out[y][0] is None: out[y][0]=lastv if 'lastv' in dir() else base
    return out
rows=[]
for v,res in R.items():
    navs=res['navs']; by={}
    start_v=navs[0][1]; cur_y=None; y_start=start_v
    for (d0,v0),(d1,v1) in zip(navs,navs[1:]):
        y=d1[:4]
        if y!=cur_y: cur_y=y; ys=v0
        by[y]=v1/ys-1
    sub={}
    for name,(a,b) in {'2008-02_to_2014-12':('2008-01-31','2015-01-01'),'2015-01_to_2026-08':('2015-01-01','2026-09-02')}.items():
        pts=[(d,x) for d,x in navs if a<=d<=b]; import datetime as dt
        yrs=(dt.date.fromisoformat(pts[-1][0])-dt.date.fromisoformat(pts[0][0])).days/365.25
        sub[name]=(pts[-1][1]/pts[0][1])**(1/yrs)-1
    rows.append(dict(variant=v,**{k:round(x,4) for k,x in sub.items()},**{y:round(x,4) for y,x in by.items()}))
with open(OUT+'/results_yearly_A_next_open_015.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
