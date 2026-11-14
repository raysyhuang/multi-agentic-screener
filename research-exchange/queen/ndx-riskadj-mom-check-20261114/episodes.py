from riskadj import *
import datetime as dt
i0,i1=IDX['2008-01-31'],IDX['2026-08-31']
rows=[]
for v,s in (('N1_12-1',('mom',12,1,1)),('N0_12-1',('mom',12,1,0)),('mean6_postrule',('mean6',))):
    res=run(s,3,i0,i1,0.0015,1); dn=res['dnav']; dd=[x[:10] for x in res['ddates']]
    pk=dn[0]; pkd=dd[0]; tr=0; trd=None
    for x,d in zip(dn+[1e18],dd+['open']):
        if x>=pk:
            if tr<-0.25: rows.append(dict(variant=v,peak=pkd,trough=trd,recovered=d,depth=round(tr,4),peak_to_recovery_days=(dt.date.fromisoformat(d)-dt.date.fromisoformat(pkd)).days if d!='open' else None))
            pk=x; pkd=d; tr=0
        elif x/pk-1<tr: tr=x/pk-1; trd=d
import datetime as dt
with open(OUT+'/drawdown_episodes_gt25pct_A_next_open_015.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
