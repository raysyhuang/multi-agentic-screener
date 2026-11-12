import pandas as pd,numpy as np
L=pd.read_csv('period_ledger_lo.csv.gz',dtype={'flag':str,'picks':str},low_memory=False)
W_={'5y':('2021-01-01','2025-12-31'),'10y':('2016-01-01','2025-12-31'),'full':('2005-01-01','2025-12-31')}
out=[]
cells=[(u,W,H) for u in ['TW50','TW150','MC100'] for (W,H) in [(12,12),(6,6)]]+[(u,W,H) for u in ['TW150','MC100'] for W in (7,8,9) for H in (3,4,5,6)]
for u,W,H in cells:
  for w,(a,b) in W_.items():
    q=L[(L.univ==u)&(L.W==W)&(L.H==H)&(L.M==12)&(~L.censored)&(L.end>=a)&(L.end<=b)].copy()
    q['rel']=np.log1p(q.tr_net)-np.log1p(q.b0050_tr)
    q=q.sort_values('rel',ascending=False)
    ex=q.rel.sum(); ex2=q.rel.iloc[2:].sum()
    out.append(dict(univ=u,W=W,H=H,window=w,n=len(q),log_excess=round(ex,2),log_excess_ex_top2=round(ex2,2),top2=' | '.join(f"{r.signal}:{r.picks} {r.tr_net*100:+.0f}% vs {r.b0050_tr*100:+.0f}%" for _,r in q.head(2).iterrows())))
O=pd.DataFrame(out);O.to_csv('concentration_check_lo.csv',index=False)
pd.set_option('display.width',300);pd.set_option('display.max_colwidth',120);print(O.to_string(index=False))
