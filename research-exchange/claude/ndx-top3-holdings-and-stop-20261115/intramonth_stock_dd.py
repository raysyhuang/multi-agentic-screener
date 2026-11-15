#!/usr/bin/env python3
"""Claude: for every stock-month held by the Top3 rule (2014-12-31 -> 2026-11-13), the worst point inside the month.
from_entry_low  = lowest intraday low vs the entry price (month-end close);  from_entry_close = lowest daily close vs entry;
peak_to_trough  = largest fall from a high inside the month to a later low (daily closes).  Usage: python3 intramonth_stock_dd.py"""
import os,csv,bisect,statistics as st
H0=os.path.dirname(os.path.abspath(__file__)); exec(open(os.path.join(H0,'stop_rebuy_sim.py')).read().split("if __name__=='__main__':")[0].replace("HERE=os.path.dirname(os.path.abspath(__file__))","HERE=H0"))
i0=IDX['2014-12-31']; n=IDX[ME[-1]]-i0; R=Run(); rows=[]
for m in range(i0,i0+n):
    for t,_ in R.rank(m,'M6')[:3]:
        b=bars(t); d0,d1=ME[m],ME[m+1]
        if d0 not in b: continue
        e=b[d0][3]; ds=[d for d in CAL[bisect.bisect_right(CAL,d0):bisect.bisect_right(CAL,d1)] if d in b]
        if not ds: continue
        lo=min((b[d][2],d) for d in ds); lc=min((b[d][3],d) for d in ds); pk=e; ptt=0
        for d in ds:
            c=b[d][3]; pk=max(pk,c); ptt=min(ptt,c/pk-1)
        rows.append(dict(month=d1[:7],ticker=t,from_entry_low=round(lo[0]/e-1,4),low_date=lo[1],from_entry_close=round(lc[0]/e-1,4),close_low_date=lc[1],peak_to_trough_close=round(ptt,4),month_return=round(b[ds[-1]][3]/e-1,4)))
w=csv.DictWriter(open(os.path.join(H0,'intramonth_stock_dd.csv'),'w',newline=''),fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows),'stock-months')
for k,lab in (('from_entry_low','相对买入价的最低点(盘中最低价)'),('from_entry_close','相对买入价的最低点(收盘价)'),('peak_to_trough_close','月内从高点回落(收盘价)'),('month_return','整月收益')):
    s=sorted(rows,key=lambda r:r[k])[:6]; v=[r[k] for r in rows]
    print(lab,'| 最差:',[(r['month'],r['ticker'],'%.1f%%'%(r[k]*100),'月末 %+.1f%%'%(r['month_return']*100)) for r in s])
    print('   中位数 %.1f%% | 低于-10%%: %d (%.0f%%) | 低于-20%%: %d (%.0f%%) | 低于-30%%: %d'%(st.median(v)*100,sum(x<-.1 for x in v),100*sum(x<-.1 for x in v)/len(v),sum(x<-.2 for x in v),100*sum(x<-.2 for x in v)/len(v),sum(x<-.3 for x in v)))
t10=[r for r in rows if r['from_entry_low']<=-0.10]; print('盘中跌破买入价10%%的 %d 个持仓月: 月末平均 %+.1f%%, 月末收正的 %d 个 (%.0f%%), 月末比-10%%更好的 %d 个'%(len(t10),st.mean(r['month_return'] for r in t10)*100,sum(r['month_return']>0 for r in t10),100*sum(r['month_return']>0 for r in t10)/len(t10),sum(r['month_return']>-0.10 for r in t10)))
