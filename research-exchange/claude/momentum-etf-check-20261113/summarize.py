#!/usr/bin/env python3
"""Claude: momentum ETF check. Reads the two FMP snapshots in this folder (fetched 2026-11-13, Asia/Taipei) and prints the tables in README.md. Stdlib only."""
import csv,os,datetime as dt
H=os.path.dirname(os.path.abspath(__file__))
q=list(csv.DictReader(open(os.path.join(H,'fmp_quote_change_20261113.csv'))))
spy=[r for r in q if r['symbol']=='SPY'][0]
short={'FDMO','JMOM','VFMO'}
print('symbol   3M     1Y     5Y    10Y   10Y annualized')
for r in sorted(q,key=lambda r:-float(r['10Y']) if r['symbol'] not in short else 1e9):
    ten='under 10y' if r['symbol'] in short else '%+.0f%%  %.1f%%/yr'%(float(r['10Y']),((1+float(r['10Y'])/100)**0.1-1)*100)
    print('%-5s %+6.1f%% %+6.1f%% %+6.0f%%  %s'%(r['symbol'],float(r['3M']),float(r['1Y']),float(r['5Y']),ten))
for k in ('3M','1Y','3Y','5Y','10Y'):
    pool=[r for r in q if r['symbol']!='SPY' and not (k=='10Y' and r['symbol'] in short)]
    print('beat SPY over %s: %d of %d'%(k,sum(float(r[k])>float(spy[k]) for r in pool),len(pool)),[r['symbol'] for r in pool if float(r[k])>float(spy[k])])
a={(r['symbol'],r['date']):float(r['adjClose']) for r in csv.DictReader(open(os.path.join(H,'fmp_adjclose_spots_20261113.csv')))}
y=(dt.date(2025,1,2)-dt.date(2015,1,2)).days/365.25
print('2015-01-02 -> 2025-01-02, dividends included:',{s:'%.1f%%/yr'%(((a[(s,'2025-01-02')]/a[(s,'2015-01-02')])**(1/y)-1)*100) for s in ('XMMO','MTUM','SPY','PDP','DWAS')})
print('2008:',{s:'%.1f%%'%((a[(s,'2008-12-31')]/a[(s,'2007-12-31')]-1)*100) for s in ('PDP','SPY')})
print('2022:',{s:'%.1f%%'%((a[(s,'2022-12-30')]/a[(s,'2021-12-31')]-1)*100) for s in ('MTUM','SPMO','SPY')})
