#!/usr/bin/env python3
"""Month-end Nasdaq-100 membership for every month-end in Queen's price file, from jmccarrell/n100tickers @16eaf93 (my own clone).
Dates before 2007-01-01 use the 2007-01-01 list (the package starts in 2007).  Usage: python3 export_membership_full.py <n100tickers/src>"""
import sys,csv,os
sys.path.insert(0,sys.argv[1]); from nasdaq_100_ticker_history import tickers_as_of
H=os.path.dirname(os.path.abspath(__file__)); Q=os.path.join(H,'..','..','queen','ndx100-fill-20261114','out','prices','queen_rank_prices_month_end_all_members.csv')
ME=sorted({r['month_end'] for r in csv.DictReader(open(Q))})
w=csv.writer(open(os.path.join(H,'ndx100_membership_month_end_2006_2026.csv'),'w',newline='')); w.writerow(['month_end','n','tickers'])
for d in ME:
    y,m,dd=map(int,d.split('-'))
    if d<'2007-01-01': y,m,dd=2007,1,1
    t=sorted(tickers_as_of(y,m,dd)); w.writerow([d,len(t),' '.join(t)])
