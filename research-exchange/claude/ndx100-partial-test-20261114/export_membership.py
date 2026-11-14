#!/usr/bin/env python3
"""Export month-end Nasdaq-100 membership from github.com/jmccarrell/n100tickers (covers 2007 onward).
Usage: python3 export_membership.py <path to n100tickers/src>  -> ndx100_membership_month_end.csv"""
import sys,csv,gzip,os
sys.path.insert(0,sys.argv[1])
from nasdaq_100_ticker_history import tickers_as_of
H=os.path.dirname(os.path.abspath(__file__)); P=os.path.join(H,'..','..','shared','inputs-pack-full-20261109')
ME=sorted({r['month_end'] for r in csv.DictReader(gzip.open(os.path.join(P,'rank_prices_month_end.csv.gz'),'rt'))})
w=csv.writer(open(os.path.join(H,'ndx100_membership_month_end.csv'),'w',newline='')); w.writerow(['month_end','n','tickers'])
for d in ME:
    y,m,dd=map(int,d.split('-')); t=sorted(tickers_as_of(y,m,dd)); w.writerow([d,len(t),' '.join(t)])
