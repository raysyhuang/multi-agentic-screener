"""Ingest Massive raw (adjusted=false) daily aggs exported as tk,t,c,v (query_data UNION) or v,vw,o,c,h,l,t,n (call_api files) into raw_massive_aggs/SYM.csv (date,close,volume). Date = NY date of bar timestamp."""
import sys,csv,datetime as dt,os,collections
from zoneinfo import ZoneInfo
NY=ZoneInfo('America/New_York');OUT='/workspace/momentum-followup-20261114/raw_massive_aggs/'
def d(ms): return dt.datetime.fromtimestamp(int(float(ms))/1000,NY).date().isoformat()
def write(sym,rows):
    with open(OUT+sym+'.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['date','close','volume'])
        for k in sorted(rows): w.writerow([k,*rows[k]])
mode=sys.argv[1]
if mode=='union':
    R=collections.defaultdict(dict)
    for f in sys.argv[2:]:
        for l in open(f):
            p=l.strip().split(',')
            if len(p)==4 and p[1][:1].isdigit(): R[p[0]][d(p[1])]=(float(p[2]),float(p[3]))
    for s,r in R.items(): write(s,r);print(s,len(r),min(r),max(r))
else:   # map file: sym<TAB>path (call_api raw text)
    for line in open(sys.argv[2]):
        s,f=line.strip().split('\t');r={}
        for l in open(f):
            p=l.strip().split(',')
            if len(p)==8 and p[0][:1].isdigit():
                try: r[d(p[6])]=(float(p[3]),float(p[0]))
                except: pass
        if s in ('ACT','ANTM','ABS','AABA'): continue   # truncated/empty previews; ACT/ANTM/ABS re-exported via union
        if r: write(s,r);print(s,len(r),min(r),max(r))
        else: print(s,'EMPTY')
