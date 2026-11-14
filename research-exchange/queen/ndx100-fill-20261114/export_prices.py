"""Export month-end prices in Claude's rank_prices_month_end format (engine_ticker,month_end,rank_price,price_source),
plus the 49 requested symbols' daily raw + split-adjusted closes, plus a coverage-by-month table."""
import json,gzip,datetime as dt
from engine import *
from claude_mask import build_mask
from zoneinfo import ZoneInfo
ET=ZoneInfo('America/New_York')
os.makedirs('out/prices',exist_ok=True)
win=json.load(open('member_windows_calendar.json'))
REQ=[r['ticker'] for r in csv.DictReader(open('repo/research-exchange/claude/ndx100-partial-test-20261114/missing_prices_request.csv'))]
def mi(d): return IDX.get(d)
rows=[]; missing=[]
for t,(a,b) in sorted(win.items()):
    ia=max(0,[i for i,d in enumerate(ME) if d>=a[:7]][0]-12); ib=min(len(ME)-1,[i for i,d in enumerate(ME) if d[:7]<=b[:7]][-1]+1)
    for i in range(ia,ib+1):
        d=ME[i]; k=series_key(t,d); p=PX.at(k,d)
        member = d in MEM and t in MEM[d]
        if p:
            src=('massive_aggs_daily_splitadj' if k in PX.k else 'massive_aggs_monthly_splitadj')+':'+k+('' if p[1]==d else '@'+p[1])
            rows.append(dict(engine_ticker=t,month_end=d,rank_price=p[0],price_source=src))
        elif member: missing.append(dict(ticker=t,month_end=d,series=k))
with open('out/prices/queen_rank_prices_month_end_all_members.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['engine_ticker','month_end','rank_price','price_source']); w.writeheader(); w.writerows(rows)
with open('out/prices/queen_rank_prices_month_end_49_requested.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['engine_ticker','month_end','rank_price','price_source']); w.writeheader(); w.writerows([r for r in rows if r['engine_ticker'] in REQ])
with open('out/prices/missing_member_months.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['ticker','month_end','series']); w.writeheader(); w.writerows(missing)
print('month-end rows',len(rows),'missing member-months (all 2006-12..2026-10):',len(missing))
from collections import Counter
print(Counter(m['ticker'] for m in missing))
# daily raw + adj for 49
def ld(p):
    out={}
    for r in csv.DictReader(open(p)):
        out[dt.datetime.fromtimestamp(int(r['t'])/1000,ET).date().isoformat()]=float(r['c'])
    return out
n=0
with open('out/prices/queen_daily_49_requested_raw_and_splitadj.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['ticker','date','close_unadjusted','close_split_adjusted','source'])
    for t in REQ:
        adj=load_daily('raw/adj/%s.csv'%t) if os.path.exists('raw/adj/%s.csv'%t) else {}
        raw=ld('raw/raw/%s.csv'%t) if os.path.exists('raw/raw/%s.csv'%t) else {}
        if not adj and not raw: continue
        a,b=win[t]; lo=ME[max(0,[i for i,d in enumerate(ME) if d>=a[:7]][0]-12)]; hi=b[:7]+'-99'
        hi=ME[min(len(ME)-1,[i for i,d in enumerate(ME) if d[:7]<=b[:7]][-1]+1)]
        for d in sorted(set(adj)|set(raw)):
            if lo<=d<=hi: w.writerow([t,d,raw.get(d,''),adj.get(d,''),'massive /v2/aggs day adjusted=false / adjusted=true (raw vendor file, before Queen junction fixes)']); n+=1
print('daily rows',n)
# coverage by month 2007-2024 (signal months 2006-12..2024-11 holding into 2024-12)
mask,_,_=build_mask()
cov=[]
for d in ME:
    if d not in MEM or d<'2006-12' or d>'2024-12-31': continue
    mem=MEM[d]; q=[t for t in mem if PX.at(series_key(t,d),d)]
    c=mask.get(d)
    cov.append(dict(month_end=d,n_members=len(mem),queen_priced=len(q),queen_missing=' '.join(sorted(set(mem)-set(q))),
                    claude_pack_priced=len(c) if c is not None else '', claude_pack_missing=' '.join(sorted(set(mem)-c)) if c is not None else 'pack starts 2013-11',
                    claude_missing_but_queen_has=' '.join(sorted((set(mem)-c)&set(q))) if c is not None else ''))
with open('out/coverage_by_month.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(cov[0])); w.writeheader(); w.writerows(cov)
import statistics as st
for lo,hi in (('2006-12','2014-11-30'),('2014-12','2024-11-30')):
    s=[c for c in cov if lo<=c['month_end']<=hi]
    print(lo,hi,'queen cov %.2f%%'%(100*sum(c['queen_priced'] for c in s)/sum(c['n_members'] for c in s)),
          'claude cov %s'%('%.2f%%'%(100*sum(c['claude_pack_priced'] for c in s if c['claude_pack_priced']!='')/sum(c['n_members'] for c in s if c['claude_pack_priced']!='')) if any(c['claude_pack_priced']!='' for c in s) else 'n/a'))
