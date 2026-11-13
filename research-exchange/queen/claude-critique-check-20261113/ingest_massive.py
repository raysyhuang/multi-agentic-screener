"""Parse Massive /v2/aggs (adjusted=false) call_api dumps listed in fetched_map.txt into massive_aggs_ext/SYM.csv.
Resolves the tool's 'Full text:' pointer files; rows = 8 fields v,vw,o,c,h,l,t,n; date = t in America/New_York."""
import re,os,datetime as dt,csv,json
from zoneinfo import ZoneInfo
NY=ZoneInfo('America/New_York');os.makedirs('massive_aggs_ext',exist_ok=True)
def text(p,depth=0):
    s=open(p).read()
    m=re.search(r'Full text: (\S+\.txt)',s)
    if m and depth<3: return text(m.group(1),depth+1)
    return s
def rows(s):
    s=s.split('Structured content')[0]
    out={}
    for l in s.split('\n'):
        p=l.strip().split(',')
        if len(p)==8 and re.match(r'^[0-9.]+$',p[0]) and p[6].isdigit():
            d=dt.datetime.fromtimestamp(int(p[6])/1000,NY).date().isoformat();out[d]=(float(p[3]),float(p[0]))
    return out
rep=[]
for l in open('fetched_map.txt'):
    if not l.strip(): continue
    sym,p=l.split()
    s=text(p);r=rows(s)
    # sanity: no omission markers left inside the CSV part
    omitted='omitted' in s.split('Structured content')[0]
    with open(f'massive_aggs_ext/{sym}.csv','w') as f:
        w=csv.writer(f);w.writerow(['date','close','volume'])
        for d in sorted(r): w.writerow([d,*r[d]])
    rep.append((sym,len(r),min(r) if r else '',max(r) if r else '',omitted))
for x in rep: print(*x)
