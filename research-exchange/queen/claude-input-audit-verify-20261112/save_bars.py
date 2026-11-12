# usage: save_bars.py <assetfile> <name>  -> massive_raw/<name>.csv with date column (America/New_York session date)
import sys,csv,datetime,zoneinfo,io
src,name=sys.argv[1],sys.argv[2]
txt=open(src).read()
lines=[l for l in txt.splitlines() if l and not l.startswith('<') ]
start=[i for i,l in enumerate(lines) if l.startswith('v,') or ',c,' in l and 't' in l.split(',')]
rows=list(csv.DictReader(io.StringIO('\n'.join(lines[start[0]:])))) if start else []
ny=zoneinfo.ZoneInfo('America/New_York')
out=open(f'/workspace/claude-input-audit-verify-20261112/massive_raw/{name}.csv','w',newline='')
w=csv.writer(out);w.writerow(['date','close','open','volume'])
n=0
for r in rows:
    try: t=int(float(r['t']))
    except: continue
    d=datetime.datetime.fromtimestamp(t/1000,ny).date().isoformat()
    w.writerow([d,r['c'],r['o'],r['v']]);n+=1
print(name,n)
