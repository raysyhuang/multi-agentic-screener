import sys,csv,gzip,collections,os,json
sys.path.insert(0,'/workspace/w10h2-w12h4-oos-2005-2015-20261112/build')
import exec_select as ES
E=[r for r in csv.DictReader(gzip.open('pack0515/execution_availability.csv.gz','rt')) if r['status']=='no_verified_series']
labs=collections.defaultdict(lambda:['9999','0000'])
for r in E:
    L='AGN' if r['engine_ticker'] in('AGN',) else r['engine_ticker'];x=labs[L];x[0]=min(x[0],r['trade_date']);x[1]=max(x[1],r['window_end'])
have=set(f[:-4] for f in os.listdir('/workspace/w10h2-w12h4-oos-2005-2015-20261112/raw/massive_aggs'))
out=[]
for L,(a,b) in sorted(labs.items()):
    syms=sorted({ES.sym(L,m) for m in ES.ME if a[:7]<=m[:7]<=b[:7]}|{L})
    for X in syms: out.append((L,X,a,b,X in have))
json.dump(out,open('scratch/need_symbols.json','w'))
print(len(labs),len(out),sum(1 for o in out if not o[4]));print(out[:30])
