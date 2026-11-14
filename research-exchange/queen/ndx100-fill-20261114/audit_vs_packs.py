"""Audit: compare Queen's month-to-month returns with the shared S&P packs (rank prices with spin factor F applied) for every NDX member-month both cover.
Flags |diff| > 3 pp."""
import csv,gzip,collections
from engine import *
def pack(P):
    px=collections.defaultdict(dict)
    for r in csv.DictReader(gzip.open(P+'/rank_prices_month_end.csv.gz','rt')):
        if r['rank_price']: px[r['engine_ticker']][r['month_end']]=float(r['rank_price'])
    ca=collections.defaultdict(list); deal={}
    for r in csv.DictReader(open(P+'/corporate_actions_applied.csv')):
        if r['kind'] in ('spin','special_div') and r['factor_F']: ca[r['engine_ticker']].append((r['ex_date'],float(r['factor_F'])))
        if r['kind'].startswith('deal') and r['ex_date']: deal[r['engine_ticker']]=r['ex_date']
    return px,ca,deal
out=[]; ncmp=collections.Counter()
for P in ('repo/research-exchange/shared/inputs-pack-full-20261109','/workspace/w10h2-w12h4-oos-2005-2015-20261112/pack'):
    px,ca,deal=pack(P)
    for i in range(len(ME)-1):
        a,b=ME[i],ME[i+1]
        if a not in MEM: continue
        for t in MEM[a]:
            pt={'BKNG':'BKNG'}.get(t,t)
            if pt not in px or a not in px[pt] or b not in px[pt]: continue
            if pt in deal and b>=deal[pt]: continue
            F=1.0
            for e,x in ca.get(pt,()):
                if a<e<=b: F*=x
            pr=px[pt][b]/(px[pt][a]*F)-1
            q=PX.hold_ret(series_key(t,a),a,b)
            if q is None: continue
            ncmp[os.path.basename(P)]+=1
            if abs(q[0]-pr)>0.03: out.append(dict(pack=os.path.basename(P),ticker=t,start=a,end=b,queen_ret=round(q[0],4),pack_ret=round(pr,4),diff=round(q[0]-pr,4)))
with open('out/audit_queen_vs_packs_monthly_returns.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['pack','ticker','start','end','queen_ret','pack_ret','diff']); w.writeheader(); w.writerows(out)
print('compared member-months:',dict(ncmp),'flags >3pp:',len(out))
for o in out: print(o)
