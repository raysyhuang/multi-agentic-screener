# For every pack month-to-month move that looks like a split factor, find any Massive split (any symbol) in that month with matching factor
import csv,gzip,math,collections,bisect
PK='/workspace/momentum432-inputs-pack-full-20261109/'
rp=collections.defaultdict(dict)
for r in csv.DictReader(gzip.open(PK+'rank_prices_month_end.csv.gz','rt')):
    if r['rank_price']: rp[r['engine_ticker']][r['month_end']]=float(r['rank_price'])
ME=sorted({m for d in rp.values() for m in d})
el=collections.defaultdict(set)
for r in csv.DictReader(gzip.open(PK+'eligibility_by_signal.csv.gz','rt')):
    if r['eligible_final']=='True': el[r['engine_ticker']].add(r['signal_date'])
ca=collections.defaultdict(list)
for r in csv.DictReader(open(PK+'corporate_actions_applied.csv')): ca[r['engine_ticker']].append(r)
sp=[r for r in csv.DictReader(open('massive_splits_all_2013_2026.csv'))]
byd=collections.defaultdict(list)
for s in sp:
    hf=float(s['split_from'])/float(s['split_to'])
    if abs(math.log(hf))>0.05: byd[s['execution_date']].append((s['ticker'],hf,s['adjustment_type']))
dates=sorted(byd)
out=[]
for t,px in rp.items():
    ms=sorted(px)
    for a,b in zip(ms,ms[1:]):
        r=px[b]/px[a]
        if abs(math.log(r))<0.15: continue
        i=ME.index(b)
        used=[s for s in el.get(t,()) if s>=b and ME.index(s)-12<i]
        if not used: continue
        cas=[c['event_id'] for c in ca[t] if a<c['ex_date']<=b]
        lo=bisect.bisect_right(dates,a);hi=bisect.bisect_right(dates,b)
        m=[(d,tk,hf,ty) for d in dates[lo:hi] for tk,hf,ty in byd[d] if abs(math.log(r)-math.log(hf))<0.10]
        out.append(dict(engine_ticker=t,m0=a,m1=b,p0=px[a],p1=px[b],ratio=round(r,4),n_signals=len(used),pack_ca='|'.join(cas),
             massive_split_matches=len(m),match_symbols='|'.join(f'{tk}:{d}:{hf:.4g}:{ty}' for d,tk,hf,ty in m[:40])))
w=csv.DictWriter(open('pattern_scan_moves_15pct.csv','w',newline=''),fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print('moves |ln r|>=0.15 inside usable lookbacks:',len(out))
# candidate: ratio near standard split factors
STD=[1/2,2/3,1/3,4/5,1/4,3/4,1/5,1/10,1/20,1/25,1/1.5,1/3,2,3,1.5]
for x in out:
    near=[f for f in STD if abs(math.log(x['ratio'])-math.log(f))<0.06]
    if near and not x['pack_ca']:
        print(x['engine_ticker'],x['m0'],x['m1'],x['ratio'],x['n_signals'],'matches:',x['massive_split_matches'],x['match_symbols'][:300])
