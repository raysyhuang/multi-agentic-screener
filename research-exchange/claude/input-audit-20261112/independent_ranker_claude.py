# Third independent ranker, written from SPEC.md sections 1-5 only.
import csv,gzip,sys,pickle,collections,os
HERE=os.path.dirname(os.path.abspath(__file__))
P=sys.argv[1]
def rd(f):
    return list(csv.DictReader(gzip.open(P+'/'+f,'rt') if f.endswith('.gz') else open(P+'/'+f)))
sess=[r['session'] for r in rd('trading_sessions.csv')]
ME=[d for i,d in enumerate(sess) if i+1==len(sess) or sess[i+1][:7]!=d[:7]]
mei={d:i for i,d in enumerate(ME)}
elig=collections.defaultdict(list)
for r in rd('eligibility_by_signal.csv.gz'):
    if r['eligible_final']=='True': elig[r['signal_date']].append(r['engine_ticker'])
px={(r['engine_ticker'],r['month_end']):float(r['rank_price']) for r in rd('rank_prices_month_end.csv.gz') if r['rank_price']!=''}
ov=collections.defaultdict(list)
for r in rd('rank_price_overrides.csv'): ov[(r['engine_ticker'],r['month_end'])].append((r['signal_from'],r['signal_to'],float(r['rank_price'])))
ca=collections.defaultdict(list)
for r in rd('corporate_actions_applied.csv'): ca[r['engine_ticker']].append(r)
def price(t,d,S):
    for a,b,p in ov.get((t,d),()):
        if a<=S<=b: return p
    return px.get((t,d))
out={}
for S in sorted(elig):
    for W in range(1,13):
        i=mei[S]-W
        if i<0: continue
        B=ME[i]; rows=[]; bad=[]
        for t in elig[S]:
            p1=price(t,S,S); p0=price(t,B,S); ok=p1 is not None and p0 is not None; F=1.0
            for e in ca.get(t,()):
                k=e['kind']
                if k in('spin','special_div') and B<e['ex_date']<=S:
                    if e['factor_F']=='': ok=False
                    else: F*=float(e['factor_F'])
                elif k=='unresolved_BLOCKER' and B<e['ex_date']<=S: ok=False
                elif k=='legacy_data_error_BLOCKER' and e['blocker_from']<=S<=e['blocker_to']: ok=False
            if not ok: bad.append(t)
            else: rows.append((-(p1/(p0*F)-1),t))
        rows.sort()
        out[(S,W)]=dict(bad=bad,top=[(t,-r) for r,t in rows[:5]],n=len(elig[S]))
pickle.dump(out,open(HERE+'/rank3.pkl','wb'))
print(len(out),'tables;',sum(1 for v in out.values() if v['bad']),'UNKNOWN', sorted({k[0] for k,v in out.items() if v['bad']}))
