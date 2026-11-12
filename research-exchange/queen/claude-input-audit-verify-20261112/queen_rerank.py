# Queen independent re-ranker (own code, written from SPEC.md sections 1-5). Research only.
import csv,gzip,collections,sys,json
PK='/workspace/momentum432-inputs-pack-full-20261109/'
sess=[r['session'] for r in csv.DictReader(open(PK+'trading_sessions.csv'))]
ME=[];
for a,b in zip(sess,sess[1:]+['9999-99-99']):
    if a[:7]!=b[:7]: ME.append(a)
base={}
for r in csv.DictReader(gzip.open(PK+'rank_prices_month_end.csv.gz','rt')):
    if r['rank_price']!='': base[(r['engine_ticker'],r['month_end'])]=float(r['rank_price'])
ov=collections.defaultdict(list)
for r in csv.DictReader(open(PK+'rank_price_overrides.csv')):
    ov[(r['engine_ticker'],r['month_end'])].append((r['signal_from'],r['signal_to'],float(r['rank_price'])))
CA=collections.defaultdict(list)
for r in csv.DictReader(open(PK+'corporate_actions_applied.csv')): CA[r['engine_ticker']].append(r)
EL=collections.defaultdict(set)
for r in csv.DictReader(gzip.open(PK+'eligibility_by_signal.csv.gz','rt')):
    if r['eligible_final']=='True': EL[r['signal_date']].add(r['engine_ticker'])
SIG=sorted(EL)

def build(corr):
    """corr: dict with 'scale': list of (ticker, last_month_end_inclusive, factor) multiplying base prices on or before that date;
       'replace_series': {new_ticker:(src_ticker,last_me,divisor)} ; 'elig_remove': {(S,t)} ; 'elig_add': {(S,t)}"""
    P=dict(base)
    for t,last,f in corr.get('scale',[]):
        for (tt,m),v in list(P.items()):
            if tt==t and m<=last: P[(tt,m)]=v*f
    for t,(src,last,div) in corr.get('replace_series',{}).items():
        for (tt,m),v in list(base.items()):
            if tt==src and m<=last: P[(t,m)]=v/div
    E={S:set(u) for S,u in EL.items()}
    for S,t in corr.get('elig_remove',()): E[S].discard(t)
    for S,t in corr.get('elig_add',()): E[S].add(t)
    return P,E

def price(P,t,d,S):
    for a,b,v in ov.get((t,d),()):
        if a<=S<=b: return v
    return P.get((t,d))

def rank(P,E,S,W):
    i=ME.index(S);B=ME[i-W]
    rows=[];bad=[]
    for t in sorted(E[S]):
        p1=price(P,t,S,S);p0=price(P,t,B,S)
        if p1 is None or p0 is None: bad.append((t,'noprice'));continue
        F=1.0;ok=True
        for c in CA.get(t,()):
            k=c['kind']
            if k in('spin','special_div') and B<c['ex_date']<=S:
                if c['factor_F']=='': ok=False
                else: F*=float(c['factor_F'])
            elif k=='unresolved_BLOCKER' and B<c['ex_date']<=S: ok=False
            elif k=='legacy_data_error_BLOCKER' and c['blocker_from']<=S<=c['blocker_to']: ok=False
        if not ok: bad.append((t,'ca'));continue
        rows.append((-(p1/(p0*F)-1),t))
    if bad: return None,bad
    rows.sort()
    return [(t,-r) for r,t in rows],[]

def all_tables(corr):
    P,E=build(corr);out={}
    for S in SIG:
        for W in range(1,13):
            if ME.index(S)-W<0: continue
            out[(S,W)]=rank(P,E,S,W)
    return out
