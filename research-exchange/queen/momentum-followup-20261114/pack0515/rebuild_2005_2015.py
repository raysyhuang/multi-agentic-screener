"""[2005-2015 cross-period copy: identical logic to frozen rebuild_full.py; only the window years changed 2015->2005, 2025->2015] Stdlib-only reference rebuild of ANY 432-grid path (W1-12, H1-12, K1-3, phase 1-12) from this folder's CSVs alone.
No engine code, no network. See SPEC.md. Usage:  python3 rebuild_full.py W H K phase"""
import csv,gzip,sys,collections,datetime as dt
c=0.0025
def rd(f):
    return list(csv.DictReader(gzip.open(f,'rt') if f.endswith('.gz') else open(f)))
SES=[r['session'] for r in rd('trading_sessions.csv')];SI={d:i for i,d in enumerate(SES)}
ME=sorted({max(s for s in SES if s[:7]==m) for m in {s[:7] for s in SES}})
ELIG=collections.defaultdict(list)
for r in rd('eligibility_by_signal.csv.gz'):
    if r['eligible_final']=='True': ELIG[r['signal_date']].append(r['engine_ticker'])
PX={(r['engine_ticker'],r['month_end']):float(r['rank_price']) for r in rd('rank_prices_month_end.csv.gz')}
OV=collections.defaultdict(list)
for r in rd('rank_price_overrides.csv'): OV[(r['engine_ticker'],r['month_end'])].append((r['signal_from'],r['signal_to'],float(r['rank_price'])))
def px(t,d,S):
    for a,b,v in OV.get((t,d),[]):
        if a<=S<=b: return v
    return PX.get((t,d))
CA=collections.defaultdict(list);GRADE={}
for r in rd('corporate_actions_applied.csv'): CA[r['engine_ticker']].append(r);GRADE[r['event_id']]=r.get('evidence_grade','')
RC={}
def rank(S,W):
    """returns (ranked [(ret,ticker)], unrankable [tickers], rank-side event ids applied)"""
    if (S,W) in RC: return RC[(S,W)]
    B=ME[ME.index(S)-W];r=[];miss=[];ids=set()
    for t in sorted(ELIG[S]):
        ev=CA[t]
        if any(e['kind']=='legacy_data_error_BLOCKER' and e['blocker_from']<=S<=e['blocker_to'] for e in ev): miss.append(t);continue
        if any(e['kind']=='unresolved_BLOCKER' and B<e['ex_date']<=S for e in ev): miss.append(t);continue
        a,b=px(t,B,S),px(t,S,S)
        if a and b:
            for e in ev:
                if e['kind'] in('spin','special_div') and B<e['ex_date']<=S:
                    if not e['factor_F']: a=None;break
                    a*=float(e['factor_F']);ids.add(e['event_id'])
        if a and b and a>0: r.append((b/a-1,t))
        else: miss.append(t)
    r.sort(key=lambda x:(-x[0],x[1]));RC[(S,W)]=(r,miss,ids);return RC[(S,W)]
# ---- execution inputs
EXA={(r['trade_date'],r['engine_ticker'],r['window_end']):r for r in rd('execution_availability.csv.gz')}
CL={(r['price_series'],r['execution_key'],r['date']):float(r['close']) for r in rd('execution_daily_closes.csv.gz')}
SLAST={(r['price_series'],r['execution_key']):r['series_last_date'] for r in rd('series_info.csv')}
HEV=collections.defaultdict(list)
for r in rd('holding_events.csv'): HEV[(r['price_series'],r['execution_key'])].append(r)
DEAL=collections.defaultdict(list)
for r in rd('untradable_deal_dates.csv'): DEAL[r['engine_ticker']].append(r['deal_date'])
def ek(t,d):   # economic execution key: two issuers reused AGN
    if t=='AGN': return 'AGN_LEGACY' if d<'2015-03-17' else 'AGN_PLC'
    return 'AGN_PLC' if t=='ACT' else t
def first_td(y,m): k=f'{y:04d}-{m:02d}';return next((s for s in SES if s[:7]==k),'9999-12-31')   # beyond the calendar = after END
def path(W,H,K,ph,want_nav=False):
    D0=first_td(2005,ph);END=first_td(2015,ph);rebs=[];j=0
    while True:
        mm=(ph-1)+j*H;d=first_td(2005+mm//12,mm%12+1)
        if d>=END: break
        rebs.append(d);j+=1
    nxt=dict(zip(rebs,rebs[1:]+[END]));hold={};cash=1.0;eq=[];dprev=D0;ids=set();log=[];ntr=0
    def fail(d,why,t=''): return dict(status='UNKNOWN',fail_reason=why,fail_ticker=t,fail_date=d,rebalances=log,n_trades=ntr)
    for d in SES[SI[D0]:SI[END]+1]:
        for t,(n,ser,last) in list(hold.items()):
            evs=[e for e in HEV[(ser,t)] if dprev<e['ex_date']<=d]
            for e in evs:
                if e['type']=='unresolved': return fail(None,'corporate_action_unresolved',e['event_id'])
            for e in evs:
                if e['action'] in('skip','ignore'): continue
                cs=float(e['cash_per_share'] or 0)
                if e['type'] in('spin','special_div'):
                    g=n*float(e['ratio'])*float(e['child_first_close'] or 0);cash+=n*cs+g-c*g;ids.add(e['event_id'])
                elif e['type']=='cash': cash+=n*cs;del hold[t];ids.add(e['event_id'])
                elif e['type']=='stock':
                    acq=e['child_or_acquirer'];aser=e['acquirer_series']
                    if e['action']=='fail' or (aser,acq,d) not in CL: return fail(d,'corporate_action_unresolved',t+'->'+acq+'(series unverified)')
                    ev=0.0;eqv=0.0
                    if acq in hold and acq!=t:
                        ne,se,_=hold[acq]
                        if (se,acq,d) not in CL: return fail(d,'daily_gap',acq+'(existing merger recipient)')
                        ev=ne*CL[(se,acq,d)];eqv=ev/CL[(aser,acq,d)]
                    cash+=n*cs;del hold[t];hold[acq]=(eqv+n*float(e['ratio']),aser,CL[(aser,acq,d)]);ids.add(e['event_id'])
        val=cash
        for t,(n,ser,last) in list(hold.items()):
            if (ser,t,d) in CL: p=CL[(ser,t,d)];hold[t]=(n,ser,p);val+=n*p
            elif d>SLAST.get((ser,t),'9999'): return fail(d,'corporate_action_unresolved',t)
            else: return fail(d,'daily_gap',t)
        if d in nxt or d==D0:
            S=SES[SI[d]-1];r,miss,rids=rank(S,W)
            if miss: return fail(d,'rank_missing_unbounded',' '.join(miss))
            sel=[t for _,t in r[:K]];keys=[ek(t,d) for t in sel]
            if len(set(keys))!=len(keys): return fail(d,'duplicate_economic_execution_key','|'.join(sel))
            sers={}
            for t,k in zip(sel,keys):
                if k in hold: sers[k]=hold[k][1];continue
                a=EXA.get((d,t,nxt[d]))
                if a is None: return fail(d,'NOT_IN_PACK',t)
                if a['status']=='not_tradable': x,y=a['detail'].split('@');return fail(y,'pick_not_tradable',x)
                if a['status']!='ok': return fail(d,'exec_price_missing',t)
                sers[k]=a['price_series']
            E0=val;h={t:v[0]*v[2] for t,v in hold.items()};Ep=E0
            for _ in range(200):
                T={t:Ep/K for t in keys};turn=sum(abs(T.get(t,0)-h.get(t,0)) for t in set(T)|set(h));En=E0-c*turn
                if abs(En-Ep)<=1e-12*E0: Ep=En;break
                Ep=En
            new={k:(T[k]/CL[(sers[k],k,d)],sers[k],CL[(sers[k],k,d)]) for k in keys}
            for k in set(hold)|set(new):
                n0=hold[k][0] if k in hold else 0.0;n1=new[k][0] if k in new else 0.0
                ntr+=not(k in hold and k in new and abs(n1-n0)<=1e-12*max(n0,1e-30))
            hold=new;cash=Ep-sum(T.values());val=Ep;ids|=rids;log.append((S,d,'|'.join(sel)))
        if d==END: g=sum(n*l for n,_,l in hold.values());val=cash+g-c*g;ntr+=len(hold)
        eq.append(val);dprev=d
    if K==1:   # engine R2 post-check (see SPEC: effective for K=1 only)
        for S,d,p in log:
            if any(S<e<=d for e in DEAL[p]): return fail(d,'pick_not_tradable',p)
    yrs=(dt.date.fromisoformat(END)-dt.date.fromisoformat(D0)).days/365.25;pk=-1;mdd=0.0
    for v in eq: pk=max(pk,v);mdd=min(mdd,v/pk-1)
    st='DATA_COMPLETE_PROVISIONAL_C' if any(GRADE.get(i)=='C' for i in ids) else 'DATA_COMPLETE_CONDITIONAL'
    return dict(status=st,final=eq[-1],CAGR=eq[-1]**(1/yrs)-1,MDD=mdd,rebalances=log,n_trades=ntr,entry=D0,end=END,nav=eq if want_nav else None)
if __name__=='__main__':
    W,H,K,ph=map(int,sys.argv[1:5]);o=path(W,H,K,ph);o.pop('nav',None);o['rebalances']=len(o['rebalances']);print(o)
