"""Queen v18 engine, extended copy for the Claude-critique check (2026-11-13, Asia/Taipei).
Logic = frozen momentum432-inputs-pack-full-20261109/rebuild_full.py (byte-identical rules), plus OPTIONAL switches:
  excl      : set of engine tickers removed from the eligible universe before ranking (next-ranked name is picked)
  cash_on   : set of tickers that, when selected, are replaced by cash for that holding period (K slot left in cash)
  skip      : True -> 'W-1' skip-month lookback: return from ME[S-W] to ME[S-1] (most recent month excluded)
  regime    : dict signal_month_end -> bool(risk_off); risk_off -> whole book in cash for that holding period
Execution layer: pack rows first (unchanged). If a pick/window is NOT in the pack's execution_availability, an
extension series is used (flagged 'ext'):
  2015-2025: the pack's own per-ticker daily series (same series the pack uses for that ticker in other windows), with
             the pack's holding_events and the pack's deal-untradable rule (deal in (d-60d,d]); else UNKNOWN.
  2005-2015: the 2005-2015 packet's own exec_select.select() (FMP / Massive daily, month-end ratio to rank price <=2% [EXT_TOL; FMP 2-dp rounding of split-adjusted old prices],
             no missing session); rank-side spins/special divs inside the hold are applied as shares /= F (flag);
             else UNKNOWN (NO_VALID_DAILY_SERIES).
Reads pack folders read-only. Nothing is written by this module."""
import csv,gzip,sys,collections,datetime as dt,os,json
c=0.0025
EXT_TOL=0.02   # extension-series month-end ratio tolerance (frozen v17 daily_layer uses 2%; 2005-2015 packet used 0.5%)
APPROX_DELIST=False  # True -> flagged approximation: a pick whose daily series stops inside the hold is cashed out at its last close
def rd(f):
    return list(csv.DictReader(gzip.open(f,'rt') if f.endswith('.gz') else open(f)))
class Eng:
    def __init__(s,pack,y0):
        s.pack=pack;s.y0=y0;j=lambda f:os.path.join(pack,f)
        s.SES=[r['session'] for r in rd(j('trading_sessions.csv'))];s.SI={d:i for i,d in enumerate(s.SES)}
        s.ME=sorted({max(x for x in s.SES if x[:7]==m) for m in {x[:7] for x in s.SES}});s.MEI={m:i for i,m in enumerate(s.ME)}
        s.ELIG=collections.defaultdict(list)
        for r in rd(j('eligibility_by_signal.csv.gz')):
            if r['eligible_final']=='True': s.ELIG[r['signal_date']].append(r['engine_ticker'])
        s.PX={(r['engine_ticker'],r['month_end']):float(r['rank_price']) for r in rd(j('rank_prices_month_end.csv.gz'))}
        s.OV=collections.defaultdict(list)
        for r in rd(j('rank_price_overrides.csv')): s.OV[(r['engine_ticker'],r['month_end'])].append((r['signal_from'],r['signal_to'],float(r['rank_price'])))
        s.CA=collections.defaultdict(list);s.GRADE={}
        for r in rd(j('corporate_actions_applied.csv')): s.CA[r['engine_ticker']].append(r);s.GRADE[r['event_id']]=r.get('evidence_grade','')
        s.RC={}
        s.EXA={(r['trade_date'],r['engine_ticker'],r['window_end']):r for r in rd(j('execution_availability.csv.gz'))}
        s.CL={(r['price_series'],r['execution_key'],r['date']):float(r['close']) for r in rd(j('execution_daily_closes.csv.gz'))}
        s.SLAST={(r['price_series'],r['execution_key']):r['series_last_date'] for r in rd(j('series_info.csv'))}
        s.SFIRST={(r['price_series'],r['execution_key']):r['series_first_date'] for r in rd(j('series_info.csv'))}
        s.HEV=collections.defaultdict(list)
        for r in rd(j('holding_events.csv')): s.HEV[(r['price_series'],r['execution_key'])].append(r)
        s.DEAL=collections.defaultdict(list)
        for r in rd(j('untradable_deal_dates.csv')): s.DEAL[r['engine_ticker']].append(r['deal_date'])
        # most-used pack series per ticker (for 2015-2025 extension)
        cnt=collections.defaultdict(collections.Counter)
        for (td,t,we),r in s.EXA.items():
            if r['status']=='ok': cnt[t][r['price_series']]+=1
        s.TSER={t:c_.most_common(1)[0][0] for t,c_ in cnt.items()}
        s.EXT=set();s.EXTSPIN=set();s._sel=None;s.DELIST={}
        s.HEVK=collections.defaultdict(list);_seen=set()
        for (ps,k_),evs in s.HEV.items():
            for e in evs:
                if e['event_id'] not in _seen: _seen.add(e['event_id']);s.HEVK[k_].append(e)
    def px(s,t,d,S):
        for a,b,v in s.OV.get((t,d),[]):
            if a<=S<=b: return v
        return s.PX.get((t,d))
    def rank(s,S,W,excl=frozenset(),skip=False):
        key=(S,W,excl,skip)
        if key in s.RC: return s.RC[key]
        i=s.MEI[S];B=s.ME[i-W];E=s.ME[i-1] if skip else S;r=[];miss=[];ids=set()
        for t in sorted(s.ELIG[S]):
            if t in excl: continue
            ev=s.CA[t]
            if any(e['kind']=='legacy_data_error_BLOCKER' and e['blocker_from']<=S<=e['blocker_to'] for e in ev): miss.append(t);continue
            if any(e['kind']=='unresolved_BLOCKER' and B<e['ex_date']<=S for e in ev): miss.append(t);continue
            a,b=s.px(t,B,S),s.px(t,E,S)
            if a and b:
                for e in ev:
                    if e['kind'] in('spin','special_div') and B<e['ex_date']<=E:
                        if not e['factor_F']: a=None;break
                        a*=float(e['factor_F']);ids.add(e['event_id'])
            if a and b and a>0: r.append((b/a-1,t))
            else: miss.append(t)
        r.sort(key=lambda x:(-x[0],x[1]));s.RC[key]=(r,miss,ids);return s.RC[key]
    def ek(s,t,d):
        if t=='AGN': return 'AGN_LEGACY' if d<'2015-03-17' else 'AGN_PLC'
        return 'AGN_PLC' if t=='ACT' else t
    def first_td(s,y,m): k=f'{y:04d}-{m:02d}';return next((x for x in s.SES if x[:7]==k),'9999-12-31')
    # ---------- extension execution layer
    def ext(s,t,k,d,we):
        """returns (status, series, detail)"""
        if any((dt.date.fromisoformat(d)-dt.timedelta(days=60)).isoformat()<x<=d for x in s.DEAL.get(t,[])): return 'not_tradable',None,f'{t}@{d}'
        if s.y0==2015:
            ser=s.TSER.get(t)
            if ser is None:   # any pack series with this execution key (non-acquirer first)
                c_=[p for (p,kk) in s.SLAST if kk==k];c_.sort(key=lambda p:p.startswith('acq:'))
                ser=c_[0] if c_ else None
            if ser is None or (ser,k,d) not in s.CL: return 'missing',None,'NO_PACK_SERIES'
            s.EXT.add((t,d,we,ser));return 'ok',ser,''
        else:
            if s._sel is None:
                sys.path.insert(0,'/workspace/w10h2-w12h4-oos-2005-2015-20261112/build');import exec_select as X;s._sel=X
                _c=X.check
                def check2(L,ser,a,b,_c=_c,X=X):
                    ok,sp,miss=_c(L,ser,a,b)
                    return (sp is not None and sp<=EXT_TOL and not miss), sp, miss
                X.check=check2
                _ag=X.aggs
                def aggs2(sym,_ag=_ag):
                    A=dict(_ag(sym));f='/workspace/claude-critique-check-20261113/massive_aggs_ext/'+sym+'.csv'
                    if os.path.exists(f):
                        for r in csv.DictReader(open(f)): A.setdefault(r['date'],float(r['close']))
                    return A
                X.aggs=aggs2
            END_=min(we,s.END)
            name,ser,res=s._sel.select(t,d,END_);delist=None
            if name is None and APPROX_DELIST:
                # approximation (flagged): the series stops trading inside the window (deal / delisting). Validate the
                # traded part [d, last] and cash out at the last close (fee c) on the next session.
                X=s._sel;lasts=set()
                for sym_ in dict.fromkeys([X.sym(t,m) for m in X.mes_in(d,END_)]+[t]):
                    for src in (X.aggs(sym_),X.fmp(sym_)):
                        ds=[x for x in src if d<=x<END_]
                        if ds: lasts.add(max(ds))
                for last in sorted(lasts,reverse=True):
                    if last<=d: continue
                    name,ser,res=X.select(t,d,last)
                    if name is not None: END_=last;delist=last;break
            if name is None: return 'missing',None,'NO_VALID_DAILY_SERIES'
            sk='ext:'+name
            for dd,v in ser.items():
                if d<=dd<=END_: s.CL[(sk,k,dd)]=v
            s.SLAST[(sk,k)]=max([dd for dd in ser if dd<=END_]+[s.SLAST.get((sk,k),'0000')])
            if delist: s.DELIST[(sk,k)]=delist
            s.EXT.add((t,d,we,sk));return 'ok',sk,''
    def path(s,W,H,K,ph,want_nav=False,excl=frozenset(),cash_on=frozenset(),skip=False,regime=None):
        y0=s.y0;D0=s.first_td(y0,ph);END=s.first_td(y0+10,ph);s.END=END;rebs=[];j=0;SES=s.SES;SI=s.SI;CL=s.CL
        while True:
            mm=(ph-1)+j*H;d=s.first_td(y0+mm//12,mm%12+1)
            if d>=END: break
            rebs.append(d);j+=1
        nxt=dict(zip(rebs,rebs[1:]+[END]));hold={};cash=1.0;eq=[];dts=[];dprev=D0;ids=set();log=[];ntr=0;flags=set()
        def fail(d,why,t=''): return dict(status='UNKNOWN',fail_reason=why,fail_ticker=t,fail_date=d,rebalances=log,n_trades=ntr,flags=flags)
        for d in SES[SI[D0]:SI[END]+1]:
            for t,(n,ser,last) in list(hold.items()):
                if ser.startswith('ext:') and s.HEVK.get(t):
                    # the pack documents a holding-side deal/spin for this execution key: apply it to the extension series too
                    evs=[e for e in s.HEVK[t] if dprev<e['ex_date']<=d]
                    for e in evs:
                        if e['type']=='unresolved' or e['action']=='fail': return fail(d,'corporate_action_unresolved',e['event_id'])
                        if e['action'] in('skip','ignore'): continue
                        cs=float(e['cash_per_share'] or 0);flags.add('ext_pack_event:'+e['event_id'])
                        if e['type'] in('spin','special_div'):
                            g=n*float(e['ratio'])*float(e['child_first_close'] or 0);cash+=n*cs+g-c*g;ids.add(e['event_id'])
                        elif e['type']=='cash': cash+=n*cs;del hold[t];ids.add(e['event_id'])
                        elif e['type']=='stock':
                            acq=e['child_or_acquirer'];aser=e['acquirer_series']
                            if (aser,acq,d) not in CL or (acq in hold and acq!=t): return fail(d,'corporate_action_unresolved',t+'->'+acq)
                            cash+=n*cs;del hold[t];hold[acq]=(n*float(e['ratio']),aser,CL[(aser,acq,d)]);ids.add(e['event_id'])
                    if t not in hold: continue
                if ser.startswith('ext:') and (ser,t) in s.DELIST and d>s.DELIST[(ser,t)] and (ser,t,d) not in CL:
                    lp=CL[(ser,t,s.DELIST[(ser,t)])];g=n*lp;cash+=g-c*g;del hold[t];flags.add('ext_delist_cash:'+t+'@'+s.DELIST[(ser,t)]);continue
                if ser.startswith('ext:'):     # 2005-2015 extension series: rank-side spin/special div as shares/=F
                    for e in s.CA.get(s.LBL[t],[]):
                        if e['kind'] in('spin','special_div') and dprev<e['ex_date']<=d:
                            if not e['factor_F']: return fail(d,'corporate_action_unresolved',t+':'+e['event_id'])
                            n=n/float(e['factor_F']);hold[t]=(n,ser,last);flags.add('ext_spin_F:'+e['event_id'])
                        if e['kind']=='unresolved_BLOCKER' and dprev<e['ex_date']<=d: return fail(d,'corporate_action_unresolved',e['event_id'])
                    continue
                evs=[e for e in s.HEV[(ser,t)] if dprev<e['ex_date']<=d]
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
                elif d>s.SLAST.get((ser,t),'9999'): return fail(d,'corporate_action_unresolved',t)
                else: return fail(d,'daily_gap',t)
            if d in nxt or d==D0:
                S=SES[SI[d]-1]
                off=bool(regime and regime.get(S))
                if off: sel=[]
                else:
                    r,miss,rids=s.rank(S,W,frozenset(excl),skip)
                    if miss: return fail(d,'rank_missing_unbounded',' '.join(miss))
                    sel=[t for _,t in r[:K]];ids|=rids
                inv=[t for t in sel if t not in cash_on]
                keys=[s.ek(t,d) for t in inv]
                if len(set(keys))!=len(keys): return fail(d,'duplicate_economic_execution_key','|'.join(inv))
                sers={}
                for t,k in zip(inv,keys):
                    s.LBL[k]=t
                    if k in hold:
                        hs=hold[k][1]
                        if s.y0==2015 or s.SLAST.get((hs,k),'0000')>=min(nxt[d],END) or any(e['type'] in('cash','stock') for e in s.HEV[(hs,k)]): sers[k]=hs;continue
                        # held series does not cover the new window (extension only): get the new window's series and
                        # rescale shares at today's close (value-preserving, no trade, no fee)
                        a=s.EXA.get((d,t,nxt[d]))
                        if a is not None and a['status']=='ok': ns_=a['price_series']
                        else:
                            st,ns_,det=s.ext(t,k,d,nxt[d]);flags.add('ext_exec')
                            if st!='ok': return fail(d,det if st!='not_tradable' else 'pick_not_tradable',t)
                        if (ns_,k,d) not in CL or (hs,k,d) not in CL: return fail(d,'daily_gap',t+'(series handover)')
                        n0,_,_=hold[k];hold[k]=(n0*CL[(hs,k,d)]/CL[(ns_,k,d)],ns_,CL[(ns_,k,d)]);sers[k]=ns_;flags.add('series_handover');continue
                    a=s.EXA.get((d,t,nxt[d]))
                    if a is None:
                        st,se,det=s.ext(t,k,d,nxt[d]);flags.add('ext_exec')
                        if st=='not_tradable': return fail(d,'pick_not_tradable',t)
                        if st!='ok': return fail(d,det,t)
                        sers[k]=se;continue
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
                hold=new;cash=Ep-sum(T.values());val=Ep;log.append((S,d,'|'.join(sel) if sel else 'CASH(regime)','|'.join(inv) if inv else 'CASH'))
            if d==END: g=sum(n*l for n,_,l in hold.values());val=cash+g-c*g;ntr+=len(hold)
            eq.append(val);dts.append(d);dprev=d
        if K==1:
            for S,d,p,_ in log:
                if any(S<e<=d for e in s.DEAL.get(p,[])): return fail(d,'pick_not_tradable',p)
        yrs=(dt.date.fromisoformat(END)-dt.date.fromisoformat(D0)).days/365.25;pk=-1;mdd=0.0
        for v in eq: pk=max(pk,v);mdd=min(mdd,v/pk-1)
        st='DATA_COMPLETE_PROVISIONAL_C' if any(s.GRADE.get(i)=='C' for i in ids) else 'DATA_COMPLETE_CONDITIONAL'
        return dict(status=st,final=eq[-1],CAGR=eq[-1]**(1/yrs)-1,MDD=mdd,rebalances=log,n_trades=ntr,entry=D0,end=END,flags=flags,
                    nav=(dts,eq) if want_nav else None)
    LBL={}
