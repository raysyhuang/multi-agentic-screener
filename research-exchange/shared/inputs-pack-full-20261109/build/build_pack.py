"""Builds the independent-reselection input pack from the frozen v17 strict4 engine + v18 KORS/CPRI dedupe (read-only).
Writes only into /workspace/momentum432-inputs-pack-20261109/. Research only."""
import sys,json,bisect,gzip,collections,pandas as pd,numpy as np
sys.path.insert(0,'/workspace/momentum432-inputs-pack-20261109/build')
from load_v17 import load
P='/workspace/momentum432-inputs-pack-20261109/'
ns=load()
MEL,SD,mpx,C,seasoned=ns['MEL'],ns['SD'],ns['mpx'],ns['C'],ns['seasoned']
OLDT,REV,IDR,RNEW,ca_of,pex3=ns['OLDT'],ns['REV'],ns['IDR'],ns['RNEW'],ns['ca_of'],ns['pex3']
# ---- v18 dedupe (identical to momentum432-v18-dedupe-20261109/run_grid_v18.py mode v18)
_rs=ns['rank_strict']
def rank_strict(S,W):
    r,miss,nu=_rs(S,W);tk={t:g for g,t in r}
    if 'CPRI' in tk and 'KORS' in tk:
        assert tk['CPRI']==tk['KORS'];r=[(g,t) for g,t in r if t!='KORS'];nu-=1
    return r,miss,nu
ns['rank_strict']=rank_strict;ns['RS'].clear();ns['RCAS'].clear()
def dedupe_drop(S,t): return t=='KORS' and 'CPRI' in seasoned(S) and 'KORS' in seasoned(S)
def tas(t,S):
    if t=='CPRI' and S<'2019-01-02': return 'KORS'
    if t in RNEW and S<RNEW[t][0]: return RNEW[t][1]
    if C(S,t,False): return t
    for x in IDR:
        if x['member']==t and x['valid_from']<=S<=x['valid_to']: return x['series'].split(':',1)[1]
    for nm in (OLDT.get(t),REV.get(t)):
        if nm and C(S,nm,False): return nm
    return t
SIG=[m for m in MEL if '2014-12-01'<=m<='2025-11-30'];PME=[m for m in MEL if '2013-11-01'<=m<='2025-11-30']
mem,md=ns['mem'],ns['md'];orig=ns['_MEM14_ORIGINAL'];omd=[d for d,_ in orig]
def memat(S,M=mem,D=md): return set(M[bisect.bisect_right(D,S)-1][1])
# ---- 1a membership intervals (engine snapshots after v14 patches), restricted to intervals touching 2012-12-01..2025-01-31
iv=[];run={}
for k,(d,s) in enumerate(mem):
    for t in list(run):
        if t not in s: iv.append((t,run.pop(t),d))
    for t in s:
        if t not in run: run[t]=d
for t,a in run.items(): iv.append((t,a,''))
patches=json.load(open('/workspace/second-sleeve-v17-targeted-ca-20261108/inputs/membership_patches.json'))
rows=[]
for t,a,b in iv:
    if (b and b<'2012-12-01') or a>'2025-01-31': continue
    src='sp500_hist_upd.csv (public reconstruction, /workspace/prior-quarter-winner-check-2026-10-20)'
    pt=[p for p in patches if p['ticker']==t and (p['from_date']==b or p['until_date']==a or (a<=p['from_date']<(b or '9999')))]
    if pt: src+=' + v14 official-date patch: '+' | '.join(p['reason'] for p in pt)
    al=[];prev=None
    for m in [x for x in MEL if a<=x<(b or '9999') and x<='2025-11-30']:
        z=tas(t,m)
        if z!=prev: al.append(f'{z} from {m}');prev=z
    note=''
    if t in('KORS','CPRI'): note='v18 dedupe: KORS and CPRI are the same company (CIK 1530721); where both are eligible on a signal date KORS is removed and the company ranks once under engine key CPRI (ticker_at_signal KORS before 2019-01-02)'
    rows.append(dict(engine_ticker=t,start_date=a,end_date_exclusive=b,ticker_at_signal_by_month_end=('; '.join(al) if any(not x.startswith(t+' ') for x in al) else t),source=src,v18_dedupe_note=note))
pd.DataFrame(rows).sort_values(['engine_ticker','start_date']).to_csv(P+'membership_intervals.csv',index=False)
# ---- 1b eligibility per signal
E1=[]
for S in SIG:
    u=seasoned(S);m=memat(S)
    for t in sorted(u|m):
        el=t in u;dd=el and dedupe_drop(S,t)
        E1.append(dict(signal_date=S,engine_ticker=t,ticker_at_signal=tas(t,S),member_at_signal=t in m,eligible_365d=el,
            eligible_via_rename_bridge=(S,t) in ns['REN_ADD'],v18_dedupe_removed=dd,eligible_final=el and not dd))
E1=pd.DataFrame(E1);E1.to_csv(P+'eligibility_by_signal.csv.gz',index=False,compression='gzip')
UNI=sorted(E1[E1.eligible_final].engine_ticker.unique())
# ---- 2a rank prices: base price p(t,m)=mpx(m,t,S=m); overrides where the engine's price at m depends on the signal S
BASE={(t,m):mpx(m,t,m) for t in UNI for m in PME}
def _src(t,m,v):
    if C(m,t,False)==v: return 'massive_grouped_close:'+t
    for nm in (OLDT.get(t),REV.get(t),RNEW.get(t,(None,None))[1]):
        if nm and C(m,nm,False)==v: return 'massive_grouped_close_alias:'+nm
    return 'other_engine_rule(identity series/successor conversion/deal value/class or receipt units)'
pd.DataFrame([dict(engine_ticker=t,month_end=m,rank_price=v,price_source=_src(t,m,v)) for (t,m),v in BASE.items() if v]).to_csv(P+'rank_prices_month_end.csv.gz',index=False,compression='gzip')
OV=collections.defaultdict(list)
for S in SIG:
    u=set(E1[(E1.signal_date==S)&E1.eligible_final].engine_ticker)
    for t in u:
        for W in range(1,13):
            for d in (MEL[MEL.index(S)-W],S):
                v=mpx(d,t,S)
                if v!=BASE[(t,d)]: OV[(t,d,v)].append(S)
ovr=[]
for (t,d,v),ss in OV.items():
    ss=sorted(set(ss));ovr.append(dict(engine_ticker=t,month_end=d,signal_from=ss[0],signal_to=ss[-1],n_signals=len(ss),rank_price=v,
        reason={'CB':'ticker reuse: for signals >=2016-01-15 CB is Chubb Ltd (ex-ACE), earlier prices from ACE','AGN':'issuer routing: AGN = Actavis/Allergan plc for signals >=2015-03-17; contractual ABBV value after 2020-05-08'}.get(t,'signal-dependent engine price')))
ovr=pd.DataFrame(ovr);ovr.to_csv(P+'rank_price_overrides.csv',index=False)
for (t,d,v),ss in OV.items(): assert all(SIG[SIG.index(ss[0]):SIG.index(ss[-1])+1]) # ranges listed explicitly by n_signals
# ---- 2b corporate actions applied on the ranking side + blockers + vendor de-adjustment q factors
CA=[]
for t in UNI:
    for ca in ca_of(t):
        if ca['type'] in('spin','special_div'):
            pe=pex3(t,ca['ex_date']);ch=float(ca['ratio'])*float(ca['child_first_close'] or 0)+float(ca['cash_per_share'])
            CA.append(dict(engine_ticker=t,event_id=ca['id'],kind=ca['type'],ex_date=ca['ex_date'],ratio=ca['ratio'],child=ca['child_or_acquirer'],child_first_close=ca['child_first_close'],
                cash_per_share=ca['cash_per_share'],parent_price_at_ex_rank_units=pe,factor_F=(pe/(pe+ch) if pe else ''),evidence_grade=ca.get('evidence_grade',''),evidence_url=ca['evidence_url'],
                rule='multiply lookback-START price by F when lookback_start < ex_date <= signal_date; F blank -> name unrankable'))
        elif ca['type']=='unresolved':
            CA.append(dict(engine_ticker=t,event_id=ca['id'],kind='unresolved_BLOCKER',ex_date=ca['ex_date'],evidence_grade=ca.get('evidence_grade',''),evidence_url=ca['evidence_url'],
                rule='lookback_start < ex_date <= signal_date -> name unrankable -> whole signal UNKNOWN'))
        elif ca['type'] in('cash','stock','cash_stock','election','merger'):
            CA.append(dict(engine_ticker=t,event_id=ca['id'],kind='deal_'+ca['type'],ex_date=ca['ex_date'],ratio=ca['ratio'],child=ca['child_or_acquirer'],cash_per_share=ca['cash_per_share'],
                evidence_grade=ca.get('evidence_grade',''),evidence_url=ca['evidence_url'],rule='holding side: deal leg at documented date; rank side: after close the rank price is the deal value (already inside rank_prices); a pick whose deal date is in (exec-60d, exec] is untradable -> path UNKNOWN'))
for a,b,e in ns['SKIPL']:
    CA.append(dict(engine_ticker=a,event_id=f'SKIP_{a}_{b}',kind='legacy_data_error_BLOCKER',ex_date=b,blocker_from=b,blocker_to=e,rule=f'signal in [{b},{e}] -> name unrankable -> signal UNKNOWN'))
for t,l in ns['QT'].items():
    for ex,q in l: CA.append(dict(engine_ticker=t,event_id=f'QADJ_{t}_{ex}',kind='vendor_spin_deadjust_q_ALREADY_IN_PRICES',ex_date=ex,factor_F=q,rule='rank_prices already divided by q for dates < ex (removes Massive own spin adjustment so F is applied once)'))
pd.DataFrame(CA).drop_duplicates().to_csv(P+'corporate_actions_applied.csv',index=False)
# ---- engine rankings for verification (all frames, v18)
RK=[]
for S in SIG:
    for W in range(1,13):
        r,miss,nu=rank_strict(S,W);ns['RS'][(S,W)]=(r,miss,nu)
        for i,(g,t) in enumerate(r): RK.append(dict(signal_date=S,W=W,rank=i+1,engine_ticker=t,ticker_at_signal=tas(t,S),lookback_return=g))
        for t in miss: RK.append(dict(signal_date=S,W=W,rank=None,engine_ticker=t,ticker_at_signal=tas(t,S),lookback_return=None))
pd.DataFrame(RK).to_csv(P+'reference_engine_rankings_v18.csv.gz',index=False,compression='gzip')
# ---- 2c execution prices + daily closes for K=1 W10H2, W12H4, W12H12 (all 12 phases); held-side events
EX=[];DY={};CONF=[];HE=[];PS=[]
import os
def fmp_open(t,d):
    p=f'/workspace/nh-matrix-bt-v2/bars/{t}.px.json'
    if not os.path.exists(p): return None,None
    x={r['date']:(r['open'],r['close']) for r in json.load(open(p))};return x.get(d,(None,None))
for W,H in [(10,2),(12,4),(12,12)]:
    for ph in range(1,13):
        hold_ser={}
        PREV={}
        def hook(w,h,k,p,d,val,cash,hold,terminal):
            cur={t:(n,nm,s,last) for t,(n,nm,s,last) in hold.items()};allh={**PREV,**cur};PREV.clear();PREV.update(cur)
            for t,(n,nm,s,last) in allh.items():
                key=(nm,t,d)
                if d in s.index:
                    v=float(s[d])
                    if key in DY and abs(DY[key]/v-1)>1e-12: CONF.append(key)
                    DY[key]=v
        ns['_snapshot_hook']=hook
        row,eq,led=ns['path_strict4'](W,H,1,ph);ns.pop('_snapshot_hook')
        PS.append(dict(config=f'W{W}H{H}K1',phase=ph,entry=row['entry'],end=row['end'],status=row['status'],final_multiple=row['wealth'],CAGR=row['CAGR'],daily_MDD=row['daily_MDD']))
        for l in led:
            if l['event']=='rebalance':
                for t,src in zip(l['picks'].split('|'),l['src'].split('|')):
                    c_=DY.get((src,t,l['date']));o,fc=fmp_open(t,l['date']) if src.startswith('fmp:') else (None,None)
                    same=fc is not None and c_ is not None and abs(fc/c_-1)<1e-9
                    EX.append(dict(config=f'W{W}H{H}K1',phase=ph,signal_date=l['signal'],trade_date=l['date'],engine_ticker=t,ticker_at_signal=tas(t,l['signal']),
                        exec_close=c_,exec_open=(o if same else None),open_note=('same FMP file/scale as close' if same else 'UNKNOWN: open not available on the same price scale'),price_series=src))
            elif l['event'] in('spin','special_div','cash_deal','stock_deal'):
                ca=ns['CAID'][l['ca']];HE.append(dict(config=f'W{W}H{H}K1',phase=ph,date=l['date'],event=l['event'],engine_ticker=l['ticker'],event_id=l['ca'],ratio=ca['ratio'],
                    child_or_acquirer=ca['child_or_acquirer'],child_first_close=ca['child_first_close'],cash_per_share=ca['cash_per_share'],evidence_url=ca['evidence_url']))
pd.DataFrame(EX).drop_duplicates().to_csv(P+'execution_prices_picks.csv',index=False)
pd.DataFrame([dict(price_series=s,engine_ticker=t,date=d,close=v) for (s,t,d),v in sorted(DY.items())]).to_csv(P+'execution_daily_closes_held.csv.gz',index=False,compression='gzip')
pd.DataFrame(HE).drop_duplicates().to_csv(P+'holding_side_events.csv',index=False)
pd.DataFrame(PS).to_csv(P+'reference_engine_paths_K1.csv',index=False)
pd.DataFrame({'session':[d for d in SD if '2013-11-01'<=d<='2025-12-31']}).to_csv(P+'trading_sessions.csv',index=False)
print('universe',len(UNI),'overrides',len(ovr),'conflicts',len(CONF),'series',len({k[0] for k in DY}))
