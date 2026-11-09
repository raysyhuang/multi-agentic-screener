"""Minimal reference rebuild from this pack's CSVs only (no engine code, no network). K=1 (Top1) paths.
Usage: python3 reference_rebuild.py   -> rebuilds annual Top1 (W12H12 phase 1), W10H2 phase 1, W12H4 phase 1."""
import csv,gzip,math,collections,datetime as dt
D='./';c=0.0025
def rd(f):
    o=gzip.open(D+f,'rt') if f.endswith('.gz') else open(D+f)
    return list(csv.DictReader(o))
SES=[r['session'] for r in rd('trading_sessions.csv')]
ME=sorted({max(s for s in SES if s[:7]==m) for m in {s[:7] for s in SES}})     # last session of each month
ME=[m for m in ME if m>='2013-11-29']
ELIG=collections.defaultdict(list)
for r in rd('eligibility_by_signal.csv.gz'):
    if r['eligible_final']=='True': ELIG[r['signal_date']].append(r['engine_ticker'])
PX={(r['engine_ticker'],r['month_end']):float(r['rank_price']) for r in rd('rank_prices_month_end.csv.gz')}
OV={}
for r in rd('rank_price_overrides.csv'):
    OV[(r['engine_ticker'],r['month_end'])]=OV.get((r['engine_ticker'],r['month_end']),[])+[(r['signal_from'],r['signal_to'],float(r['rank_price']))]
def px(t,d,S):
    for a,b,v in OV.get((t,d),[]):
        if a<=S<=b: return v
    return PX.get((t,d))
CA=collections.defaultdict(list)
for r in rd('corporate_actions_applied.csv'): CA[r['engine_ticker']].append(r)
def rank(S,W):
    B=ME[ME.index(S)-W];r=[];miss=[]
    for t in sorted(ELIG[S]):
        ev=CA[t]
        if any(e['kind']=='legacy_data_error_BLOCKER' and e['blocker_from']<=S<=e['blocker_to'] for e in ev): miss.append(t);continue
        if any(e['kind']=='unresolved_BLOCKER' and B<e['ex_date']<=S for e in ev): miss.append(t);continue
        a,b=px(t,B,S),px(t,S,S)
        if a and b:
            for e in ev:
                if e['kind'] in('spin','special_div') and B<e['ex_date']<=S:
                    if not e['factor_F']: a=None;break
                    a*=float(e['factor_F'])
        if a and b and a>0: r.append((b/a-1,t))
        else: miss.append(t)
    r.sort(key=lambda x:(-x[0],x[1]));return r,miss
EXS={(r['config'],r['phase'],r['trade_date'],r['engine_ticker']):r['price_series'] for r in rd('execution_prices_picks.csv')}
DC={(r['price_series'],r['engine_ticker'],r['date']):float(r['close']) for r in rd('execution_daily_closes_held.csv.gz')}
HE=collections.defaultdict(list)
for r in rd('holding_side_events.csv'): HE[(r['config'],r['phase'],r['engine_ticker'])].append(r)   # events the path actually met
def first_td(y,m): return min(s for s in SES if s[:7]==f'{y:04d}-{m:02d}')
def path(W,H,ph):
    cfg=f'W{W}H{H}K1';D0=first_td(2015,ph);END=first_td(2025,ph);rebs=[];j=0
    while True:
        mm=(ph-1)+j*H;d=first_td(2015+mm//12,mm%12+1)
        if d>=END: break
        rebs.append(d);j+=1
    days=SES[SES.index(D0):SES.index(END)+1];hold=None;cash=1.0;eq=[];picks=[];dprev=D0;seen=set()
    for d in days:
        if hold:
            t,n,ser=hold
            for e in HE[(cfg,str(ph),t)]:
                k=(e['event_id'],e['date'])
                if dprev<e['date']<=d and k not in seen:
                    seen.add(k)
                    if e['event'] in('spin','special_div'):
                        g=n*float(e['ratio'])*float(e['child_first_close'] or 0);cash+=n*float(e['cash_per_share'] or 0)+g-c*g
                    elif e['event']=='cash_deal': cash+=n*float(e['cash_per_share']);hold=None
                    elif e['event']=='stock_deal':   # holder receives ratio acquirer shares (+cash), then follows the acquirer series
                        cash+=n*float(e['cash_per_share'] or 0);hold=(e['child_or_acquirer'],n*float(e['ratio']),'acq:'+e['child_or_acquirer'])
                    else: return dict(status='UNSUPPORTED_EVENT '+e['event'],picks=picks)
        val=cash+(hold[1]*DC[(hold[2],hold[0],d)] if hold else 0)
        if d in rebs:
            S=SES[SES.index(d)-1];r,miss=rank(S,W)
            if miss: return dict(status='UNKNOWN rank_missing '+S,picks=picks)
            t=r[0][1];picks.append((S,t,r[0][0],d))
            if hold and hold[0]==t: h=hold[1]*DC[(hold[2],t,d)];Ep=(val+c*h)/(1+c) if cash>0 else val
            else:
                h=hold[1]*DC[(hold[2],hold[0],d)] if hold else 0.0;Ep=(val-c*h)/(1+c)
                ser=EXS[(cfg,str(ph),d,t)];hold=(t,0,ser)
            hold=(t,Ep/DC[(hold[2],t,d)],hold[2]);cash=0.0;val=Ep
        if d==END: g=hold[1]*DC[(hold[2],hold[0],d)] if hold else 0;val=cash+g-c*g
        eq.append(val);dprev=d
    yrs=(dt.date.fromisoformat(END)-dt.date.fromisoformat(D0)).days/365.25;peak=-1;mdd=0
    for v in eq: peak=max(peak,v);mdd=min(mdd,v/peak-1)
    return dict(status='OK',picks=picks,final=eq[-1],CAGR=eq[-1]**(1/yrs)-1,MDD=mdd,entry=D0,end=END)
if __name__=='__main__':
    REF={(r['config'],r['phase']):r for r in rd('reference_engine_paths_K1.csv')}
    ENG=collections.defaultdict(list)
    for r in rd('execution_prices_picks.csv'): ENG[(r['config'],r['phase'])].append((r['signal_date'],r['engine_ticker']))
    for W,H in [(12,12),(10,2),(12,4)]:
        cfg=f'W{W}H{H}K1';o=path(W,H,1);ref=REF[(cfg,'1')]
        same=[(s,t) for s,t,_,_ in o['picks']]==ENG[(cfg,'1')]
        print(f"{cfg} phase1 {o['status']} picks={len(o['picks'])} picks_match_engine={same}")
        if W==12 and H==12: print('  annual picks:',' '.join(f"{d[:4]}:{t}" for s,t,_,d in o['picks']))
        if 'CAGR' in o:
            print(f"  final={o['final']:.6f}x CAGR={o['CAGR']*100:.4f}% MDD={o['MDD']*100:.4f}% | engine CAGR={float(ref['CAGR'])*100:.4f}% MDD={float(ref['daily_MDD'])*100:.4f}% "
                  f"|dCAGR|={abs(o['CAGR']-float(ref['CAGR'])):.2e}")
