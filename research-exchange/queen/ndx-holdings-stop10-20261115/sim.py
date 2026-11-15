"""Queen 2026-11-15: holdings ledger + Ray's stop/re-entry rule on the verified post rule
(NDX100 PIT, 6m mean of monthly returns, Top3 EW, monthly, 25bp/side, price only).
Imports the read-only engine at /workspace/ndx100-fill-20261114-v1.1 (unmodified). Run with system python3."""
import sys,os,math,csv,json,bisect
ENG='/workspace/ndx100-fill-20261114-v1.1'
sys.path.insert(0,ENG); os.chdir(ENG)
from engine import *          # ME, IDX, PX, ranking, run, stats, qqq, bench, days_between
import lib
OUT='/workspace/ndx-holdings-stop10-20261115/out'
COST=0.0025; K=3
A,B='2014-12-31','2026-11-13'; i0,i1=IDX[A],IDX[B]

# ---------- adjusted open = adjusted close(d) * raw_open(d)/raw_close(d); mirror lib's rename stitching for the o/c ratio ----------
def load_ratio(path):
    out={}
    for r in csv.DictReader(open(path)):
        if not r.get('t'): continue
        try: o=float(r['o']); c=float(r['c'])
        except: continue
        if o>0 and c>0: out[lib.d_of(r['t']).isoformat()]=o/c
    return out
RAT={os.path.basename(f)[:-4]:load_ratio(f) for f in glob.glob(os.path.join(ENG,'raw/adj/*.csv'))}
for p,s,d in lib.RENAMES:   # same order as lib.build_series
    P=RAT.get(p) or {}; S=RAT.get(s) or {}
    D_P=PX.D.get(p); 
    pb={x:v for x,v in P.items() if x<d}; sa={x:v for x,v in S.items() if x>=d}
    if not pb or not sa: continue
    new=dict(pb); new.update(sa); RAT[s]=new
    newp=dict(pb); newp.update(sa); RAT[p]=newp
OPEN_MISS=[0]
def adj_open(key,d):
    c=PX.D.get(key,{}).get(d)
    r=RAT.get(key,{}).get(d)
    if c is None: return None
    if r is None: OPEN_MISS[0]+=1; return c
    return c*r

# ---------- simulator ----------
def simulate(stop=None,mode='entry',exe='close',log_events=None,reent='sell'):
    """stop: fraction (0.10) or None. mode 'entry' (line = entry*(1-x)) or 'trail' (line = post-entry peak close*(1-x)).
    exe 'close' = act at the triggering close (optimistic); 'open' = act at next trading day's open.
    reent='sell' (Ray's literal rule, default): re-entry trigger = close >= executed sell price.
    reent='line': re-entry trigger = close >= max(executed sell price, stop line at time of stop) (sensitivity).
    The stop line itself is unchanged after a re-entry (entry*(1-x), or post-entry peak*(1-x) with the peak kept)."""
    nav=1.0; navs=[(ME[i0],1.0)]; dnav=[1.0]; ddates=[ME[i0]]; rec=[]; prev={}
    ev=dict(stops=0,reentries=0,whipsaw=0,stop_helped=0,stop_hurt=0,cost_stop=0.0,cash_slot_days=0,slot_days=0,month_only=0)
    for i in range(i0,i1):
        picks=ranking(i)[:K]
        tgt={t:1.0/len(picks) for s,t,k in picks}
        traded=sum(abs(tgt.get(t,0)-prev.get(t,0)) for t in set(tgt)|set(prev))
        c=COST*traded; base=nav*(1-c)
        dd=days_between(ME[i],ME[i+1])
        sleeves={}; endw={}; cashv=0.0; mcost=0.0
        daily_tot=[0.0]*len(dd)
        for s,t,key in picks:
            path=PX.daily_path(key,ME[i],ME[i+1],dd)      # closes rel. to entry close, forward-filled within segment
            v0=base*tgt[t]; e0=PX.at(key,ME[i])[0]
            if stop is None or key not in PX.k:
                if stop is not None and key not in PX.k: ev['month_only']+=1
                for j in range(len(dd)): daily_tot[j]+=v0*path[j]
                endw[t]=v0*path[-1]; ev['slot_days']+=len(dd); continue
            inpos=True; units=v0/1.0; cash=0.0; peak=1.0; line=None; sellpx=None; trig=None; pend=None
            stoplog=[]
            for j,d in enumerate(dd):
                px=path[j]
                # execute pending order at today's open
                if pend and exe=='open':
                    o=adj_open(key,d); o=(o/e0) if o is not None else px
                    if PX.D[key].get(d) is None: o=px   # no bar today (ffill) -> act at carried price
                    if pend=='sell' and inpos:
                        val=units*o; cst=val*COST; cash=val-cst; units=0; inpos=False; sellpx=o
                        trig=(sellpx if reent=='sell' else max(sellpx,line)); ev['stops']+=1; ev['cost_stop']+=cst/nav; mcost+=cst; stoplog.append(['S',d,o])
                    elif pend=='buy' and not inpos:
                        cst=cash*COST; units=(cash-cst)/o; ev['cost_stop']+=cst/nav; mcost+=cst; cash=0; inpos=True; ev['reentries']+=1
                        if o>sellpx: ev['whipsaw']+=1
                        stoplog.append(['B',d,o])
                    pend=None
                last=(j==len(dd)-1)
                if inpos:
                    peak=max(peak,px)
                    L=(1-stop) if mode=='entry' else peak*(1-stop)
                    if px<=L+1e-12 and not last:
                        line=L
                        if exe=='close':
                            val=units*px; cst=val*COST; cash=val-cst; units=0; inpos=False; sellpx=px; trig=(px if reent=='sell' else max(px,L))
                            ev['stops']+=1; ev['cost_stop']+=cst/nav; mcost+=cst; stoplog.append(['S',d,px])
                        else: pend='sell'
                else:
                    if px>=trig-1e-12 and not last:
                        if exe=='close':
                            cst=cash*COST; units=(cash-cst)/px; ev['cost_stop']+=cst/nav; mcost+=cst; cash=0; inpos=True; ev['reentries']+=1
                            if px>sellpx: ev['whipsaw']+=1
                            stoplog.append(['B',d,px])
                        else: pend='buy'
                val=units*px+cash; daily_tot[j]+=val
                ev['slot_days']+=1
                if not inpos: ev['cash_slot_days']+=1
            # each stop: helped if month-end close below that sell price
            for e in stoplog:
                if e[0]=='S':
                    if path[-1]<e[2]: ev['stop_helped']+=1
                    else: ev['stop_hurt']+=1
            if log_events is not None and stoplog:
                log_events.append(dict(signal=ME[i],ticker=t,events=' '.join('%s:%s@%.4f'%(a,b,cc) for a,b,cc in stoplog),
                                       stock_month_ret=path[-1]-1,sleeve_month_ret=(units*path[-1]+cash)/v0-1))
            if inpos: endw[t]=units*path[-1]
            else: endw[t]=0.0; cashv+=cash
        V=daily_tot[-1]
        for j in range(len(dd)): dnav.append(daily_tot[j]); ddates.append(dd[j])
        net=V/nav-1; nav=V; navs.append((ME[i+1],nav))
        prev={t:w/V for t,w in endw.items() if w>0}
        rec.append(dict(signal=ME[i],end=ME[i+1],picks=' '.join(t for s,t,k in picks),net=net,traded=traded,cost=c,stopcost=mcost))
    return dict(navs=navs,dnav=dnav,ddates=ddates,rec=rec,ev=ev)

if __name__=='__main__':
    # 1) baseline reproduction with the engine itself
    eng=run(K,i0,i1,(),COST); se=stats(eng)
    print('ENGINE baseline: CAGR %.4f%% MDDd %.2f%% MDDm %.2f%% total x%.2f'%(se['CAGR']*100,se['MDD_daily']*100,se['MDD_monthly']*100,se['total']))
    mine=simulate(None); sm=stats(mine)
    print('SIM(no stop)  : CAGR %.4f%% MDDd %.2f%% MDDm %.2f%% total x%.2f'%(sm['CAGR']*100,sm['MDD_daily']*100,sm['MDD_monthly']*100,sm['total']))
    md=max(abs(a[1]-b[1]) for a,b in zip(eng['navs'],mine['navs'])); print('max |NAV diff| engine vs sim:',md)
    dd2=max(abs(a-b) for a,b in zip(eng['dnav'],mine['dnav'])); print('max |daily NAV diff|:',dd2, len(eng['dnav']),len(mine['dnav']))
