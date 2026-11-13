#!/usr/bin/env python3
"""Claude: independent re-implementation of Queen's "SPMO-rule TopK" from her SPEC.md section 2 (not from spmo_topk.py).
Uses her shared derived daily closes plus the two packs' membership and corporate-action files.
Checks (a) Top3 picks at every rebalance, (b) CAGR / max drawdown for Top1-3: no stop, no-NVDA, 30% trailing stop.
Windows: 2015-01-02 -> 2025-01-02 and 2005-01-03 -> 2015-01-02. Stdlib only.  Usage: python3 check_spmo_topk.py"""
import csv,gzip,os,collections,statistics as st,datetime as dt,bisect
HERE=os.path.dirname(os.path.abspath(__file__)); R=os.path.join(HERE,'..','..'); Q=os.path.join(R,'queen','spmo-topk-20261113'); C=0.0025
PACK={'2015':os.path.join(R,'shared','inputs-pack-full-20261109'),'2005':os.path.join(R,'queen','w10h2-w12h4-oos-2005-2015-20261112','pack')}
def rd(f): return list(csv.DictReader(gzip.open(f,'rt') if f.endswith('.gz') else open(f)))
def load(era):
    px=collections.defaultdict(dict)
    for r in rd(os.path.join(Q,'data',f'derived_daily_closes_{era}_px.csv.gz')): px[r['engine_ticker']][r['date']]=float(r['close'])
    cnt=collections.Counter(d for v in px.values() for d in v); sess=sorted(d for d,n in cnt.items() if n>=100)
    iv=collections.defaultdict(list)
    for r in rd(os.path.join(PACK[era],'membership_intervals.csv')): iv[r['engine_ticker']].append((r['start_date'],r['end_date_exclusive'] or '9999-12-31'))
    ca=collections.defaultdict(list)
    for r in rd(os.path.join(PACK[era],'corporate_actions_applied.csv')):
        if r['kind'] in('spin','special_div') and r['factor_F']: ca[r['engine_ticker']].append((r['ex_date'],float(r['factor_F'])))
    rp={(r['engine_ticker'],r['month_end']):float(r['rank_price']) for r in rd(os.path.join(PACK[era],'rank_prices_month_end.csv.gz')) if r['rank_price']}
    dates={t:sorted(v) for t,v in px.items()}
    return px,dates,sess,iv,ca,rp
def last_session(sess,y,m):
    while m<=0: y-=1; m+=12
    pre='%04d-%02d'%(y,m); z=[d for d in sess if d[:7]==pre]; return z[-1] if z else None
def third_friday(sess,y,m):
    d=dt.date(y,m,1); d+=dt.timedelta((4-d.weekday())%7+14); s=d.isoformat(); i=bisect.bisect_right(sess,s)-1; return sess[i]
def price_on(px,sess,t,d,back=10):
    i=sess.index(d)
    for j in range(i,max(i-back,-1)-1,-1):
        if sess[j] in px[t]: return sess[j],px[t][sess[j]]
    return None
def select_spec(era,D,y,m,excl=()):
    """Selection exactly as SPEC.md section 2 describes it: daily prices only; a name without a usable daily series is skipped."""
    px,dates,sess,iv,ca,rp=D
    ref=last_session(sess,y,m-1); pe=last_session(sess,y,m-2); ps=last_session(sess,y-1,m-2); ps9=last_session(sess,y,m-11); lim=ps9
    if None in (ref,pe): return None
    rows=[]
    for t in iv:
        if t in excl or t not in px or not any(a<=ref<b for a,b in iv[t]): continue
        if lim is None or dates[t][0]>lim: continue
        e=price_on(px,sess,t,pe)
        if not e: continue
        s=price_on(px,sess,t,ps) if ps else None
        if not s and ps9: s=price_on(px,sess,t,ps9)
        if not s: continue
        (d0,p0),(d1,p1)=s,e
        F=1.0; evs=[(x,f) for x,f in ca.get(t,()) if d0<x<=d1]
        for x,f in evs: F*=f
        adj=True
        if evs:
            a,b=rp.get((t,d0)),rp.get((t,d1))
            if a and b and abs((p1/p0)/((b/(a*F)))-1)<=0.02: adj=False
        mv=p1/(p0*(F if adj else 1.0))-1
        ds=[d for d in dates[t] if d0<=d<=d1]; rets=[]
        for u,v in zip(ds,ds[1:]):
            f=1.0
            if adj:
                for x,ff in evs:
                    if u<x<=v: f*=ff
            rets.append(px[t][v]/(px[t][u]*f)-1)
        if len(rets)<2: continue
        sg=st.stdev(rets)
        if sg<=0: continue
        rows.append((-(mv/sg),t,mv))
    rows.sort(); return dict(ref=ref,exec=third_friday(sess,y,m),top=[(t,-r,mv) for r,t,mv in rows],n=len(rows))
def select(era,D,y,m,excl=(),fix=None):
    """Queen's selection INCLUDING the data-quality fallbacks that are in her code but not in SPEC.md section 2:
    - daily return disagrees with the pack's month-end return by >2%  -> momentum value taken from the pack, sigma untrusted
    - a daily move above 75% inside the window                       -> sigma untrusted (momentum value kept)
    - no usable daily series                                          -> momentum value from the pack, sigma untrusted
    - untrusted sigma -> monthly-return stdev from the pack x median(daily/monthly) ratio, else the universe median sigma."""
    px,dates,sess,iv,ca,rp=D
    ref=last_session(sess,y,m-1); pe=last_session(sess,y,m-2); ps=last_session(sess,y-1,m-2); ps9=last_session(sess,y,m-11); lim=ps9
    if None in (ref,pe): return None
    recs=[]
    for t in iv:
        if t in excl or not any(a<=ref<b for a,b in iv[t]): continue
        e=price_on(px,sess,t,pe) if t in px else None
        s=price_on(px,sess,t,ps) if (t in px and ps) else None
        a=rp.get((t,pe)); b=rp.get((t,ps)) if ps else None; pstart=ps
        if s is None and b is None and ps9:
            s=price_on(px,sess,t,ps9) if t in px else None; b=rp.get((t,ps9)); pstart=ps9
        mvp=None
        if a and b:
            F=1.0
            for x,f in ca.get(t,()):
                if pstart<x<=pe: F*=f
            mvp=a/(b*F)-1
        has=e is not None and s is not None and lim is not None and dates[t][0]<=lim
        flag=''
        if has:
            (d0,p0),(d1,p1)=s,e; exs={x:f for x,f in ca.get(t,()) if d0<x<=d1}; F=1.0
            for f in exs.values(): F*=f
            raw=p1/p0-1; mvF=p1/(p0*F)-1; useF=True
            if exs and mvp is not None and abs((1+raw)/(1+mvp)-1)<=0.02<abs((1+mvF)/(1+mvp)-1): useF=False
            mv=mvF if useF else raw
            ds=[d for d in dates[t] if d0<=d<=d1]; r=[]
            for u,v in zip(ds,ds[1:]):
                q=px[t][v]/px[t][u]
                if useF and v in exs: q/=exs[v]
                r.append(q-1)
            sg=st.stdev(r) if len(r)>20 else None
            if r and max(abs(x) for x in r)>0.75:
                sg=None; flag+='jump;'
                if fix=='drop_jump_names': continue
            if mvp is not None and abs((1+mv)/(1+mvp)-1)>0.02: mv=mvp; sg=None; flag+='mismatch;'
        elif mvp is not None: mv=mvp; sg=None; flag+='nodaily;'
        else: continue
        recs.append(dict(t=t,mv=mv,sg=sg,flag=flag))
    mes=[last_session(sess,y-1,m-2+k) for k in range(13)]
    def msig(t):
        p=[rp.get((t,x)) for x in mes]
        if any(v is None for v in p): return None
        r=[]
        for i in range(1,13):
            q=p[i]/p[i-1]
            for x,f in ca.get(t,()):
                if mes[i-1]<x<=mes[i]: q/=f
            r.append(q-1)
        return st.stdev(r)
    ratios=[r['sg']/msig(r['t']) for r in recs if r['sg'] and msig(r['t'])]
    k=st.median(ratios) if ratios else None; med=st.median([r['sg'] for r in recs if r['sg']])
    for r in recs:
        if not r['sg']:
            ms=msig(r['t']); r['sg']=ms*k if (ms and k) else med
    rows=sorted((-(r['mv']/r['sg']),r['t'],r['mv'],r['flag']) for r in recs)
    return dict(ref=ref,exec=third_friday(sess,y,m),top=[(t,-x,mv) for x,t,mv,_ in rows],flags={t:f for _,t,_,f in rows},n=len(rows))
def backtest(D,sels,start,end,K,stop=None):
    px,dates,sess,iv,ca,rp=D; hold={}; peak={}; cash=1.0; pk=0; mdd=0; nstop=0
    execs={s['exec']:s for s in sels if s}; init=max((e for e in execs if e<=start)); days=[d for d in sess if start<=d<=end]
    def rebalance(names,d):
        nonlocal hold,cash,peak
        mv={t:n*px[t][d] for t,n in hold.items()}; E0=cash+sum(mv.values()); Ep=E0
        for _ in range(200):
            T=Ep/K; new=E0-C*(sum(abs(T-mv.get(t,0.0)) for t in names)+sum(v for t,v in mv.items() if t not in names))
            if abs(new-Ep)<1e-14: break
            Ep=new
        peak={t:(max(peak[t],px[t][d]) if t in hold and t in peak else px[t][d]) for t in names}
        hold={t:(Ep/K)/px[t][d] for t in names}; cash=0.0
    def tradable(sel,d): return [t for t,_,_ in sel['top'] if t in px and d in px[t]][:K]
    prev=None
    for d in days:
        for t in list(hold):
            for x,f in ca.get(t,()):
                if prev and prev<x<=d and vendor_unadjusted(D,t,x): hold[t]/=f       # spin / special dividend: value kept
            if d not in px[t]:
                last=[u for u in dates[t] if u<d]
                if d>dates[t][-1]: cash+=hold.pop(t)*px[t][last[-1]]*(1-C); peak.pop(t,None)   # series ended: cashed out at last close
        if d==start: rebalance(tradable(execs[init],d),d)
        elif d in execs and d<end: rebalance(tradable(execs[d],d),d)
        elif stop and d!=end:
            for t in list(hold):
                if d in px[t]:
                    peak[t]=max(peak[t],px[t][d])
                    if px[t][d]<=peak[t]*(1-stop): cash+=hold.pop(t)*px[t][d]*(1-C); peak.pop(t); nstop+=1
        val=cash+sum(n*(px[t][d] if d in px[t] else px[t][[u for u in dates[t] if u<d][-1]]) for t,n in hold.items())
        if d==end: val=cash+sum(n*px[t][d] for t,n in hold.items())*(1-C)
        pk=max(pk,val); mdd=min(mdd,val/pk-1); prev=d
    yrs=(dt.date.fromisoformat(end)-dt.date.fromisoformat(start)).days/365.25
    return val**(1/yrs)-1,mdd,val,nstop
_va={}
def vendor_unadjusted(D,t,x):
    """True when the daily series shows the spin as a real price drop (so the holder must be compensated)."""
    px,dates,sess,iv,ca,rp=D
    if (t,x) not in _va:
        f=[ff for xx,ff in ca[t] if xx==x][0]; b=[u for u in dates[t] if u<x]; a=[u for u in dates[t] if u>=x]
        if not a or not b: _va[(t,x)]=False
        else:
            m0=max(k[1] for k in rp if k[0]==t and k[1]<x) if any(k[0]==t and k[1]<x for k in rp) else None
            m1=min(k[1] for k in rp if k[0]==t and k[1]>=x) if any(k[0]==t and k[1]>=x for k in rp) else None
            ok=True
            if m0 and m1 and m0 in px[t] and m1 in px[t]:
                ok=abs((px[t][m1]/px[t][m0])/(rp[(t,m1)]/(rp[(t,m0)]*f))-1)>0.02     # daily return != F-adjusted pack return -> not vendor-adjusted
            _va[(t,x)]=ok
    return _va[(t,x)]
ref_picks={(r['era'],r['excl_NVDA'],r['reb']):[r['top1'],r['top2'],r['top3']] for r in rd(os.path.join(Q,'out','picks_top3_by_rebalance.csv'))}
score_top=collections.defaultdict(list)
for r in rd(os.path.join(Q,'out','score_tables_full.csv.gz')):
    if int(r['rank'])<=3: score_top[(r['era'],r['reb'])].append(r['ticker'])
summ={(r['window'],r['variant']):r for r in rd(os.path.join(Q,'out','summary.csv'))} if os.path.exists(os.path.join(Q,'out','summary.csv')) else {}
report={('2015','no',1):(13.3917,-74.438),('2015','no',2):(10.9875,-63.3331),('2015','no',3):(15.1325,-46.6448),('2015','yes',1):(-0.6065,-75.6499),('2015','yes',2):(4.7221,-64.3963),('2015','yes',3):(7.2893,-49.6634),
        ('2015','stop',1):(16.2804,-55.2887),('2015','stop',2):(8.6915,-67.602),('2015','stop',3):(11.4142,-58.8494),
        ('2005','no',1):(-10.9531,-87.6753),('2005','no',2):(-2.704,-67.2696),('2005','no',3):(3.2812,-59.6636),('2005','stop',1):(-10.3504,-86.8156),('2005','stop',2):(-3.7409,-67.2852),('2005','stop',3):(2.0739,-53.9542)}
same_dates={1:14.61,2:13.79,3:18.48}     # Queen REPORT: 2015-10-12 -> 2025-01-02, price only (SPMO 15.55, SPY 12.24)
out=[]; lines=[]
for era,(start,end,years,fn,label) in {'2015':('2015-01-02','2025-01-02',range(2014,2025),select_spec,'SPEC-only selection'),'2005':('2005-01-03','2015-01-02',range(2004,2015),select,'selection with Queen\'s code fallbacks')}.items():
    D=load(era)
    for excl,tag in (((),'no'),)+((( 'NVDA',),'yes'),) if era=='2015' else (((),'no'),):
        sels=[fn(era,D,y,m,excl) for y in years for m in (3,9) if not (y==years[0] and m==3)]
        sels=[s for s in sels if s]; same=0; tot=0; diffs=[]
        for s in sels:
            key=(era,s['exec'][:7]); mine=[t for t,_,_ in s['top'][:3]]
            ref=score_top.get(key) if tag=='no' else ref_picks.get((era,'yes',s['exec'][:7]))
            if ref:
                tot+=1; same+=mine==ref
                if mine!=ref: diffs.append((key[1],mine,ref))
        lines.append(f'{era} era ({label}), NVDA excluded={tag}: Top3 identical to Queen at {same} of {tot} rebalances; differences: {diffs}')
        for K in (1,2,3):
            for stp,lab in ((None,tag),)+(((0.30,'stop'),) if tag=='no' else ()):
                c,m,v,ns=backtest(D,sels,start,end,K,stp); q=report[(era,lab,K)]
                out.append(dict(era=era,variant={'no':'no stop','yes':'no NVDA','stop':'30% trailing stop'}[lab],K=K,claude_CAGR_pct=round(c*100,4),queen_CAGR_pct=q[0],claude_MDD_pct=round(m*100,4),queen_MDD_pct=q[1],stops=ns,final_value=round(v,4)))
                lines.append('  %s Top%d %-18s CAGR %8.4f%% (Queen %8.4f%%)  MDD %8.4f%% (Queen %8.4f%%) stops %d'%(era,K,out[-1]['variant'],c*100,q[0],m*100,q[1],ns))
        if era=='2015' and tag=='no':
            for K in (1,2,3):
                c,m,v,ns=backtest(D,sels,'2015-10-12','2025-01-02',K,None)
                lines.append('  same dates as SPMO (2015-10-12 -> 2025-01-02) Top%d: CAGR %.2f%% (Queen %.2f%%), MDD %.1f%%'%(K,c*100,same_dates[K],m*100))
    if era=='2005':
        px=D[0]; leg={t:px[t]['2005-03-18']/px[t]['2005-01-03'] for t in ('BMS','ADSK','MWW','VLO','JWN')}; yrs=3651/365.25
        lines.append('  2005 era first leg 2005-01-03 -> 2005-03-18 from the shared daily closes: '+str({t:'%+.1f%%'%((v-1)*100) for t,v in leg.items()}))
        rep={1:0.3136,2:0.7603,3:1.3809}; old={1:leg['BMS'],2:(leg['BMS']+leg['ADSK'])/2,3:(leg['BMS']+leg['ADSK']+leg['MWW'])/3}; new={1:leg['ADSK'],2:(leg['ADSK']+leg['VLO'])/2,3:(leg['ADSK']+leg['VLO']+leg['JWN'])/3}
        for K in (1,2,3):
            f=rep[K]*new[K]/old[K]; lines.append('  if BMS and MWW are dropped from the Sep-2004 selection (ADSK, VLO, JWN move up): Top%d CAGR %.2f%% -> about %.2f%% (first-leg approximation)'%(K,(rep[K]**(1/yrs)-1)*100,(f**(1/yrs)-1)*100))
w=csv.DictWriter(open(os.path.join(HERE,'claude_vs_queen_spmo_topk.csv'),'w',newline=''),fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print('\n'.join(lines))
