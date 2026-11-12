"""Research overlay: 30% trailing stop (post-entry peak, daily close) on K=1 momentum paths, variants A/B.
Own simulation loop (written for this study). The frozen v17-strict4 engine + v18 KORS/CPRI dedupe is loaded READ-ONLY
and used only as a data oracle: rank tables (rank_strict), execution series lookup (series_strict), held corporate
actions (held_ca / verify_post), deal dates (_dealdate). No engine file is modified; nothing is written into frozen folders.
Usage: python3 overlay.py  -> writes results under this folder."""
import sys,os,json,pickle,datetime as dt
import numpy as np,pandas as pd
OUT='/workspace/w10-12-h2-4-stop-ab-20261112'
sys.path.insert(0,'/workspace/momentum432-answer-key-full-20261109/build')
from load_v17 import load
ns=load()
_rs=ns['rank_strict']
def rank_v18(S,W):
    r,miss,nu=_rs(S,W);tk={t:g for g,t in r}
    if 'CPRI' in tk and 'KORS' in tk:
        assert tk['CPRI']==tk['KORS'];r=[(g,t) for g,t in r if t!='KORS'];nu-=1
    return r,miss,nu
ns['rank_strict']=rank_v18;ns['RS'].clear();ns['RCAS'].clear()   # same in-memory wrapper as the answer-key build (run_part.py)
SD=ns['SD'];SDI=ns['SDI'];MEL=ns['MEL'];c=ns['c'];first_td=ns['first_td'];prev_td=ns['prev_td']
series_strict=ns['series_strict'];held_ca=ns['held_ca'];verify_post=ns['verify_post'];dealdate=ns['_dealdate']
ek16=ns['_execution_key16'];TICKER_REUSE=ns['TICKER_REUSE'];GR=ns['GR'];RCAS=ns['RCAS'];NotTradable=ns['_NotTradable'];Unres=ns['_Unres']
RK={}
def RANK(S,W):
    if (S,W) not in RK: RK[(S,W)]=rank_v18(S,W)
    return RK[(S,W)]
STOP=0.30
class Fail(Exception): pass
def last_me_on_or_before(d): return [m for m in MEL if m<=d][-1]

def run(W,H,ph,variant=None,K=1,lag=0):
    """variant None = no stop (baseline); 'A' = replace with W12 Top1 at latest ME<=trigger; 'B' = same-W Top1."""
    assert K==1
    D0=first_td(2015,ph);END=first_td(2025,ph);rebs=[];j=0
    while True:
        mm=(ph-1)+j*H;d=first_td(2015+mm//12,mm%12+1)
        if d>=END: break
        rebs.append(d);j+=1
    nxt=dict(zip(rebs,rebs[1:]+[END]))
    hold={}   # key -> [shares, series_name, series, last_close]
    cash=1.0;eq=[];dprev=D0;ids=set();log=[];stops=[];peak=None;peak_date=None;nxt_reb=None;pending=None;superseded=[];cur_kind=None;cur_since=None
    nreb=0;fees=0.0
    def trade_to(d,E0,h,newkey,newser):
        """K=1 re-target at close d (same fee fixed point as the engine). newkey None = all cash."""
        Ep=E0
        for _ in range(200):
            T={newkey:Ep} if newkey else {}
            turn=sum(abs(T.get(t,0)-h.get(t,0)) for t in set(T)|set(h));En=E0-c*turn
            if abs(En-Ep)<=1e-12*E0: Ep=En;break
            Ep=En
        return Ep,turn
    days=SD[SDI[D0]:SDI[END]+1]
    try:
      for d in days:
        # ---- corporate actions on holdings (same rules as engine path loop)
        for t,(n,nm,s,last,*_) in list(hold.items()):
            try: cas=held_ca(t,dprev,d)
            except Unres as e: raise Fail(('corporate_action_unresolved',t,str(e),d))
            for ca in cas:
                cs=float(ca['cash_per_share'])
                if ca['type'] in('cash','stock') and s.index[-1]>=ca['ex_date'] and ca['id'] not in TICKER_REUSE: continue
                vb=n*last
                if ca['type'] in('spin','special_div'):
                    g=n*float(ca['ratio'])*float(ca['child_first_close'] or 0);v=n*cs+g-c*g;cash+=v;ids.add(ca['id'])
                    if peak is not None and d in s.index: peak*= float(s[d])/(float(s[d])+(v/n))   # peak re-expressed ex-distribution
                    log.append(dict(date=d,event='ca_'+ca['type'],ticker=t,ca=ca['id']))
                elif ca['type']=='cash':
                    cash+=n*cs;del hold[t];ids.add(ca['id']);peak=None;log.append(dict(date=d,event='cash_deal',ticker=t,ca=ca['id']))
                elif ca['type']=='stock':
                    acq=ca['child_or_acquirer'];i=[k for k,m in enumerate(MEL) if m>=d][0];vr=verify_post(acq,d,MEL[i:i+4])
                    if vr is None or d not in vr[1].index: raise Fail(('corporate_action_unresolved',t,acq,d))
                    s2=vr[1];ev=0.0;eqv=0.0
                    if acq in hold and acq!=t:
                        ne,_,se,*_=hold[acq]
                        if d not in se.index: raise Fail(('daily_gap',acq,'',d))
                        ev=ne*float(se[d]);eqv=ev/float(s2[d])
                    rs=n*float(ca['ratio']);cash+=n*cs;del hold[t];hold[acq]=[eqv+rs,"acq:"+acq,s2,float(s2[d]),d];ids.add(ca['id'])
                    if peak is not None and d in s.index:  # peak converted into acquirer units by value equivalence
                        peak=peak*(rs*float(s2[d]))/(n*float(s[d])) if float(s[d])>0 else float(s2[d])
                    log.append(dict(date=d,event='stock_deal',ticker=t,to=acq,ca=ca['id']))
        val=cash
        for t,v in list(hold.items()):
            n,nm,s,last=v[:4]
            if d in s.index: v[3]=float(s[d]);val+=n*v[3]
            elif d>s.index[-1]: raise Fail(('corporate_action_unresolved',t,'series_end',d))
            else: raise Fail(('daily_gap',t,'',d))
        sched=(d in nxt or d==D0)
        # ---- stop check (not on scheduled rebalance days, not on END)
        if variant and (sched or d==END) and pending is not None:
            superseded.append(dict(pending,superseded_on=d));pending=None
        if variant and not sched and d!=END and hold:
            (t,v),=hold.items();px=v[3];trig=None
            if pending is not None: trig=pending;pending=None
            else:
                if peak is None or px>peak: peak=px;peak_date=d
                if px<=(1-STOP)*peak:
                    info=dict(date=d,stopped=t,stop_close=px,peak=peak,peak_date=peak_date,entry_date=v[4] if len(v)>4 else None,dd=px/peak-1,held_kind=cur_kind,since=cur_since)
                    if lag: pending=info
                    else: trig=info
            if trig is not None and trig['stopped']==t:
                M=last_me_on_or_before(trig['date']);Wr=12 if variant=='A' else W
                r,miss,nu=RANK(M,Wr);cands=[x for _,x in r[:2]]
                base=dict(trig,exec_date=d,exec_close=px,signal=M,W_rank=Wr,top1=cands[0] if cands else None,top2=cands[1] if len(cands)>1 else None)
                rep=None;why=''
                if miss: why='rank_table_UNKNOWN:'+' '.join(miss)
                elif not cands: why='empty_table'
                else:
                    rep=cands[0] if cands[0]!=t and ek16(cands[0],d)!=t else (cands[1] if len(cands)>1 else None)
                    if rep is None: why='no_top2'
                ser=None
                if rep:
                    if any(M<e<=d for e in dealdate(rep)): why='replacement_deal_date_in_(M,d]';rep=None
                    else:
                        try: ser=series_strict(rep,d,nxt_reb)
                        except NotTradable as e: why='replacement_not_tradable:'+str(e);rep=None
                        if rep and ser is None: why='replacement_no_verified_series';rep=None
                E0=val;h={t:v[0]*v[3]};key=ek16(rep,d) if rep else None
                Ep,turn=trade_to(d,E0,h,key,None);fees+=E0-Ep
                if rep:
                    hold={key:[Ep/float(ser[1][d]),ser[0],ser[1],float(ser[1][d]),d]};cash=0.0;peak=float(ser[1][d]);peak_date=d;ids|=RCAS.get((M,Wr),set())
                else: hold={};cash=Ep;peak=None
                val=Ep;cur_kind='stop_replacement' if rep else 'cash';cur_since=d
                stops.append(dict(base,replacement=rep,replacement_series=ser[0] if rep else None,cash_reason=why,E_before=E0,E_after=Ep,next_rebalance=nxt_reb))
        if sched:
            S=prev_td(d);r,miss,nu=RANK(S,W)
            if miss: raise Fail(('rank_missing_unbounded',' '.join(miss),'',d))
            t=r[0][1];key=ek16(t,d)
            if key in hold: sr=(hold[key][1],hold[key][2])
            else:
                if any(S<e<=d for e in dealdate(t)): raise Fail(('pick_not_tradable',t,'',d))
                try: sr=series_strict(t,d,nxt[d])
                except NotTradable as e: raise Fail(('pick_not_tradable',t,str(e),d))
                if sr is None: raise Fail(('exec_price_missing',t,'',d))
                sr=(sr[0],sr[1])
            E0=val;h={k:v[0]*v[3] for k,v in hold.items()}
            Ep,turn=trade_to(d,E0,h,key,None);fees+=E0-Ep
            if key in hold:
                hold={key:hold[key]};hold[key][0]=Ep/hold[key][3]   # no trade if same name (turnover 0); peak continues
                if variant and hold[key][3]>0 and peak is None: peak=hold[key][3]
            else:
                p0=float(sr[1][d]);hold={key:[Ep/p0,sr[0],sr[1],p0,d]};peak=p0;peak_date=d
            cash=0.0   # engine: cash=Ep-sum(T)=0 for K=1
            val=Ep;nreb+=1;ids|=RCAS.get((S,W),set());cur_kind='scheduled';cur_since=d;nxt_reb=nxt[d]
            log.append(dict(date=d,event='rebalance',signal=S,pick=t,series=sr[0],E_before=E0,E_after=Ep))
        if d==END:
            g=sum(v[0]*v[3] for v in hold.values());val=cash+g-c*g;fees+=c*g
        eq.append(val);dprev=d
    except Fail as e:
        fr,ft,fx,fd=e.args[0]
        return dict(W=W,H=H,K=K,phase=ph,variant=variant or 'none',entry=D0,end=END,status='UNKNOWN',fail_reason=fr,fail_ticker=ft,fail_detail=fx,fail_date=fd,stops=stops,log=log,nav=None)
    # engine R2 post-check (K=1): scheduled picks with a deal date in (S,d]
    for x in log:
        if x['event']=='rebalance' and any(x['signal']<e<=x['date'] for e in dealdate(x['pick'])):
            return dict(W=W,H=H,K=K,phase=ph,variant=variant or 'none',entry=D0,end=END,status='UNKNOWN',fail_reason='pick_not_tradable',fail_ticker=x['pick'],fail_detail='',fail_date=x['date'],stops=stops,log=log,nav=None)
    eq=np.array(eq);yrs=(pd.Timestamp(END)-pd.Timestamp(D0)).days/365.25
    st='DATA_COMPLETE_PROVISIONAL_C' if any(GR.get(i)=='C' for i in ids) else 'DATA_COMPLETE_CONDITIONAL'
    return dict(W=W,H=H,K=K,phase=ph,variant=variant or 'none',entry=D0,end=END,status=st,final=float(eq[-1]),CAGR=float(eq[-1]**(1/yrs)-1),
                MDD=float(np.min(eq/np.maximum.accumulate(eq)-1)),n_rebalances=nreb,superseded=superseded,lag=lag,fees=fees,stops=stops,log=log,nav=eq,days=days)

if __name__=='__main__':
    jobs=[(W,H,ph) for W in (10,11,12) for H in (2,3,4) for ph in range(1,13)]+[(12,12,1)]
    res={}
    for W,H,ph in jobs:
        for v in (None,'A','B'):
            res[(W,H,ph,v or 'none')]=run(W,H,ph,v)
        for v in ('A','B'):
            res[(W,H,ph,v+'_next')]=run(W,H,ph,v,lag=1)
        print(W,H,ph,[ (round(res[(W,H,ph,v)].get('CAGR') or -9,4),len(res[(W,H,ph,v)]['stops'])) for v in ('none','A','B')],flush=True)
    pickle.dump(res,open(OUT+'/work/results.pkl','wb'))
