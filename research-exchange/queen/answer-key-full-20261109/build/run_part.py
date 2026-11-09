"""Run v17 strict4 + v18 dedupe for a set of W values, all H, K1-3, 12 phases; capture everything needed for the pack (A)
and the answer key (B). Engine read-only. Output: /workspace/m432full-scratch/part_W{..}.pkl"""
import sys,pickle,collections,datetime as dt,numpy as np,pandas as pd
sys.path.insert(0,'/workspace/momentum432-answer-key-full-20261109/build')
from load_v17 import load
Ws=[int(x) for x in sys.argv[1].split(',')]
ns=load()
_rs=ns['rank_strict']
def rank_strict(S,W):
    r,miss,nu=_rs(S,W);tk={t:g for g,t in r}
    if 'CPRI' in tk and 'KORS' in tk:
        assert tk['CPRI']==tk['KORS'];r=[(g,t) for g,t in r if t!='KORS'];nu-=1
    return r,miss,nu
ns['rank_strict']=rank_strict;ns['RS'].clear();ns['RCAS'].clear()
SD=ns['SD'];c=ns['c']
# ---- execution availability log (every purchase-time series lookup)
EXA={};_ss=ns['series_strict']
def series_strict(t,d0,d1):
    try: r=_ss(t,d0,d1)
    except ns['_NotTradable'] as e:
        EXA[(d0,t,d1)]=('not_tradable',str(e),None);raise
    EXA[(d0,t,d1)]=('ok',vname(r[0],r[1]),r[1]) if r is not None else ('no_verified_series','',None)
    return r
ns['series_strict']=series_strict
import hashlib
_VN={}
def vname(nm,s):
    if '+vendor_deadj' not in nm: return nm
    k=id(s)
    if k not in _VN: _VN[k]=(nm+'#'+hashlib.md5(np.ascontiguousarray(s.values,dtype=float).tobytes()+'|'.join(s.index).encode()).hexdigest()[:8],s)
    return _VN[k][0]
SER={}      # (nm,key) -> series object (first seen); conflicts checked
CONF=[];SPELLS=collections.defaultdict(list)  # (nm,key)->[(start,end)]
PATHS=[];REB=[];TRD=[];NAV={};EVU=[]
for W in Ws:
  for H in range(1,13):
    for K in (1,2,3):
      for ph in range(1,13):
        state={'prev':{}, 'open':{}}
        snaps=[]
        def hook(w,h,k,p,d,val,cash,hold,terminal):
            prev=state['prev'];cur={t:(n,nm,s,last) for t,(n,nm,s,last) in hold.items()}
            for t,(n,nm,s,last) in {**prev,**cur}.items():
                nm=vname(nm,s);key=(nm,t)
                if key in SER and SER[key] is not s:
                    # different object: compare values on overlapping dates later
                    SER.setdefault(('ALT',)+key,[]).append(s)
                else: SER[key]=s
                op=state['open']
                if key not in op: op[key]=d
            # close spells for names no longer held after today
            for key in list(state['open']):
                if key[1] not in cur or vname(cur[key[1]][1],cur[key[1]][2])!=key[0]:
                    SPELLS[key].append((state['open'].pop(key),d))
            state['prev']=cur
            snaps.append((d,float(val),float(cash),{t:(float(v[0]),float(v[3]),vname(v[1],v[2])) for t,v in hold.items()},terminal))
        ns['_snapshot_hook']=hook
        row,eq,led=ns['path_strict4'](W,H,K,ph)
        ns.pop('_snapshot_hook')
        for key,a in state['open'].items():
            e=snaps[-1][0] if snaps else a
            if row['status']=='UNKNOWN' and e in SD and SD.index(e)+1<len(SD): e=SD[SD.index(e)+1]   # abort day: marked before the failing step
            SPELLS[key].append((a,e))
        pid=f'W{W}H{H}K{K}P{ph}'
        nreb=0;ntr=0
        byd={s[0]:i for i,s in enumerate(snaps)}
        for x in led:
            if x['event']=='rebalance':
                d=x['date'];i=byd.get(d)
                post=snaps[i];pre=snaps[i-1] if i and i>0 else (None,1.0,1.0,{},False)
                # pre-trade holdings marked at d: shares before trade x close at d
                keys=x['execution_keys'].split('|');picks=x['picks'].split('|')
                REB.append(dict(path_id=pid,W=W,H=H,K=K,phase=ph,signal_date=x['signal'],trade_date=d,picks=x['picks'],execution_keys=x['execution_keys'],price_series=x['src'],
                    equity_before=x['E_before'],equity_after=x['E_after'],fee=x['fee'],turnover=x['turnover'],weights='|'.join(['%.10g'%(1/K)]*K)))
                prevh=dict(pre[3])
                # corporate-action legs processed earlier on the same day (before the rebalance) change pre-trade holdings
                for y in led:
                    if y is x: break
                    if y.get('date')!=d: continue
                    if y['event']=='cash_deal': prevh.pop(y['ticker'],None)
                    elif y['event']=='stock_deal':
                        prevh.pop(y['ticker'],None);prevh[y['to']]=(float(y['total_successor_shares']),None,'acq:'+y['to'])
                for t in dict.fromkeys(list(prevh)+list(post[3])):
                    n0=prevh.get(t,(0.0,None,None))[0];n1=post[3].get(t,(0.0,None,None))[0]
                    px=post[3][t][1] if t in post[3] else None
                    if px is None:   # sold: price = its series close at d
                        nm=prevh[t][2];s=SER.get((nm,t));px=float(s[d]) if s is not None and d in s.index else None
                    act='buy' if t not in prevh else ('sell' if t not in post[3] else ('hold' if abs(n1-n0)<=1e-12*max(n0,1e-30) else ('rebalance_buy' if n1>n0 else 'rebalance_sell')))
                    dv=(n1-n0)*px if px is not None else None
                    TRD.append(dict(path_id=pid,trade_date=d,event='rebalance',execution_key=t,action=act,price=px,shares_before=n0,shares_after=n1,trade_value=dv,
                        fee=(c*abs(dv) if act!='hold' and dv is not None else 0.0),weight_before=(n0*px/x['E_before'] if px and n0 else 0.0),weight_after=(n1*px/x['E_after'] if n1 else 0.0)))
                    ntr+=act!='hold'
                nreb+=1
            elif x['event']=='terminal_liquidation':
                for t,(n,px,nm) in snaps[-1][3].items():
                    TRD.append(dict(path_id=pid,trade_date=x['date'],event='terminal_liquidation',execution_key=t,action='sell',price=px,shares_before=n,shares_after=0.0,trade_value=-n*px,fee=c*n*px,weight_before=None,weight_after=0.0));ntr+=1
            else:
                TRD.append(dict(path_id=pid,trade_date=x['date'],event=x['event'],execution_key=x['ticker'],action='corporate_action',ca_id=x.get('ca'),cash=x.get('cash'),fee=x.get('fee',0.0),
                    to=x.get('to'),received_shares=x.get('shares')))
                EVU.append((x['ticker'],x['ca'],x['date']))
        PATHS.append(dict(path_id=pid,W=W,H=H,K=K,phase=ph,entry=row['entry'],end=row['end'],status=row['status'],fail_reason=row.get('fail_reason'),fail_ticker=row.get('fail_ticker'),
            fail_date=row.get('fail_date'),final_multiple=row['wealth'],CAGR=row['CAGR'],daily_MDD=row['daily_MDD'],n_rebalances_executed=nreb,n_trades=ntr,grade_C_events=row.get('grade_C_events'),n_events=row.get('n_events')))
        if eq is not None:
            D0=row['entry'];END=row['end'];days=SD[SD.index(D0):SD.index(END)+1];assert len(days)==len(eq);NAV[pid]=np.asarray(eq,dtype=float)
  print('W',W,'done',flush=True)
# closes per (nm,key) over spells
CL=[]
for key,sp in SPELLS.items():
    s=SER[key]
    for a,b in sp:
        z=s[(s.index>=a)&(s.index<=b)]
        CL.append(pd.DataFrame({'price_series':key[0],'execution_key':key[1],'date':z.index,'close':z.values.astype(float)}))
    for alt in SER.get(('ALT',)+key,[]):
        ix=s.index.intersection(alt.index)
        if len(ix) and float(np.max(np.abs(alt.loc[ix].values/s.loc[ix].values-1)))>1e-12: CONF.append(key)
SERLAST={key:str(SER[key].index[-1]) for key in SPELLS}
SERFIRST={key:str(SER[key].index[0]) for key in SPELLS}
pickle.dump(dict(PATHS=PATHS,REB=REB,TRD=TRD,NAV=NAV,CL=pd.concat(CL) if CL else None,EXA={k:v[:2] for k,v in EXA.items()},
    EXAC={k:(float(v[2][k[0]]) if v[2] is not None and k[0] in v[2].index else None) for k,v in EXA.items()},
    CONF=CONF,SERLAST=SERLAST,SERFIRST=SERFIRST,SPELLKEYS=list(SPELLS),EVU=EVU,RCAS={k:sorted(v) for k,v in ns['RCAS'].items()}),
    open(f'/workspace/m432full-scratch/part_{sys.argv[1].replace(",","_")}.pkl','wb'))
print('saved',len(PATHS),'conflicts',len(CONF),flush=True)
