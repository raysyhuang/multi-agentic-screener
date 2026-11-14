"""Claude's fixed rule on the 2005-2015 PROVISIONAL pack engine (pack0515fx/rebuild_2005_2015.py = stdlib copy of the frozen
rebuild_full.py; execution inputs extended for these jobs by build_pack0515fx.py). Same rule/schedule as fixed_rule_1525.py:
rank-average of W10/W11/W12, top N EW, 12 tranches all bought 2005-01-03, tranche off rebalances first session of month off, off+12..,
all liquidated 2015-01-02. Only the schedule header of path() is replaced. mode 'strict' (engine) or 's1' (SENSITIVITY: cash-out/skip
instead of UNKNOWN, every deviation logged) or 's1t' (s1 on pack0515fx_s1: series truncated at a deal/tail end are accepted, then cashed out). usage: fixed_rule_0515.py mode"""
import os,sys,pickle,inspect
MODE=sys.argv[1];PK='/workspace/momentum-followup-20261114/'+('pack0515fx_s1/' if MODE=='s1t' else 'pack0515fx/');os.chdir(PK);sys.path.insert(0,PK);sys.argv=[sys.argv[0]]
import rebuild_2005_2015 as R
NS=(1,2,3,5,10,20,30,50)
src=inspect.getsource(R.path).splitlines()
assert src[1].lstrip().startswith('D0=first_td(2005,ph)') and src[5].lstrip().startswith('rebs.append(d)')
code='\n'.join(['def path_sched(W,H,K,ph,want_nav=False):','    D0,rebs,END=_SCHED']+src[6:])
if MODE in('s1','s1t'):
    REP=[("if e['type']=='unresolved': return fail(None,'corporate_action_unresolved',e['event_id'])","if e['type']=='unresolved': _EV.append(('unresolved_ca_ignored',t+':'+e['event_id'],d))"),
         ("if e['action']=='fail' or (aser,acq,d) not in CL: return fail(d,'corporate_action_unresolved',t+'->'+acq+'(series unverified)')",
          "if e['action']=='fail' or (aser,acq,d) not in CL: cash+=n*last;del hold[t];_EV.append(('cashout_stock_deal_unverified',t,d));break"),
         ("elif d>SLAST.get((ser,t),'9999'): return fail(d,'corporate_action_unresolved',t)","elif d>SLAST.get((ser,t),'9999'): cash+=n*last;val+=n*last;del hold[t];_EV.append(('cashout_series_end',t,d))"),
         ("else: return fail(d,'daily_gap',t)","else: val+=n*last;_EV.append(('stale_price_gap',t,d))"),
         ("if miss: return fail(d,'rank_missing_unbounded',' '.join(miss))","if miss: _EV.append(('rank_missing_ignored',' '.join(miss),d))"),
         ("if a is None: return fail(d,'NOT_IN_PACK',t)","if a is None: _EV.append(('not_in_pack_slot_cash',t,d));continue"),
         ("if a['status']=='not_tradable': x,y=a['detail'].split('@');return fail(y,'pick_not_tradable',x)","if a['status']=='not_tradable': _EV.append(('not_tradable_slot_cash',t,d));continue"),
         ("if a['status']!='ok': return fail(d,'exec_price_missing',t)","if a['status']!='ok': _EV.append(('exec_missing_slot_cash',t,d));continue"),
         ("T={t:Ep/K for t in keys}","T={t:Ep/K for t in sers}"),
         ("new={k:(T[k]/CL[(sers[k],k,d)],sers[k],CL[(sers[k],k,d)]) for k in keys}","new={k:(T[k]/CL[(sers[k],k,d)],sers[k],CL[(sers[k],k,d)]) for k in sers}"),
         ("return fail(d,'pick_not_tradable',p)","_EV.append(('k1_postcheck_ignored',p,d))")]
    for a,b in REP:
        assert code.count(a)==1,a
        code=code.replace(a,b)
g=R.__dict__;g['_EV']=[];exec(compile(code,'path_sched','exec'),g)
PACK={k for k,v in R.ELIG.items() if v};orig=R.rank
def rank_avg(S,W):
    assert W==1012
    if S not in PACK: return ([],['__SIGNAL_OUTSIDE_PACK__'],set())
    rr={w:orig(S,w) for w in (10,11,12)};pos={}
    for w,(r,_,_) in rr.items():
        for j,(_,t) in enumerate(r): pos.setdefault(t,[]).append(j)
    out=sorted((sum(v)/3,t) for t,v in pos.items() if len(v)==3)
    return [(-x,t) for x,t in out],sorted(set().union(*[set(m) for _,m,_ in rr.values()])),set().union(*[i for _,_,i in rr.values()])
g['rank']=rank_avg
D0=R.first_td(2005,1);END=R.first_td(2015,1)
P=[];NAV={};RB=[]
for N in NS:
    for off in range(12):
        rebs=[D0]+[d for d in (R.first_td(2005+m//12,m%12+1) for m in range(off if off else 12,120,12)) if d<END]
        g['_SCHED']=(D0,rebs,END);g['_EV'].clear()
        o=g['path_sched'](1012,12,N,off+1,want_nav=True);ev=list(g['_EV']);tid=f'0515_{MODE}_N{N}_T{off}'
        st=o['status']
        if MODE in('s1','s1t') and st!='UNKNOWN': st='SENS_CASHOUT' if ev else 'SENS_CLEAN'
        for S,d,p in o['rebalances']: RB.append(dict(tranche=tid,signal_date=S,trade_date=d,picks=p))
        P.append(dict(tranche=tid,mode=MODE,N=N,off=off,entry=D0,end=END,n_rebal=len(rebs),status=st,fail_reason=o.get('fail_reason'),fail_ticker=o.get('fail_ticker'),fail_date=o.get('fail_date'),
                      final_multiple=o.get('final'),CAGR=o.get('CAGR'),daily_MDD=o.get('MDD'),events=';'.join(f'{a}|{b}|{c}' for a,b,c in ev[:60])))
        if o.get('nav'): NAV[tid]=o['nav']
        print(tid,st,o.get('CAGR'),o.get('fail_reason'),o.get('fail_ticker'),len(ev),flush=True)
days=R.SES[R.SI[D0]:R.SI[END]+1]
pickle.dump(dict(P=P,NAV=NAV,RB=RB,days=days),open(f'/workspace/momentum-followup-20261114/scratch/fixed0515_{MODE}.pkl','wb'));print('saved')
