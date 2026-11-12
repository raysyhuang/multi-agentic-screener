# Third independent executor, written from SPEC.md sections 6-8 (not from rebuild_full.py or GPT's code).
import csv,gzip,sys,pickle,collections,os,datetime as dt
HERE=os.path.dirname(os.path.abspath(__file__))
P=sys.argv[1]; C=0.0025
def rd(f): return list(csv.DictReader(gzip.open(P+'/'+f,'rt') if f.endswith('.gz') else open(P+'/'+f)))
sess=[r['session'] for r in rd('trading_sessions.csv')]; si={d:i for i,d in enumerate(sess)}
first={}
for d in sess: first.setdefault(d[:7],d)
rank=pickle.load(open(os.environ.get('RANK_PKL',HERE+'/rank3.pkl'),'rb'))   # set RANK_PKL=rank_corrected.pkl to run on corrected data
avail={(r['trade_date'],r['engine_ticker'],r['window_end']):r for r in rd('execution_availability.csv.gz')}
close={(r['price_series'],r['execution_key'],r['date']):float(r['close']) for r in rd('execution_daily_closes.csv.gz')}
lastd={(r['price_series'],r['execution_key']):r['series_last_date'] for r in rd('series_info.csv')}
hev=collections.defaultdict(list)
for r in rd('holding_events.csv'): hev[(r['price_series'],r['execution_key'])].append(r)
deal=collections.defaultdict(list)
for r in rd('untradable_deal_dates.csv'): deal[r['engine_ticker']].append(r['deal_date'])
def ekey(t,d):
    if t=='AGN': return 'AGN_LEGACY' if d<'2015-03-17' else 'AGN_PLC'
    return 'AGN_PLC' if t=='ACT' else t
def fl(x): return float(x) if x not in('',None) else 0.0
BASE_YEAR=int(os.environ.get('BASE_YEAR','2015'))   # first year of the 10-year window (2005 for Queen's cross-period pack)
def ym(m): return '%04d-%02d'%(BASE_YEAR+m//12,m%12+1)
class Fail(Exception): pass
def run(W,H,K,ph):
    END=first[ym(120+ph-1)]; rebs=[]; m=ph-1
    while m<120+ph-1: rebs.append(first[ym(m)]); m+=H
    rset={d:(rebs[i+1] if i+1<len(rebs) else END) for i,d in enumerate(rebs)}
    D0=rebs[0]; hold={}  # key -> [series, shares]
    cash=1.0; peak=0.0; mdd=0.0; picks_log=[]
    for i in range(si[D0],si[END]+1):
        d=sess[i]; prev=sess[i-1]
        for key in list(hold):
            if key not in hold: continue
            s,n=hold[key]
            evs=[e for e in hev.get((s,key),()) if prev<e['ex_date']<=d]
            if any(e['type']=='unresolved' for e in evs): raise Fail('corporate_action_unresolved')
            for e in evs:
                a=e['action']
                if a=='fail': raise Fail('corporate_action_unresolved')
                if a!='apply': continue
                t=e['type']
                if t=='spin': cash+=n*fl(e['ratio'])*fl(e['child_first_close'])*(1-C)+n*fl(e['cash_per_share'])
                elif t=='special': cash+=n*fl(e['cash_per_share'])
                elif t=='cash': cash+=n*fl(e['cash_per_share']); del hold[key]; break
                elif t=='stock':
                    cash+=n*fl(e['cash_per_share']); ns=e['acquirer_series']; nk=e['child_or_acquirer']; add=n*fl(e['ratio'])
                    del hold[key]
                    if nk in hold:
                        os_,on=hold[nk]
                        if os_!=ns: on=on*close[(os_,nk,d)]/close[(ns,nk,d)]
                        add+=on
                    hold[nk]=[ns,add]; break
                else: raise Exception('type '+t)
        mv={}
        for key,(s,n) in hold.items():
            p=close.get((s,key,d))
            if p is None: raise Fail('corporate_action_unresolved' if d>lastd.get((s,key),'9999') else 'daily_gap')
            mv[key]=n*p
        if d in rset:
            S=prev; r=rank[(S,W)]
            if r['bad']: raise Fail('rank_missing_unbounded')
            names=[t for t,_ in r['top'][:K]]; keys=[ekey(t,d) for t in names]
            if len(set(keys))<K: raise Fail('duplicate_economic_execution_key')
            picks_log.append((S,d,names)); ser={}
            for t,k in zip(names,keys):
                if k in hold: ser[k]=hold[k][0]; continue
                a=avail.get((d,t,rset[d]))
                if a is None or a['status']=='no_verified_series': raise Fail('exec_price_missing')
                if a['status']=='not_tradable': raise Fail('pick_not_tradable')
                ser[k]=a['price_series']
            E0=cash+sum(mv.values()); Ep=E0
            for _ in range(200):
                T=Ep/K
                turn=sum(abs(T-mv.get(k,0.0)) for k in keys)+sum(v for k,v in mv.items() if k not in keys)
                new=E0-C*turn
                if abs(new-Ep)<=1e-13*E0: Ep=new; break
                Ep=new
            hold={k:[ser[k],(Ep/K)/close[(ser[k],k,d)]] for k in keys}; cash=0.0; eq=Ep
        else: eq=cash+sum(mv.values())
        if d==END: eq=cash+sum(mv.values())*(1-C)
        peak=max(peak,eq); mdd=min(mdd,eq/peak-1)
    if K==1:
        for S,d,names in picks_log:
            if any(S<x<=d for x in deal.get(names[0],())): raise Fail('pick_not_tradable')
    yrs=(dt.date.fromisoformat(END)-dt.date.fromisoformat(D0)).days/365.25
    return eq,eq**(1/yrs)-1,mdd
ref={r['path_id']:r for r in csv.DictReader(open(sys.argv[2]+'/paths_summary.csv'))}
c=collections.Counter(); mx=collections.Counter(); bad=[]; rows=[]
for W in range(1,13):
  for H in range(1,13):
    for K in (1,2,3):
      for ph in range(1,13):
        pid=f'W{W}H{H}K{K}P{ph}'; q=ref[pid]
        try:
            f,cg,md=run(W,H,K,ph)
            rows.append([pid,W,H,K,ph,'KNOWN','',repr(f),repr(cg),repr(md),q['status'],q['final_multiple'],q['CAGR'],q['daily_MDD']])
            if q['status']=='UNKNOWN': c['I_known_Q_unknown']+=1; bad.append((pid,q['fail_reason'])); continue
            c['both_known']+=1
            for nm,a,b in(('final_rel',f,float(q['final_multiple'])),('CAGR',cg,float(q['CAGR'])),('MDD',md,float(q['daily_MDD']))):
                dlt=abs(a-b)/(abs(b) if nm=='final_rel' else 1); mx[nm]=max(mx[nm],dlt)
                if dlt>1e-9: c['num_mismatch_'+nm]+=1; bad.append((pid,nm,a,b))
        except Fail as e:
            rows.append([pid,W,H,K,ph,'UNKNOWN',str(e),'','','',q['status'],q['final_multiple'],q['CAGR'],q['daily_MDD']])
            if q['status']!='UNKNOWN': c['I_unknown_Q_known']+=1; bad.append((pid,str(e)))
            else: c['both_unknown']+=1; c['reason_'+('same' if str(e)==q['fail_reason'] else 'DIFF:'+str(e)+'/'+q['fail_reason'])]+=1
print(dict(c)); print(dict(mx)); print(bad[:15], len(bad))
if os.environ.get('OUT_CSV'):
    w=csv.writer(open(os.environ['OUT_CSV'],'w',newline='')); w.writerow(['path_id','W','H','K','phase','status','fail_reason','final_multiple','CAGR','daily_MDD','queen_status','queen_final_multiple','queen_CAGR','queen_daily_MDD']); w.writerows(rows)
