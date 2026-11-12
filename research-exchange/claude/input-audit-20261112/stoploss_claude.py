# My own 30% fixed-stop implementation on top of my from-spec executor (not GPT's code).
import csv,gzip,sys,pickle,collections,os,datetime as dt,statistics as st
HERE=os.path.dirname(os.path.abspath(__file__))
P=sys.argv[1]; C=0.0025
def rd(f): return list(csv.DictReader(gzip.open(P+'/'+f,'rt') if f.endswith('.gz') else open(P+'/'+f)))
sess=[r['session'] for r in rd('trading_sessions.csv')]; si={d:i for i,d in enumerate(sess)}
first={}
for d in sess: first.setdefault(d[:7],d)
rank=pickle.load(open(os.environ.get('RANK_PKL',HERE+'/rank3.pkl'),'rb'))   # set RANK_PKL=rank_corrected.pkl to run on corrected data
avail={}; availday=collections.defaultdict(list)
for r in rd('execution_availability.csv.gz'):
    avail[(r['trade_date'],r['engine_ticker'],r['window_end'])]=r; availday[(r['trade_date'],r['engine_ticker'])].append(r)
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
def ym(m): return '%04d-%02d'%(2015+m//12,m%12+1)
class Fail(Exception): pass
def run(W,H,K,ph,stop=None,trail=False):
    END=first[ym(120+ph-1)]; rebs=[]; m=ph-1
    while m<120+ph-1: rebs.append(first[ym(m)]); m+=H
    rset={d:(rebs[i+1] if i+1<len(rebs) else END) for i,d in enumerate(rebs)}
    D0=rebs[0]; hold={}; basis={}
    cash=1.0; peak=0.0; mdd=0.0; picks_log=[]; stops=[]; fb=0
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
                t=e['type']; cps=fl(e['cash_per_share']); ratio=fl(e['ratio'])
                if t in('spin','special','special_div'):
                    cv=ratio*fl(e['child_first_close']); cash+=n*cv*(1-C)+n*cps; basis[key]=max(0.0,basis[key]-cps-cv)
                elif t=='cash': cash+=n*cps; del hold[key]; basis.pop(key); break
                elif t=='stock':
                    cash+=n*cps; ns=e['acquirer_series']; nk=e['child_or_acquirer']; add=n*ratio
                    ob=basis.pop(key); del hold[key]; tb=max(0.0,(ob-cps)/ratio); eq0=0.0; eb=0.0
                    if nk in hold:
                        os_,on=hold[nk]; eq0=on*close[(os_,nk,d)]/close[(ns,nk,d)]; eb=basis[nk]
                    tot=eq0+add; basis[nk]=(eq0*eb+add*tb)/tot; hold[nk]=[ns,tot]; break
        mv={}; px={}
        for key,(s,n) in hold.items():
            p=close.get((s,key,d))
            if p is None: raise Fail('corporate_action_unresolved' if d>lastd.get((s,key),'9999') else 'daily_gap')
            mv[key]=n*p; px[key]=p
        if d in rset:
            S=prev; r=rank[(S,W)]
            if r['bad']: raise Fail('rank_missing_unbounded')
            names=[t for t,_ in r['top'][:K]]; keys=[ekey(t,d) for t in names]
            if len(set(keys))<K: raise Fail('duplicate_economic_execution_key')
            picks_log.append((S,d,names)); ser={}
            for t,k in zip(names,keys):
                if k in hold: ser[k]=hold[k][0]; continue
                a=avail.get((d,t,rset[d]))
                if a is None:
                    peers=availday.get((d,t),[]); ss={x['price_series'] for x in peers if x['status']=='ok'}
                    if peers and all(x['status']=='ok' for x in peers) and len(ss)==1:
                        s0=next(iter(ss))
                        if all((s0,k,sess[j]) in close for j in range(i,si[rset[d]]+1)): a={'status':'ok','price_series':s0}; fb+=1
                if a is None: raise Fail('NOT_IN_PACK '+t+' '+d)
                if a['status']=='not_tradable': raise Fail('pick_not_tradable')
                ser[k]=a['price_series']
            E0=cash+sum(mv.values()); Ep=E0
            for _ in range(200):
                T=Ep/K
                new=E0-C*(sum(abs(T-mv.get(k,0.0)) for k in keys)+sum(v for k,v in mv.items() if k not in keys))
                if abs(new-Ep)<=1e-13*E0: Ep=new; break
                Ep=new
            nh={}; nb={}
            for k in keys:
                p=close[(ser[k],k,d)]; nq=(Ep/K)/p; oq=hold[k][1] if k in hold else 0.0; ob=basis.get(k,p)
                nb[k]=(oq*ob+(nq-oq)*p)/nq if nq>oq else ob
                if trail: nb[k]=max(nb[k],p) if k in hold else p
                nh[k]=[ser[k],nq]
            hold=nh; basis=nb; cash=0.0; eq=Ep
        else:
            if stop is not None:
                for key in list(hold):
                    if trail: basis[key]=max(basis[key],px[key])
                    if basis[key]>0 and px[key]<=basis[key]*(1-stop):
                        cash+=mv[key]*(1-C); stops.append(key); del hold[key]; del basis[key]; del mv[key]
            eq=cash+sum(mv.values())
        if d==END: eq=cash+sum(mv.values())*(1-C)
        peak=max(peak,eq); mdd=min(mdd,eq/peak-1)
    if K==1:
        for S,d,names in picks_log:
            if any(S<x<=d for x in deal.get(names[0],())): raise Fail('pick_not_tradable')
    yrs=(dt.date.fromisoformat(END)-dt.date.fromisoformat(D0)).days/365.25
    return eq,eq**(1/yrs)-1,mdd,stops,fb
if __name__=='__main__':
    g={r['path_id']:r for r in csv.DictReader(open(sys.argv[2]))}
    rows=[]; mx=collections.Counter(); res=collections.defaultdict(list); nv=collections.Counter(); fbt=0; sc=collections.Counter(); bad=[]
    for W in (10,11,12):
      for H in (2,3,4):
        for K in (1,2,3):
          for ph in range(1,13):
            pid=f'W{W}H{H}K{K}P{ph}'; r=g[pid]
            f,c,m,s,fb=run(W,H,K,ph,0.30); f0,c0,m0,_,_=run(W,H,K,ph,None); ft,ct,mt,stt,_=run(W,H,K,ph,0.30,True)
            fbt+=fb
            rows.append([pid,W,H,K,ph,repr(c0),repr(m0),repr(c),repr(m),len(s),'|'.join(s),repr(ct),repr(mt),len(stt),r['cagr'],r['mdd'],r['stops']])
            d=(abs(c-float(r['cagr'])),abs(m-float(r['mdd'])),abs(c0-float(r['base_CAGR'])))
            mx['cagr']=max(mx['cagr'],d[0]); mx['mdd']=max(mx['mdd'],d[1]); mx['ctrl']=max(mx['ctrl'],d[2])
            if d[0]>1e-9 or d[1]>1e-9 or len(s)!=int(r['stops']) or '|'.join(s)!=r['stop_names']: bad.append((pid,c,r['cagr'],m,r['mdd'],len(s),r['stops']))
            res[K].append((c0,m0,c,m,len(s),ct,mt,len(stt)))
            if K==1: nv['NVDA_stops']+=s.count('NVDA'); nv['NVDA_trail']+=stt.count('NVDA')
            sc[K]+=len(s)
    print('max diffs vs GPT stop file',dict(mx),'fallback uses',fbt,'mismatching paths',len(bad),bad[:5])
    for K in (1,2,3):
        z=res[K]; med=lambda j:st.median(x[j] for x in z)*100
        print(f'K{K} base CAGR med {med(0):.2f} MDD med {med(1):.2f} worst {min(x[1] for x in z)*100:.2f} | stop CAGR med {med(2):.2f} MDD med {med(3):.2f} worst {min(x[3] for x in z)*100:.2f} paths w/stop {sum(1 for x in z if x[4])} stops {sum(x[4] for x in z)} | TRAIL CAGR med {med(5):.2f} MDD med {med(6):.2f} worst {min(x[6] for x in z)*100:.2f} stops {sum(x[7] for x in z)}')
        print('   paths where stop improved CAGR',sum(1 for x in z if x[2]>x[0]),'/108; improved MDD',sum(1 for x in z if x[3]>x[1]+1e-12),'/108; trail improved CAGR',sum(1 for x in z if x[5]>x[0]),' MDD',sum(1 for x in z if x[6]>x[1]+1e-12))
    print(dict(nv))
    if os.environ.get('OUT_CSV'):
        w=csv.writer(open(os.environ['OUT_CSV'],'w',newline='')); w.writerow(['path_id','W','H','K','phase','base_CAGR','base_MDD','stop30_CAGR','stop30_MDD','stop30_count','stop30_names','trail30_CAGR','trail30_MDD','trail30_count','gpt_stop_CAGR','gpt_stop_MDD','gpt_stop_count']); w.writerows(rows)
