"""Extend the PROVISIONAL 2005-2015 pack's EXECUTION inputs to every path needed for the full 5,184 grid (W1-12 x H1-12 x K1-3 x 12 phases)
plus the blended sleeves (W10-12, H in {1,3,6,12}, K in {5,10,20}). Rank-side inputs (eligibility, rank prices, overrides, rank CAs,
sessions, deal dates) are the 2026-11-12 pack files, unchanged. Execution-series selection = the 2026-11-12 build's exec_select.select()
(same candidate order, same 0.5% month-end ratio test vs rank price, no missing session), applied per merged holding spell, with
fallback to each original spell / buy window. Holding events: the 6 original rows (re-keyed to the series id used) + spin/special-div
rows generated from the rank-side CA table for any held segment spanning the ex-date. Acquisitions while held are NOT added: the
series then ends early, selection fails, and those paths become UNKNOWN (reported)."""
import sys,os,csv,gzip,collections,json,datetime as dt
OUT='/workspace/momentum-followup-20261114/pack0515/'
sys.path.insert(0,'/workspace/w10h2-w12h4-oos-2005-2015-20261112/build')
os.chdir(OUT);sys.argv=[sys.argv[0]];sys.path.insert(0,OUT)
import rebuild_2005_2015 as R
import exec_select as ES
_MY='/workspace/momentum-followup-20261114/raw_massive_aggs/'
def _aggs(X):
    for f in (_MY+X+'.csv', ES.ROOT+'raw/massive_aggs/'+X+'.csv'):
        if os.path.exists(f):
            return {r['date']:float(r['close']) for r in csv.DictReader(open(f))}
    return {}
ES.aggs=_aggs
SES=R.SES;SI=R.SI;SIGSET=set(R.ELIG)
def sched(H,ph):
    D0=R.first_td(2005,ph);END=R.first_td(2015,ph);rebs=[];j=0
    while True:
        mm=(ph-1)+j*H;d=R.first_td(2005+mm//12,mm%12+1)
        if d>=END: break
        rebs.append(d);j+=1
    return D0,END,rebs
jobs=[(W,H,K,ph) for W in range(1,13) for H in range(1,13) for K in (1,2,3) for ph in range(1,13)]
jobs+=[(W,H,K,ph) for W in (10,11,12) for H in (1,3,6,12) for K in (5,10,20) for ph in range(1,13)]
SPELL=set();WIN=set();outside=0
for W,H,K,ph in jobs:
    D0,END,rebs=sched(H,ph);nxt=dict(zip(rebs,rebs[1:]+[END]));hold={};stop=None
    for d in rebs:
        S=SES[SI[d]-1]
        if S not in SIGSET: outside+=1;stop=d;break
        r,miss,_=R.rank(S,W)
        if miss: stop=d;break
        sel=[t for _,t in r[:K]];keys=[R.ek(t,d) for t in sel]
        if len(set(keys))!=len(keys): stop=d;break
        for t in list(hold):
            if t not in keys: SPELL.add((t,hold.pop(t),d))
        for t,k in zip(sel,keys):
            if k not in hold: hold[k]=d;WIN.add((d,t,nxt[d]))
    for t,a in hold.items(): SPELL.add((t,a,stop or END))
print('jobs',len(jobs),'spells',len(SPELL),'buy windows',len(WIN),'paths hitting a signal outside the pack',outside,flush=True)
by=collections.defaultdict(list)
for k,a,b in SPELL: by[k].append((a,b))
MER={}
for k,v in by.items():
    m=[]
    for a,b in sorted(v):
        if m and a<=m[-1][1]: m[-1][1]=max(m[-1][1],b)
        else: m.append([a,b])
    MER[k]=m
KL=collections.defaultdict(set)
for d,t,e in WIN: KL[R.ek(t,d)].add(t)
print('AGN_PLC labels',KL.get('AGN_PLC'),'AGN_LEGACY labels',KL.get('AGN_LEGACY'))
def label_of(k): return 'AGN' if k=='AGN_LEGACY' else (sorted(KL[k])[0] if k=='AGN_PLC' else k)   # AGN_PLC: tried with its rank label; if the 0.5% test fails the spell is UNKNOWN
SER={};EXR={};SINFO=[];SEL=[];fails=[]
def add_series(k,L,name,ser,a,b):
    sid=f'{name}#{a}#{b}';last=min(b,ES.LASTTRADE.get(L,'9999'));ds=[s for s in SES if a<=s<=last]
    if not ds or ds[0]!=a: return None   # no session at entry (e.g. deal already closed)
    if sid not in SER:
        SER[sid]=(k,a,ds[-1])
        for s in ds: EXR[(sid,k,s)]=ser[s]
        SINFO.append([sid,k,ds[0],ds[-1]])
    return sid
def sel(L,a,b):
    try: return ES.select(L,a,b)
    except Exception as e: return None,None,[('error',repr(e))]
SEGS=collections.defaultdict(list);nfail_m=0
for k,m in sorted(MER.items()):
    L=label_of(k)
    for a,b in m:
        name,ser,res=sel(L,a,b)
        sid=add_series(k,L,name,ser,a,b) if name is not None else None
        if sid is not None:
            SEGS[k].append((a,b,sid));SEL.append([k,a,b,name,'merged_spell']);continue
        nfail_m+=1
        subs=sorted({(x,y) for kk,x,y in SPELL if kk==k and a<=x and y<=b}|{(d,e) for d,t,e in WIN if R.ek(t,d)==k and a<=d and e<=b})
        for a2,b2 in subs:
            name,ser,res=sel(L,a2,b2)
            sid=add_series(k,L,name,ser,a2,b2) if name is not None else None
            if sid is not None: SEGS[k].append((a2,b2,sid));SEL.append([k,a2,b2,name,'fallback_subspell'])
            else: fails.append([k,a2,b2,json.dumps(res,default=str)[:400]])
print('merged spells',sum(map(len,MER.values())),'merged fails',nfail_m,'subspell/window fails',len(fails),flush=True)
DEALS={r['engine_ticker']:r['deal_date'] for r in csv.DictReader(open('untradable_deal_dates.csv'))}
EA=[];cnt=collections.Counter()
for d,t,e in sorted(WIN):
    k=R.ek(t,d);dd=DEALS.get(t)
    if dd and (dt.date.fromisoformat(d)-dt.timedelta(60)).isoformat()<dd<=d:
        EA.append([d,t,e,'not_tradable','',f'{t}@{dd}','','','']);cnt['not_tradable']+=1;continue
    c=[(SER[sid][2],sid) for a,b,sid in SEGS.get(k,[]) if a<=d and e<=b]
    if not c: EA.append([d,t,e,'no_verified_series','','no series passing 0.5% month-end test with no missing session','','','']);cnt['no_verified_series']+=1;continue
    b,sid=max(c);EA.append([d,t,e,'ok',sid,'',repr(EXR[(sid,k,d)]),'','UNKNOWN: open not collected (engine trades at close)']);cnt['ok']+=1
print('availability',dict(cnt),flush=True)
def w(name,hdr,rows):
    f=gzip.open(OUT+name,'wt',newline='') if name.endswith('.gz') else open(OUT+name,'w',newline='');x=csv.writer(f);x.writerow(hdr);x.writerows(rows);f.close()
w('execution_availability.csv.gz',['trade_date','engine_ticker','window_end','status','price_series','detail','exec_close','exec_open','open_note'],EA)
acq=[r for r in csv.reader(gzip.open(OUT+'original_25path_exec/execution_daily_closes.csv.gz','rt')) if r[0].startswith('acq:')]
w('execution_daily_closes.csv.gz',['price_series','execution_key','date','close'],[[s,k,d,repr(v)] for (s,k,d),v in sorted(EXR.items())]+acq)
SI0=[r for r in csv.reader(open(OUT+'original_25path_exec/series_info.csv')) if r[0].startswith('acq:')]
w('series_info.csv',['price_series','execution_key','series_first_date','series_last_date'],SINFO+SI0)
HE0=list(csv.DictReader(open(OUT+'original_25path_exec/holding_events.csv')));hdr=list(HE0[0].keys());HE=[]
def prev(x): return max(s for s in SES if s<x)
def covering(k,ex): return [sid for a,b,sid in SEGS.get(k,[]) if a<ex and SER[sid][2]>=prev(ex)]
done=set()
for r in HE0:
    for sid in covering(r['execution_key'],r['ex_date']):
        x=dict(r);x['price_series']=sid;HE.append([x[h] for h in hdr])
    done.add((r['execution_key'],r['ex_date']))
nsp=0
for r in csv.DictReader(open('corporate_actions_applied.csv')):
    if r['kind'] not in('spin','special_div') or (r['engine_ticker'],r['ex_date']) in done: continue
    for sid in covering(r['engine_ticker'],r['ex_date']):
        HE.append([r['engine_ticker'],sid,r['event_id'],r['kind'],r['ex_date'],r['ex_date'],r['ratio'],r['cash_per_share'] or 0,r['child_first_close'],r['child'],r['evidence_grade'],'','False','apply','generated from rank-side CA row (pack corporate_actions_applied.csv)']);nsp+=1
w('holding_events.csv',hdr,HE)
print('holding events',len(HE),'generated spin/special rows',nsp)
w('../results/pack0515_exec_selection.csv',['execution_key','from','to','price_series','kind'],SEL)
w('../results/pack0515_exec_failures.csv',['execution_key','from','to','candidates_tried'],fails)
