#!/usr/bin/env python3
"""Claude: three quick tests suggested by the literature review, run on data already in the repo (stdlib only).
 1. Where the 2005-2015 Top1 losses came from (by calendar year and by name).
 2. A 12-tranche composite portfolio vs the median of the 12 phases, both decades.
 3. Top1 holding-period returns after UP vs DOWN market states, both decades.
Usage: python3 extra_tests.py   -> prints and writes three CSVs next to this script."""
import csv,gzip,os,math,collections,statistics as st,datetime as dt
HERE=os.path.dirname(os.path.abspath(__file__)); R=os.path.join(HERE,'..','..')
Q05=os.path.join(R,'queen','w10h2-w12h4-oos-2005-2015-20261112'); AK=os.path.join(R,'queen','answer-key-full-20261109')
P15=os.path.join(R,'shared','inputs-pack-full-20261109'); P05=os.path.join(Q05,'pack')
def wr(name,rows):
    w=csv.DictWriter(open(os.path.join(HERE,name),'w',newline=''),fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# ---------- load daily NAVs ----------
nav05=collections.defaultdict(dict)
for r in csv.DictReader(open(os.path.join(Q05,'results','daily_nav.csv'))): nav05[r['path']][r['date']]=float(r['nav'])
nav15=collections.defaultdict(dict)
for W in (10,12):
    rows=list(csv.reader(gzip.open(os.path.join(AK,'nav_daily',f'nav_daily_W{W:02d}.csv.gz'),'rt'))); h=rows[0]
    for pid in [f'W10H2K1P{p}' for p in range(1,13)] if W==10 else [f'W12H4K1P{p}' for p in range(1,13)]+['W12H12K1P1']:
        j=h.index(pid)
        for x in rows[1:]:
            if x[j]!='': nav15[pid][x[0]]=float(x[j])
def stats(series):
    ds=sorted(series); v=[series[d] for d in ds]; pk=0; mdd=0
    for x in v: pk=max(pk,x); mdd=min(mdd,x/pk-1)
    y=(dt.date.fromisoformat(ds[-1])-dt.date.fromisoformat(ds[0])).days/365.25
    return (v[-1]/v[0])**(1/y)-1,mdd
# ---------- 1. loss attribution 2005-2015 ----------
hr=list(csv.DictReader(open(os.path.join(Q05,'results','holdings_rebalances.csv'))))
out1=[]
for fam in ('W10H2K1','W12H4K1'):
    byyear=collections.defaultdict(list); byname=collections.defaultdict(float); n=0
    for r in hr:
        if not r['path'].startswith(fam+'P') or r['segment_return_incl_fees']=='': continue
        g=math.log(1+float(r['segment_return_incl_fees'])); byyear[r['exec_date'][:4]].append(g); byname[r['pick_K1']]+=g; n+=1
    paths=len({r['path'] for r in hr if r['path'].startswith(fam+'P')})
    print(fam,'holding periods',n,'paths',paths)
    for y in sorted(byyear):
        out1.append(dict(family=fam,year_of_entry=y,holding_periods=len(byyear[y]),sum_log_return_all_paths=sum(byyear[y]),avg_log_return_per_path=sum(byyear[y])/paths,share_losing=sum(1 for g in byyear[y] if g<0)/len(byyear[y])))
        print('  entered %s: %3d periods, avg log return per path %+.3f (%.0f%% of periods lost)'%(y,len(byyear[y]),sum(byyear[y])/paths,100*out1[-1]['share_losing']))
    worst=sorted(byname.items(),key=lambda x:x[1])[:6]; best=sorted(byname.items(),key=lambda x:-x[1])[:4]
    print('  worst names (avg log per path):',[(t,round(v/paths,2)) for t,v in worst]); print('  best names:',[(t,round(v/paths,2)) for t,v in best])
    tot=sum(sum(v) for v in byyear.values())/paths; c0809=sum(sum(byyear[y]) for y in ('2007','2008'))/paths
    print('  total avg log per path %.2f; entries in 2007-2008 contribute %.2f; everything else %.2f'%(tot,c0809,tot-c0809))
wr('loss_attribution_2005_2015.csv',out1)
# ---------- 2. tranche composite ----------
out2=[]
def composite(navs,pids):
    pids=[p for p in pids if p in navs]; days=sorted({d for p in pids for d in navs[p]})
    start={p:min(navs[p]) for p in pids}; end={p:max(navs[p]) for p in pids}; comp={}
    for d in days:
        tot=0
        for p in pids:   # each tranche holds 1/n: cash (value 1) before its start, frozen at its final value after its end
            tot+= 1.0 if d<start[p] else (navs[p][d] if d<=end[p] else navs[p][end[p]])
        comp[d]=tot/len(pids)
    return comp
for label,navs in (('2005-2015',nav05),('2015-2025',nav15)):
    for fam in ('W10H2K1','W12H4K1'):
        pids=[f'{fam}P{p}' for p in range(1,13)]; known=[p for p in pids if p in navs]
        each=[stats(navs[p]) for p in known]
        comp=composite(navs,pids); lo=max(min(navs[p]) for p in known); hi=min(max(navs[p]) for p in known)
        c_full=stats(comp); c_common=stats({d:v for d,v in comp.items() if lo<=d<=hi})
        row=dict(decade=label,family=fam,phases_used=len(known),median_phase_CAGR=st.median(x[0] for x in each),median_phase_MDD=st.median(x[1] for x in each),
                 composite_CAGR_full_span=c_full[0],composite_MDD_full_span=c_full[1],composite_CAGR_common_window=c_common[0],composite_MDD_common_window=c_common[1],common_window=f'{lo}..{hi}')
        out2.append(row)
        print(label,fam,'phases',len(known),'| median phase CAGR %.2f%% MDD %.2f%% | 12-tranche composite, common window %s..%s: CAGR %.2f%% MDD %.2f%%'%(row['median_phase_CAGR']*100,row['median_phase_MDD']*100,lo,hi,c_common[0]*100,c_common[1]*100))
wr('tranche_composite.csv',out2)
# ---------- 3. market state ----------
def state_table(P):
    rd=lambda f:list(csv.DictReader(gzip.open(os.path.join(P,f),'rt') if f.endswith('.gz') else open(os.path.join(P,f))))
    px=collections.defaultdict(dict)
    for r in rd('rank_prices_month_end.csv.gz'):
        if r['rank_price']: px[r['engine_ticker']][r['month_end']]=float(r['rank_price'])
    el=collections.defaultdict(list)
    for r in rd('eligibility_by_signal.csv.gz'):
        if r['eligible_final']=='True': el[r['signal_date']].append(r['engine_ticker'])
    ME=sorted({d for v in px.values() for d in v}); mi={d:i for i,d in enumerate(ME)}; out={}
    for S in el:
        B=ME[mi[S]-12]; v=[px[t][S]/px[t][B]-1 for t in el[S] if S in px[t] and B in px[t] and -0.9<px[t][S]/px[t][B]-1<9]
        out[S]=st.median(v)      # median trailing 12-month price return of eligible names (robust market proxy from the pack itself)
    return out
st05=state_table(P05); st15=state_table(P15); out3=[]
seg15=collections.defaultdict(list)
for r in csv.DictReader(gzip.open(os.path.join(AK,'rebalances.csv.gz'),'rt')):
    if r['path_id'].startswith('W10H2K1P') or r['path_id'].startswith('W12H4K1P'): seg15[r['path_id']].append(r)
fin={r['path_id']:float(r['final_multiple']) for r in csv.DictReader(open(os.path.join(AK,'paths_summary.csv'))) if r['final_multiple']}
for label in ('2005-2015','2015-2025'):
    for fam in ('W10H2K1','W12H4K1'):
        obs=[]
        if label=='2005-2015':
            for r in hr:
                if r['path'].startswith(fam+'P') and r['segment_return_incl_fees']!='' and r['signal_date'] in st05: obs.append((st05[r['signal_date']],float(r['segment_return_incl_fees'])))
        else:
            for pid,rs in seg15.items():
                if not pid.startswith(fam+'P'): continue
                for i,r in enumerate(rs):
                    nxt=float(rs[i+1]['equity_before']) if i+1<len(rs) else fin[pid]
                    if r['signal_date'] in st15: obs.append((st15[r['signal_date']],nxt/float(r['equity_after'])-1))
        for name,sel in (('UP (median stock up over prior 12 months)',lambda m:m>=0),('DOWN',lambda m:m<0)):
            z=[x[1] for x in obs if sel(x[0])]
            if not z: continue
            row=dict(decade=label,family=fam,state=name,holding_periods=len(z),mean_return=st.mean(z),median_return=st.median(z),share_positive=sum(1 for x in z if x>0)/len(z),worst=min(z))
            out3.append(row); print(label,fam,name[:4],'n=%d mean %+.1f%% median %+.1f%% positive %.0f%% worst %+.0f%%'%(len(z),row['mean_return']*100,row['median_return']*100,row['share_positive']*100,row['worst']*100))
wr('market_state_split.csv',out3)
