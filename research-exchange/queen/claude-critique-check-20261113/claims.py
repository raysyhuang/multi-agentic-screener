"""Claims 1, 2, 4, 5 (Claude critique check). Writes claim*_*.csv. Engine: engine_ext (v18 rules)."""
import sys,json,csv,math,statistics as st,datetime as dt;sys.path.insert(0,'.')
import engine_ext as M
from engine_ext import Eng
P15='/workspace/momentum432-inputs-pack-full-20261109';P05='/workspace/w10h2-w12h4-oos-2005-2015-20261112/pack'
SPY15={x['date']:x['close'] for x in json.load(open('/workspace/nh-matrix-bt-v2/bars/SPY.px.json'))}
SPY05={x['date']:x['close'] for x in json.load(open('/workspace/w10h2-w12h4-oos-2005-2015-20261112/raw/fmp/SPY.json'))}
def spy(E,SP,ph):
    D0=E.first_td(E.y0,ph);END=E.first_td(E.y0+10,ph);yrs=(dt.date.fromisoformat(END)-dt.date.fromisoformat(D0)).days/365.25
    xs=[SP[d] for d in E.SES if D0<=d<=END];pk=0;mdd=0
    for v in xs: pk=max(pk,v);mdd=min(mdd,v/pk-1)
    m=SP[END]/SP[D0];return dict(CAGR=m**(1/yrs)-1,MDD=mdd,mult=m,CAGR_fee=(m*(1-.0025)/(1+.0025))**(1/yrs)-1) # fee version: 25bp in, 25bp out
def regime(E,SP):
    """risk_off[S]=True when SPY close at month-end S < mean of the 10 month-end closes ending at S (price-only)"""
    out={}
    for i,m in enumerate(E.ME):
        if i<9 or any(x not in SP for x in E.ME[i-9:i+1]): continue
        out[m]=SP[m]<sum(SP[x] for x in E.ME[i-9:i+1])/10
    return out
def segs(o):
    """K1 attribution: per rebalance segment log growth, NAV points at rebalance dates (post-trade), start=1.0 pre-fee"""
    dts,eq=o['nav'];idx={d:i for i,d in enumerate(dts)};log_=o['rebalances'];out=[]
    for j,(S,d,sel,inv) in enumerate(log_):
        a=1.0 if j==0 else eq[idx[d]]
        b=eq[idx[log_[j+1][1]]] if j+1<len(log_) else eq[-1]
        out.append((d,inv,math.log(b/a)))
    return out
def med(x): return st.median(x) if x else None
# ---------------- Claim 1 (2015-2025)
E=Eng(P15,2015);rows=[]
cfg=[(10,2,p) for p in range(1,13)]+[(12,4,p) for p in range(1,13)]+[(12,12,1)]
for W,H,ph in cfg:
    b=E.path(W,H,1,ph,want_nav=True);x=E.path(W,H,1,ph,want_nav=True,excl=frozenset({'NVDA'}));c=E.path(W,H,1,ph,want_nav=True,cash_on=frozenset({'NVDA'}))
    sp=spy(E,SPY15,ph);r=dict(path=f'W{W}H{H}K1P{ph}',base_status=b['status'],base_CAGR=b.get('CAGR'),base_MDD=b.get('MDD'),base_mult=b.get('final'))
    if b['status']!='UNKNOWN':
        sg=segs(b);tot=math.log(b['final']);nv=sum(g for d,p,g in sg if p=='NVDA')
        r.update(nvda_segments=sum(p=='NVDA' for d,p,g in sg),nvda_logshare=nv/tot,nvda_log=nv,total_log=tot,
                 nvda_hold_dates=' '.join(d for d,p,g in sg if p=='NVDA'))
    r.update(exNVDA_status=x['status'],exNVDA_CAGR=x.get('CAGR'),exNVDA_MDD=x.get('MDD'),exNVDA_mult=x.get('final'),exNVDA_fail=x.get('fail_reason',''),exNVDA_ext=('ext_exec' in x['flags']),
             cashNVDA_status=c['status'],cashNVDA_CAGR=c.get('CAGR'),cashNVDA_MDD=c.get('MDD'),cashNVDA_mult=c.get('final'),
             SPY_CAGR=sp['CAGR'],SPY_CAGR_fee=sp['CAGR_fee'],SPY_MDD=sp['MDD'],SPY_mult=sp['mult'])
    if x['status']!='UNKNOWN':
        r['exNVDA_picks_replacing']=' '.join(f"{d}:{inv}" for (S,d,sel,inv),(S2,d2,sel2,inv2) in zip(x['rebalances'],b['rebalances']) if inv!=inv2)
    rows.append(r)
w=csv.DictWriter(open('claim1_nvda_2015_2025.csv','w',newline=''),fieldnames=sorted({k for r in rows for k in r},key=lambda k:list(rows[0]).index(k) if k in rows[0] else 99));w.writeheader();w.writerows(rows)
summ=[]
for nm,pre in (('W10H2K1','W10H2K1P'),('W12H4K1','W12H4K1P')):
    rr=[r for r in rows if r['path'].startswith(pre)]
    def m(k): v=[r[k] for r in rr if r.get(k) is not None];return med(v),len(v)
    summ.append(dict(cell=nm,**{f'median_{k}':m(k)[0] for k in ('base_CAGR','base_MDD','nvda_logshare','exNVDA_CAGR','exNVDA_MDD','cashNVDA_CAGR','cashNVDA_MDD','SPY_CAGR','SPY_MDD')},
        n_exNVDA_known=m('exNVDA_CAGR')[1],P1_base=rr[0]['base_CAGR'],P1_nvda_logshare=rr[0].get('nvda_logshare'),P1_exNVDA=rr[0]['exNVDA_CAGR'],P1_cashNVDA=rr[0]['cashNVDA_CAGR'],P1_SPY=rr[0]['SPY_CAGR']))
r=rows[-1];summ.append(dict(cell='W12H12K1P1',median_base_CAGR=r['base_CAGR'],median_base_MDD=r['base_MDD'],median_nvda_logshare=r.get('nvda_logshare'),median_exNVDA_CAGR=r['exNVDA_CAGR'],median_exNVDA_MDD=r['exNVDA_MDD'],median_cashNVDA_CAGR=r['cashNVDA_CAGR'],median_cashNVDA_MDD=r['cashNVDA_MDD'],median_SPY_CAGR=r['SPY_CAGR'],median_SPY_MDD=r['SPY_MDD']))
w=csv.DictWriter(open('claim1_summary.csv','w',newline=''),fieldnames=list(summ[0]));w.writeheader();[w.writerow({k:s.get(k) for k in summ[0]}) for s in summ]
print('claim1 done')
# ---------------- Claim 5 (both windows) skip-month and SPY 10-month SMA regime filter
out=[]
for y0,pk,SP in ((2015,P15,SPY15),(2005,P05,SPY05)):
  for approx in ((False,) if y0==2015 else (False,True)):
    M.APPROX_DELIST=approx;E=Eng(pk,y0);RG=regime(E,SP)
    for W,H in ((10,2),(12,4)):
        for var in ('base','skip1','regime_sma10','skip1+regime'):
            for ph in range(1,13):
                kw=dict(skip=var in('skip1','skip1+regime'),regime=RG if 'regime' in var else None)
                o=E.path(W,H,1,ph,**kw)
                out.append(dict(window=f'{y0}-{y0+10}',data_mode=('approx_delist' if approx else 'strict'),cell=f'W{W}H{H}K1',variant=var,phase=ph,status=o['status'],CAGR=o.get('CAGR'),MDD=o.get('MDD'),final=o.get('final'),
                    fail=o.get('fail_reason',''),fail_ticker=o.get('fail_ticker',''),ext=('ext_exec' in o['flags']),flags=';'.join(sorted(f for f in o['flags'] if f!='ext_exec')),
                    n_cash_periods=sum(1 for r in o['rebalances'] if r[3]=='CASH')))
w=csv.DictWriter(open('claim5_variants_paths.csv','w',newline=''),fieldnames=list(out[0]));w.writeheader();w.writerows(out)
import collections;g=collections.defaultdict(list)
for r in out: g[(r['window'],r['data_mode'],r['cell'],r['variant'])].append(r)
srows=[]
for k,v in g.items():
    kn=[r for r in v if r['status']!='UNKNOWN']
    srows.append(dict(window=k[0],data_mode=k[1],cell=k[2],variant=k[3],n_known=len(kn),median_CAGR=med([r['CAGR'] for r in kn]),median_MDD=med([r['MDD'] for r in kn]),
        min_CAGR=min([r['CAGR'] for r in kn],default=None),max_CAGR=max([r['CAGR'] for r in kn],default=None),P1_CAGR=v[0]['CAGR'],
        median_final=med([r['final'] for r in kn]),unknown=' '.join(f"P{r['phase']}:{r['fail']}:{r['fail_ticker']}" for r in v if r['status']=='UNKNOWN')))
w=csv.DictWriter(open('claim5_summary.csv','w',newline=''),fieldnames=list(srows[0]));w.writeheader();w.writerows(srows)
# regime months in cash
for y0,pk,SP in ((2015,P15,SPY15),(2005,P05,SPY05)):
    E=Eng(pk,y0);RG=regime(E,SP);ms=[m for m in RG if f'{y0-1}-12'<=m[:7]<=f'{y0+9}-12'];print(y0,'risk_off signals',sum(RG[m] for m in ms),'of',len(ms))
print('claim5 done')
# ---------------- Claim 2 & 4 (2005-2015) from packet results + own engine annual returns
M.APPROX_DELIST=False;E=Eng(P05,2005);o=E.path(12,12,1,1,want_nav=True);dts,eq=o['nav'];idx={d:i for i,d in enumerate(dts)}
ann=[];L=o['rebalances']
for j,(S,d,sel,inv) in enumerate(L):
    a=1.0 if j==0 else eq[idx[d]];b=eq[idx[L[j+1][1]]] if j+1<len(L) else eq[-1]
    ann.append(dict(year=d[:4],signal=S,entry=d,pick=inv,period_return=b/a-1,SPY_period=SPY05[L[j+1][1] if j+1<len(L) else o['end']]/SPY05[d]-1))
w=csv.DictWriter(open('claim4_W12H12K1P1_annual_2005_2014.csv','w',newline=''),fieldnames=list(ann[0]));w.writeheader();w.writerows(ann)
print('claim4',[(a['year'],a['pick'],round(a['period_return']*100,2)) for a in ann])
rp=list(csv.DictReader(open('/workspace/w10h2-w12h4-oos-2005-2015-20261112/results/results_paths.csv')))
c2=[]
for pre in ('W10H2K1P','W12H4K1P'):
    rr=[r for r in rp if r['path'].startswith(pre) and r['status']!='UNKNOWN']
    fin=[float(r['final_equity']) for r in rr]
    c2.append(dict(cell=pre[:-1],n_known=len(rr),median_final=med(fin),min_final=min(fin),max_final=max(fin),P1_final=[float(r['final_equity']) for r in rp if r['path']==pre+'1'][0],
        median_CAGR=med([float(r['CAGR']) for r in rr]),median_MDD=med([float(r['MDD']) for r in rr]),n_final_below_0p5=sum(f<0.5 for f in fin),n_final_below_1=sum(f<1 for f in fin)))
c2.append(dict(cell='W12H12K1P1',n_known=1,median_final=o['final'],P1_final=o['final'],median_CAGR=o['CAGR'],median_MDD=o['MDD']))
sp=[spy(E,SPY05,ph) for ph in range(1,13)]
c2.append(dict(cell='SPY price-only (12 phases)',n_known=12,median_final=med([s['mult'] for s in sp]),min_final=min(s['mult'] for s in sp),max_final=max(s['mult'] for s in sp),P1_final=sp[0]['mult'],median_CAGR=med([s['CAGR'] for s in sp]),median_MDD=med([s['MDD'] for s in sp])))
w=csv.DictWriter(open('claim2_2005_2015_multiples.csv','w',newline=''),fieldnames=list(c2[0]));w.writeheader();[w.writerow(r) for r in c2]
print('claim2 done')
