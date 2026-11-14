"""Summaries for the fixed rule (rank-avg W10/11/12, H12, 12 tranches, top N EW): exact-engine strict, S1 sensitivity, Claude-proxy rerun.
Benchmarks: EW-universe (Claude proxy code, month-end), SPY price-only and total return (FMP daily, same first-session dates as the engine).
Significance on monthly excess (120 months): simple t, Newey-West t (Bartlett, lag 6), circular block bootstrap (block 6, 10000, one-sided p)."""
import pickle,glob,csv,math,numpy as np,pandas as pd,datetime as dt
B='/workspace/momentum-followup-20261114/';OUT=B+'results/fixed_rule/'
NS=[1,2,3,5,10,20,30,50]
CP=pickle.load(open(B+'scratch/claude_proxy_comp.pkl','rb'))['COMP']
spy={k:pd.read_csv(f'/workspace/momentum-etf-check-20261113/raw/SPY_{k}_daily.csv').set_index('date').iloc[:,0].astype(float) for k in ('tr','px')}
rng=np.random.default_rng(20261114)
def nw_t(x,L=6):
    x=np.asarray(x);n=len(x);m=x.mean();e=x-m;v=(e@e)/n
    for l in range(1,L+1): v+=2*(1-l/(L+1))*(e[l:]@e[:-l])/n
    return m/math.sqrt(v/n)
def simple_t(x): x=np.asarray(x);return x.mean()/(x.std(ddof=1)/math.sqrt(len(x)))
def boot_p(x,b=6,R=10000):
    x=np.asarray(x);n=len(x);k=math.ceil(n/b);st=rng.integers(0,n,(R,k));idx=(st[:,:,None]+np.arange(b))%n;ms=x[idx.reshape(R,-1)[:,:n]].mean(1)
    return float((ms<=0).mean())
def mdd(v): v=np.asarray(v);return float((v/np.maximum.accumulate(v)-1).min())
def firsts(days):
    out=[];seen=set()
    for d in days:
        if d[:7] not in seen: seen.add(d[:7]);out.append(d)
    return out
def load_engine(period,variant):
    out={}
    if period=='2015-2025':
        for f in glob.glob(B+f'scratch/fixed1525_{variant}_N*.pkl'):
            d=pickle.load(open(f,'rb'));N=d['P'][0]['N'];out[N]=d
    else:
        d=pickle.load(open(B+f'scratch/fixed0515_{variant}.pkl','rb'))
        for N in NS: out[N]=dict(P=[p for p in d['P'] if p['N']==N],NAV={k:v for k,v in d['NAV'].items() if f'_N{N}_' in k},days=d['days'],RB=[r for r in d['RB'] if f'_N{N}_' in r['tranche']])
    return out
rows=[];trows=[];EXS={}
def stats_block(label,period,univ,N,comp_m,ew_m,spy_tr_m,extra):
    exE=np.diff(comp_m)/comp_m[:-1]-np.diff(ew_m)/ew_m[:-1]
    exS=np.diff(comp_m)/comp_m[:-1]-np.diff(spy_tr_m)/spy_tr_m[:-1]
    EXS[(label,period,univ,N)]=exE
    r=dict(period=period,universe=univ,N=N,variant=label,**extra,
           excess_vs_EW_mean_pp_yr=exE.mean()*1200,t_simple_vs_EW=simple_t(exE),t_NW6_vs_EW=nw_t(exE),boot_p_vs_EW=boot_p(exE),months_ahead_EW=float((exE>0).mean()),
           excess_vs_SPYTR_mean_pp_yr=exS.mean()*1200,t_NW6_vs_SPYTR=nw_t(exS),boot_p_vs_SPYTR=boot_p(exS))
    rows.append(r)
for period in ('2015-2025','2005-2015'):
    univs=(('all names','base'),('no NVDA','nonvda')) if period=='2015-2025' else (('all names',None),)
    for univ,tag in univs:
        ew=np.array(CP[(period,univ,'EW')][0]);ewc=ew[-1]**0.1-1
        for variant in (['strict','s1','s1t'] if period=='2005-2015' else [tag,tag+'_s1']):
            E=load_engine(period,variant)
            for N in NS:
                d=E[N];P=d['P'];known=[p for p in P if p['status']!='UNKNOWN'];days=d['days'];fm=firsts(days)
                for p in P: trows.append(dict(period=period,universe=univ,variant='s1t_sensitivity' if variant=='s1t' else ('s1_sensitivity' if variant.endswith('s1') else 'strict'),**{k:p.get(k) for k in ('tranche','N','off','entry','end','status','fail_reason','fail_date','CAGR','daily_MDD','postcheck_all_picks_hit')},fail_or_events=(p.get('fail_ticker') or '')+(p.get('events') or '')))
                lab='s1t_sensitivity_tail_cashout' if variant=='s1t' else ('s1_sensitivity_cashout' if variant.endswith('s1') else 'exact_engine_strict')
                nev=sum(1 for p in P if p['status']=='SENS_CASHOUT')
                if len(known)<12:
                    pc=[p['CAGR'] for p in known]
                    rows.append(dict(period=period,universe=univ,N=N,variant=lab,known_tranches=len(known),status='UNKNOWN',
                        partial_known_tranche_median_CAGR=float(np.median(pc)) if pc else None,tranche_min_CAGR=min(pc) if pc else None,tranche_max_CAGR=max(pc) if pc else None,
                        fail_examples='; '.join(sorted({f"{p['fail_reason']}:{p['fail_ticker']}" for p in P if p['status']=='UNKNOWN'}))[:200]))
                    continue
                navs=np.array([d['NAV'][p['tranche']] for p in P]);comp=navs.mean(0)
                yrs=(dt.date.fromisoformat(days[-1])-dt.date.fromisoformat(days[0])).days/365.25
                ix=[days.index(x) for x in fm];cm=np.concatenate([[1.0],comp[ix[1:]]]) if len(ix)==121 else None
                assert len(ix)==121 and days[-1]==fm[-1],(len(ix),days[-1],fm[-1])
                cm=np.array([1.0]+[comp[i] for i in ix[1:]])
                st=np.array([spy['tr'][x] for x in fm]);sp=np.array([spy['px'][x] for x in fm])
                ewu=ew if univ=='all names' else np.array(CP[(period,univ,'EW')][0])
                tc=[p['CAGR'] for p in P]
                stats_block(lab,period,univ,N,cm,ewu,st,dict(known_tranches=12,status='KNOWN' if lab.startswith('exact') else 'SENSITIVITY',CAGR=comp[-1]**(1/yrs)-1,daily_MDD=mdd(comp),monthly_MDD=mdd(cm),
                    tranche_min_CAGR=min(tc),tranche_max_CAGR=max(tc),tranche_median_CAGR=float(np.median(tc)),EW_proxy_CAGR=ewc if univ=='all names' else ewu[-1]**0.1-1,
                    SPY_px_CAGR=(sp[-1]/sp[0])**(1/yrs)-1,SPY_tr_CAGR=(st[-1]/st[0])**(1/yrs)-1,SPY_tr_daily_MDD=mdd(spy['tr'][(spy['tr'].index>=days[0])&(spy['tr'].index<=days[-1])].values),
                    CAGR_minus_EW_pp=(comp[-1]**(1/yrs)-1-(ewu[-1]**0.1-1))*100,tranches_with_cashout=nev))
        # Claude proxy rerun rows (his code, local re-run, identical to his CSV)
        for N in NS:
            comp,tc=CP[(period,univ,'W10-12',N,12)];comp=np.array(comp)
            months=pd.date_range('2014-12-31' if period=='2015-2025' else '2004-12-31',periods=121,freq='ME')
            ewu=np.array(CP[(period,univ,'EW')][0])
            # SPY on month-end dates (nearest prior session)
            def at(s,dd): return s[s.index<=dd.strftime('%Y-%m-%d')].iloc[-1]
            st=np.array([at(spy['tr'],m) for m in months])
            stats_block('claude_proxy_rerun',period,univ,N,comp,ewu,st,dict(known_tranches=12,status='PROXY',CAGR=comp[-1]**0.1-1,daily_MDD=None,monthly_MDD=mdd(comp),
                tranche_min_CAGR=min(tc),tranche_max_CAGR=max(tc),tranche_median_CAGR=float(np.median(tc)),EW_proxy_CAGR=ewu[-1]**0.1-1,CAGR_minus_EW_pp=(comp[-1]**0.1-1-(ewu[-1]**0.1-1))*100))
D=pd.DataFrame(rows);D.to_csv(OUT+'fixed_rule_summary.csv',index=False)
pr=[]
for N in NS:
    for name,a,b in (('engine: 2015-25 strict-or-S1 + 2005-15 S1t',('exact_engine_strict' if ('exact_engine_strict','2015-2025','all names',N) in EXS else 's1_sensitivity_cashout','2015-2025','all names',N),('s1t_sensitivity_tail_cashout','2005-2015','all names',N)),
                     ('claude proxy rerun',('claude_proxy_rerun','2015-2025','all names',N),('claude_proxy_rerun','2005-2015','all names',N))):
        x=np.r_[EXS[b],EXS[a]];pr.append(dict(N=N,source=name,months=len(x),mean_pp_yr=x.mean()*1200,t_simple=simple_t(x),t_NW6=nw_t(x),boot_p=boot_p(x),months_ahead=float((x>0).mean())))
pd.DataFrame(pr).to_csv(OUT+'pooled_20y_vs_EW.csv',index=False);print(pd.DataFrame(pr).round(3).to_string())
pd.DataFrame(trows).to_csv(OUT+'fixed_rule_tranches.csv',index=False)
cols=['period','universe','N','variant','known_tranches','CAGR','daily_MDD','monthly_MDD','tranche_min_CAGR','tranche_max_CAGR','partial_known_tranche_median_CAGR','EW_proxy_CAGR','CAGR_minus_EW_pp','excess_vs_EW_mean_pp_yr','t_NW6_vs_EW','boot_p_vs_EW','excess_vs_SPYTR_mean_pp_yr','t_NW6_vs_SPYTR','tranches_with_cashout','fail_examples']
pd.set_option('display.width',250);pd.set_option('display.max_columns',30)
print(D[[c for c in cols if c in D]].round(4).to_string())
