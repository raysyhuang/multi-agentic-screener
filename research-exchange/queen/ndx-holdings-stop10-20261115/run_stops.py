"""Part 2: Ray's stop / buy-back rule and sensitivity grid. Writes out/stop_*.csv, out/charts/stop_nav_dd.png"""
from sim import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, datetime as dt
YRS=(i1-i0)/12
def mdd(seq):
    pk=-1e9; m=0
    for v in seq: pk=max(pk,v); m=min(m,v/pk-1)
    return m
def year_stats(res):
    out={}; dates=res['ddates']; dn=res['dnav']
    for y in range(2015,2027):
        idx=[j for j,d in enumerate(dates) if d[:4]==str(y)]
        start=max(j for j,d in enumerate(dates) if d[:4]<str(y))
        seq=[dn[start]]+[dn[j] for j in idx]
        out[y]=(seq[-1]/seq[0]-1, mdd(seq))
    return out
variants=[('baseline (no stop)',None,'entry','close','sell')]
for mode in ('entry','trail'):
    for x in (0.05,0.10,0.15,0.20):
        for exe in ('close','open'):
            variants.append(('%s %d%% %s'%('from-entry' if mode=='entry' else 'trailing',round(x*100),'same-close' if exe=='close' else 'next-open'),x,mode,exe,'sell'))
for exe in ('close','open'):   # sensitivity: re-enter only once back above the stop line (matters when the stop close gapped through the line)
    variants.append(('from-entry 10%% %s reenter@line'%('same-close' if exe=='close' else 'next-open'),0.10,'entry',exe,'line'))
rows=[]; yrows=[]; R={}; evlog=[]
for name,x,mode,exe,rm in variants:
    lg=[] if (x==0.10 and mode=='entry') else None
    r=simulate(x,mode,exe,lg,rm); s=stats(r); R[name]=r
    if lg: 
        for e in lg: e['variant']=name; evlog.append(e)
    ev=r['ev']; ys=year_stats(r)
    rows.append(dict(variant=name,stop=x or '',mode=mode if x else '',execution=exe if x else '',reentry_trigger=rm if x else '',CAGR=s['CAGR'],MDD_daily=s['MDD_daily'],MDD_monthly=s['MDD_monthly'],
        multiple=s['total'],worst_month=s['worst_month'],worst_month_end=s['worst_month_date'],stop_events=ev['stops'],reentries=ev['reentries'],
        whipsaw_rebuy_above_sell=ev['whipsaw'],stops_helped=ev['stop_helped'],stops_hurt=ev['stop_hurt'],
        stop_cost_drag_pct_per_yr=ev['cost_stop']/YRS*100,rebalance_cost_pct_per_yr=sum(rr['cost'] for rr in r['rec'])/YRS*100,
        pct_slot_days_in_cash=(ev['cash_slot_days']/ev['slot_days']*100) if ev['slot_days'] else 0,
        calmar=s['CAGR']/abs(s['MDD_daily']),
        y2020_ret=ys[2020][0],y2020_mdd=ys[2020][1],y2022_ret=ys[2022][0],y2022_mdd=ys[2022][1]))
    yrows.append(dict(variant=name,**{str(y):ys[y][0] for y in ys}))
b=rows[0]
for r in rows:
    r['dCAGR_vs_base_pp']=(r['CAGR']-b['CAGR'])*100; r['dMDDd_vs_base_pp']=(r['MDD_daily']-b['MDD_daily'])*100
def wcsv(fn,rs):
    with open(os.path.join(OUT,fn),'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0])); w.writeheader()
        for r in rs: w.writerow({k:(round(v,6) if isinstance(v,float) else v) for k,v in r.items()})
wcsv('stop_variants_summary.csv',rows); wcsv('stop_variants_yearly_returns.csv',yrows); wcsv('stop10_from_entry_event_log.csv',evlog)
print('%-30s %7s %7s %7s %8s %7s %5s %5s %5s %6s %6s %6s %6s'%('variant','CAGR','MDDd','MDDm','x','worstM','stops','reent','whip','costpy','cash%','dCAGR','dMDD'))
for r in rows:
    print('%-30s %6.2f%% %6.1f%% %6.1f%% %8.2f %6.1f%% %5d %5d %5d %5.2f%% %5.1f%% %+6.2f %+6.1f'%(r['variant'],r['CAGR']*100,r['MDD_daily']*100,r['MDD_monthly']*100,r['multiple'],
      r['worst_month']*100,r['stop_events'],r['reentries'],r['whipsaw_rebuy_above_sell'],r['stop_cost_drag_pct_per_yr'],r['pct_slot_days_in_cash'],r['dCAGR_vs_base_pp'],r['dMDDd_vs_base_pp']))
print('\n2020 / 2022 (calendar-year return, intra-year daily MDD)')
for r in rows: print('%-30s 2020 %+7.1f%% mdd %6.1f%% | 2022 %+7.1f%% mdd %6.1f%% | helped %d hurt %d'%(r['variant'],r['y2020_ret']*100,r['y2020_mdd']*100,r['y2022_ret']*100,r['y2022_mdd']*100,r['stops_helped'],r['stops_hurt']))
print('\nYearly returns'); 
print('%-30s'%'variant'+''.join('%8s'%y for y in range(2015,2027)))
for y in yrows: print('%-30s'%y['variant']+''.join('%7.1f%%'%(y[str(k)]*100) for k in range(2015,2027)))
P,TR=qqq(); q=bench(P,i0,i1); 
cand=[r for r in rows[1:] if r['reentry_trigger']=='sell']; best=max(cand,key=lambda r:r['calmar'])
print('\nbest by Calmar (CAGR/|daily MDD|, in-sample pick):',best['variant'],round(best['calmar'],3),'baseline calmar',round(b['calmar'],3))
print('missing opens fallback count:',OPEN_MISS[0])
json.dump(dict(best=best['variant']),open(os.path.join(OUT,'best_variant.json'),'w'))
# chart
fig,(a1,a2)=plt.subplots(2,1,figsize=(10,9),sharex=True,gridspec_kw=dict(height_ratios=[3,1.4]))
show=[('baseline (no stop)','black'),('from-entry 10% same-close','tab:red'),('from-entry 10% next-open','tab:orange')]
if best['variant'] not in [s for s,_ in show]: show.append((best['variant'],'tab:blue'))
qd=[dt.date.fromisoformat(d) for d in sorted(P) if A<=d<=B]; qv=[P[d.isoformat()]/P[A] for d in qd]
a1.plot(qd,qv,color='gray',lw=1,label='QQQ (price)')
for nm,col in show:
    r=R[nm]; d=[dt.date.fromisoformat(x) for x in r['ddates']]; v=r['dnav']
    s=stats(r); a1.plot(d,v,color=col,lw=1.4,label='%s  CAGR %.1f%% MDD %.1f%%'%(nm,s['CAGR']*100,s['MDD_daily']*100))
    pk=0; ddv=[]
    for x in v: pk=max(pk,x); ddv.append(x/pk-1)
    a2.plot(d,[x*100 for x in ddv],color=col,lw=1)
a1.set_yscale('log'); a1.set_ylabel('NAV (log)'); a1.legend(fontsize=9,loc='upper left'); a1.grid(alpha=.3)
a1.set_title('NDX100 6m-mean Top3, 25bp/side: baseline vs stop/buy-back variants (2015-01 to 2026-11-13)',fontsize=11)
a2.set_ylabel('Drawdown %'); a2.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT,'charts/stop_nav_drawdown.png'),dpi=130)
