"""Part 1: holdings ledger, monthly summary, charts, ticker stats for the baseline post rule (engine run(), unmodified)."""
from sim import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.colors as mcolors, datetime as dt, numpy as np
res=run(K,i0,i1,(),COST); st0=stats(res)
P,TR=qqq()
led=[]; summ=[]; prevset=set(); nav=1.0; pk=1.0; tick={}
for i,r in zip(range(i0,i1),res['rec']):
    picks=ranking(i)[:K]; partial=ME[i+1]=='2026-11-13'
    for rank,(s,t,key) in enumerate(picks,1):
        a=PX.at(key,ME[i]); hr=PX.hold_ret(key,ME[i],ME[i+1])
        exitpx=a[0]*(1+hr[0])
        led.append(dict(signal_date=ME[i],holding_month=ME[i+1][:7]+(' (partial to 2026-11-13)' if partial else ''),exit_date=hr[1],rank=rank,ticker=t,
             score_6m_mean=round(s,6),entry_close_adj=round(a[0],4),exit_close_adj=round(exitpx,4),month_return=round(hr[0],6)))
        T=tick.setdefault(t,dict(months=0,arith=0.0,logc=0.0,first=ME[i+1][:7],last=None,rets=[]))
        T['months']+=1; T['arith']+=hr[0]/K; T['last']=ME[i+1][:7]; T['rets'].append(hr[0])
        G=r['gross']
        T['logc']+=(hr[0]/K)/G*math.log(1+G) if abs(G)>1e-12 else hr[0]/K
    cur=set(r['picks'].split()); changed=len(cur-prevset) if prevset else K; prevset=cur
    q=P[ME[i+1]]/P[ME[i]]-1
    nav*=1+r['net']; pk=max(pk,nav)
    summ.append(dict(signal_date=ME[i],holding_month=ME[i+1][:7]+('*' if partial else ''),picks=r['picks'],basket_gross=round(r['gross'],6),cost=round(r['cost'],6),
        basket_net=round(r['net'],6),qqq_price_ret=round(q,6),excess_vs_qqq=round(r['net']-q,6),nav=round(nav,6),qqq_nav=round(P[ME[i+1]]/P[A],6),
        drawdown_monthend=round(nav/pk-1,6),names_changed=changed,oneway_turnover=round(r['traded']/2,4)))
assert abs(nav-st0['total'])<1e-9
def wcsv(fn,rs):
    with open(os.path.join(OUT,fn),'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0])); w.writeheader(); w.writerows(rs)
wcsv('holdings_ledger_2015_2026.csv',led); wcsv('monthly_summary_2015_2026.csv',summ)
trows=[dict(ticker=t,months_held=v['months'],first_month=v['first'],last_month=v['last'],avg_month_ret=round(sum(v['rets'])/len(v['rets']),6),
    hit_rate=round(sum(x>0 for x in v['rets'])/len(v['rets']),4),contrib_arith_sum_w_r=round(v['arith'],6),contrib_log=round(v['logc'],6)) for t,v in tick.items()]
trows.sort(key=lambda x:-x['contrib_log']); wcsv('ticker_stats.csv',trows)
# ---- prints
nm=len(summ)
print('baseline engine: CAGR %.2f%% MDDd %.1f%% MDDm %.1f%% x%.2f months %d'%(st0['CAGR']*100,st0['MDD_daily']*100,st0['MDD_monthly']*100,st0['total'],nm))
ch=[s['names_changed'] for s in summ[1:]]
print('names changed per month (excl. first): avg %.2f of 3; dist 0:%d 1:%d 2:%d 3:%d; avg one-way turnover %.1f%%/month'%(sum(ch)/len(ch),ch.count(0),ch.count(1),ch.count(2),ch.count(3),100*sum(s['oneway_turnover'] for s in summ[1:])/len(ch)))
print('unique tickers held:',len(trows),' total ticker-months:',len(led))
print('\nMost held:'); 
for r in sorted(trows,key=lambda x:(-x['months_held'],x['ticker']))[:15]: print('  %-6s %3d months  avg %+.1f%%  hit %.0f%%  logcontrib %+.3f'%(r['ticker'],r['months_held'],r['avg_month_ret']*100,r['hit_rate']*100,r['contrib_log']))
print('\nTop 10 contributors (log attribution; sum of all = ln(final gross NAV)):')
for r in trows[:10]: print('  %-6s %3dm  log %+.3f  arith sum(w*r) %+.3f'%(r['ticker'],r['months_held'],r['contrib_log'],r['contrib_arith_sum_w_r']))
print('Bottom 10:')
for r in trows[-10:][::-1]: print('  %-6s %3dm  log %+.3f  arith sum(w*r) %+.3f'%(r['ticker'],r['months_held'],r['contrib_log'],r['contrib_arith_sum_w_r']))
print('sum log contrib %.4f ; ln(gross NAV) %.4f ; total cost drag log %.4f'%(sum(r['contrib_log'] for r in trows),sum(math.log(1+s['basket_gross']) for s in summ),sum(math.log(1+s['basket_net'])-math.log(1+s['basket_gross']) for s in summ)))
wins=sum(s['basket_net']>s['qqq_price_ret'] for s in summ); print('months beating QQQ: %d/%d'%(wins,nm))
# yearly
yr={}
for s in summ:
    y=s['holding_month'][:4]; a=yr.setdefault(y,[1,1]); a[0]*=1+s['basket_net']; a[1]*=1+s['qqq_price_ret']
yrows=[dict(year=y,strategy=round(v[0]-1,6),qqq_price=round(v[1]-1,6)) for y,v in sorted(yr.items())]; wcsv('yearly_returns.csv',yrows)
print('\nYear  strategy  QQQ'); 
for r in yrows: print(' %s %+8.1f%% %+7.1f%%'%(r['year'],r['strategy']*100,r['qqq_price']*100))
# ---- charts
CH=os.path.join(OUT,'charts'); plt.rcParams.update({'font.size':11})
dd_=[dt.date.fromisoformat(d) for d in [ME[i0]]+[x for i in range(i0,i1) for x in days_between(ME[i],ME[i+1])]]
dn=res['dnav']; qd=[P[d.isoformat()]/P[A] for d in dd_]
fig,(a1,a2)=plt.subplots(2,1,figsize=(10,9),sharex=True,gridspec_kw=dict(height_ratios=[3,1.3]))
a1.plot(dd_,dn,color='tab:red',lw=1.5,label='6m-mean Top3 (25bp/side)  CAGR %.1f%%, x%.1f'%(st0['CAGR']*100,st0['total']))
a1.plot(dd_,qd,color='gray',lw=1.2,label='QQQ price  CAGR %.1f%%'%(stats(bench(P,i0,i1))['CAGR']*100))
a1.set_yscale('log'); a1.legend(loc='upper left'); a1.grid(alpha=.3,which='both'); a1.set_ylabel('NAV (log, start=1)')
a1.set_title('NDX100 PIT, 6-month-mean momentum Top3, monthly\n2015-01 to 2026-11-13 (daily closes)')
def ddser(v):
    p=0; o=[]
    for x in v: p=max(p,x); o.append((x/p-1)*100)
    return o
a2.fill_between(dd_,ddser(dn),0,color='tab:red',alpha=.4,label='strategy DD (max %.1f%%)'%(st0['MDD_daily']*100))
a2.plot(dd_,ddser(qd),color='gray',lw=1,label='QQQ DD'); a2.legend(loc='lower left',fontsize=9); a2.grid(alpha=.3); a2.set_ylabel('Drawdown %')
plt.tight_layout(); plt.savefig(os.path.join(CH,'a_nav_log_vs_qqq_drawdown.png'),dpi=130); plt.close()
# (b) monthly bars
fig,ax=plt.subplots(figsize=(14,6)); x=np.arange(nm)
ax.bar(x-0.2,[s['basket_net']*100 for s in summ],0.4,color=['tab:green' if s['basket_net']>=0 else 'tab:red' for s in summ],label='Top3 net')
ax.bar(x+0.2,[s['qqq_price_ret']*100 for s in summ],0.4,color='gray',alpha=.7,label='QQQ')
lab=[s['holding_month'] for s in summ]; ax.set_xticks(x[::6]); ax.set_xticklabels(lab[::6],rotation=60,fontsize=9)
ax.axhline(0,color='k',lw=.6); ax.set_ylabel('Monthly return %'); ax.legend(); ax.grid(axis='y',alpha=.3)
ax.set_title('Monthly return: Top3 basket (net) vs QQQ, 2015-01 to 2026-11 (Nov partial)')
plt.tight_layout(); plt.savefig(os.path.join(CH,'b_monthly_returns_vs_qqq.png'),dpi=130); plt.close()
# (c) heatmap timeline
order=sorted(tick,key=lambda t:(tick[t]['first'],t))
M=np.full((len(order),nm),np.nan)
for r in led:
    j=[s['signal_date'] for s in summ].index(r['signal_date']); M[order.index(r['ticker']),j]=r['month_return']*100
fig,ax=plt.subplots(figsize=(16,max(8,len(order)*0.17)))
cm=plt.get_cmap('RdYlGn').copy(); cm.set_bad('white')
im=ax.imshow(np.ma.masked_invalid(M),aspect='auto',cmap=cm,norm=mcolors.TwoSlopeNorm(vmin=-30,vcenter=0,vmax=30),interpolation='nearest')
ax.set_yticks(range(len(order))); ax.set_yticklabels(['%s (%d)'%(t,tick[t]['months']) for t in order],fontsize=7)
ax.set_xticks(range(0,nm,6)); ax.set_xticklabels(lab[::6],rotation=60,fontsize=8)
for j in range(nm): ax.axvline(j-0.5,color='#eee',lw=.3)
cb=plt.colorbar(im,ax=ax,fraction=.02,pad=.01); cb.set_label('month return % (clipped at +/-30)')
ax.set_title('Holdings timeline: which 3 NDX100 names were held each month (color = that month\'s return); (n) = months held')
plt.tight_layout(); plt.savefig(os.path.join(CH,'c_holdings_heatmap.png'),dpi=120); plt.close()
# (d) yearly
fig,ax=plt.subplots(figsize=(10,6)); x=np.arange(len(yrows))
b1=ax.bar(x-0.2,[r['strategy']*100 for r in yrows],0.4,color='tab:red',label='Top3 net'); b2=ax.bar(x+0.2,[r['qqq_price']*100 for r in yrows],0.4,color='gray',label='QQQ')
for b in list(b1)+list(b2): ax.annotate('%.0f'%b.get_height(),(b.get_x()+b.get_width()/2,b.get_height()),ha='center',va='bottom' if b.get_height()>=0 else 'top',fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([r['year']+('*' if r['year']=='2026' else '') for r in yrows]); ax.axhline(0,color='k',lw=.6)
ax.set_ylabel('Calendar-year return %'); ax.legend(); ax.grid(axis='y',alpha=.3); ax.set_title('Per-year return vs QQQ (*2026 to Nov 13)')
plt.tight_layout(); plt.savefig(os.path.join(CH,'d_yearly_returns_vs_qqq.png'),dpi=130); plt.close()
# (e) top/bottom contributors bar
sel=trows[:10]+trows[-10:]
fig,ax=plt.subplots(figsize=(9,8)); ax.barh(range(len(sel)),[r['contrib_log'] for r in sel],color=['tab:green' if r['contrib_log']>0 else 'tab:red' for r in sel])
ax.set_yticks(range(len(sel))); ax.set_yticklabels(['%s (%dm)'%(r['ticker'],r['months_held']) for r in sel]); ax.invert_yaxis(); ax.grid(axis='x',alpha=.3)
ax.set_xlabel('log contribution (sum of all names = ln of gross NAV)'); ax.set_title('Top 10 and bottom 10 contributors, 2015-01 to 2026-11-13')
plt.tight_layout(); plt.savefig(os.path.join(CH,'e_top_bottom_contributors.png'),dpi=130); plt.close()
