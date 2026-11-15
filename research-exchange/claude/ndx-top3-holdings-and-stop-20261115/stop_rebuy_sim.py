#!/usr/bin/env python3
"""Claude: (1) every monthly holding of the Nasdaq-100 "mean of 6 monthly returns, Top3, monthly" rule, 2014-12-31 -> 2026-11-13;
(2) Ray's idea: sell a holding when it falls 10% below its entry price, stay out while it keeps falling, buy it back when it climbs back
to the price it was sold at.  Daily simulation with Massive daily OHLC bars (Queen's raw/adj files, vendor split-adjusted; the spin-offs Queen
corrected are applied here too).  Picks: my engine + my membership export + Queen v1.1 month-end prices (same as the verified runs).

Order model ("orders"): a stop-sell order rests at entry x (1 - X).  If the day opens below it, it fills at the open (gap), otherwise at the
stop price when the low touches it.  After a sale a buy-stop order rests at the sale price: fills at the open if the day opens above it,
otherwise at that price when the high touches it.  The new entry price becomes the buy-back price.  No same-day round trip.
"close" model: decisions and fills at the daily close only.
Month-end: rebalance to the new Top3 at the close, equal weight; a name that is stopped out is bought again if it is still a pick ("reset"),
or stays out until its buy-back price is hit ("carry").  25 bp one-way on every trade, cash earns 0, price only.
Usage: python3 stop_rebuy_sim.py"""
import os,csv,json,datetime as dt,bisect,collections,statistics as st,math
HERE=os.path.dirname(os.path.abspath(__file__)); RV=os.path.join(HERE,'..','queen-ndx100-fill-review-20261114'); V=os.path.join(HERE,'..','..','queen','ndx100-fill-20261114-v1.1')
exec(open(os.path.join(RV,'verify_queen_ndx.py')).read().split("P=lambda x:")[0].replace("H=os.path.dirname(os.path.abspath(__file__))","H=RV").replace("Q=os.path.join(H,'..','..','queen','ndx100-fill-20261114')","Q=V"))
FACTORS=[('WDC','2025-02-24',1/1.32335),('ADP','2007-04-02',0.8993),('ADP','2014-10-01',0.8736),('EXC','2022-02-02',0.7081),('MAR','2011-11-22',0.9504),('HON','2016-10-03',0.9944),('HON','2018-10-29',0.9704),('BIIB','2017-02-02',264.23/(264.23+0.5*44.98))]
C=0.0025; BARS={}
def bars(t):
    if t not in BARS:
        b={}; p=os.path.join(V,'raw','adj',t+'.csv')
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                d=(dt.datetime.fromtimestamp(int(r['t'])/1000,dt.timezone.utc)-dt.timedelta(hours=4)).date().isoformat(); b[d]=[float(r['o']),float(r['h']),float(r['l']),float(r['c'])]
            for tt,ex,f in FACTORS:
                if tt==t:
                    for d in b:
                        if d<ex: b[d]=[x*f for x in b[d]]
        if t in SUCC and b:   # renamed mid-month (e.g. WFMI -> WFM): continue with the successor's bars after this symbol's last day
            last=max(b); sb=bars(SUCC[t]); cm=sorted(set(px[t])&set(px[SUCC[t]])); k=(px[t][cm[-1]]/px[SUCC[t]][cm[-1]]) if cm else 1.0   # put the successor on this symbol's price basis (month-end file ratio)
            for d,x in sb.items():
                if d>last: b[d]=[v*k for v in x]
        BARS[t]=b
    return BARS[t]
CAL=sorted(set(bars('QQQ'))|set(bars('QQQQ')))
def sim(i0,n,X=None,mode='orders',carry=False,rebuy=True,K=3,log=None,daily=None):
    R=Run(); cash=1.0; slots={}   # t -> dict(sh, state, ref, sellp, cash)
    nav=[]; ev=collections.Counter(); peak=1.0; mdd=0.0
    for m in range(i0,i0+n):
        d0=ME[m]; picks=[t for t,_ in R.rank(m,'M6')[:K]]
        # value everything at the close of d0
        val={}; idle=cash
        for t,s in slots.items():
            if s['state'] in ('in','monthly'): val[t]=s['sh']*s['px']
            else: idle+=s['cash']
        E0=idle+sum(val.values())
        keep_out={t for t,s in slots.items() if carry and s['state']=='out' and t in picks}
        buy=[t for t in picks if t not in keep_out]
        # stopped-out carried names keep their cash share (1/K of equity is reserved for them)
        Ep=E0
        for _ in range(200):
            T=Ep/K; new=E0-C*(sum(abs(T-val.get(t,0.0)) for t in buy)+sum(v for t,v in val.items() if t not in buy))
            if abs(new-Ep)<1e-14: break
            Ep=new
        ns={}
        for t in picks:
            b=bars(t).get(d0)
            if t in keep_out: s=slots[t]; ns[t]=dict(sh=0,state='out',ref=s['ref'],sellp=s['sellp'],cash=Ep/K,px=s['px'])
            elif b is None: ns[t]=dict(sh=Ep/K,state='monthly',ref=None,sellp=None,cash=0,px=1.0); ev['monthly_only']+=1
            else: ns[t]=dict(sh=Ep/K/b[3],state='in',ref=b[3],sellp=None,cash=0,px=b[3])
        slots=ns; cash=0.0; d1=ME[m+1]; days=CAL[bisect.bisect_right(CAL,d0):bisect.bisect_right(CAL,d1)]
        start_val={t:(s['sh']*s['px'] if s['state'] in('in','monthly') else s['cash']) for t,s in slots.items()}
        for d in days:
            tot=cash
            for t,s in slots.items():
                if s['state']=='monthly':
                    if d==days[-1]:
                        r=ret(t,d0,d1); s['px']=1.0+(r if r is not None else 0.0)
                    tot+=s['sh']*s['px']; continue
                b=bars(t).get(d)
                if b is None:
                    if s['state']=='in' and not any(x>d and x<=d1 for x in bars(t)):   # series ended: cash out at last close
                        s['cash']=s['sh']*s['px']; s['state']='gone'; ev['delisted_cash_out']+=1
                    tot+=s['sh']*s['px'] if s['state']=='in' else s['cash']; continue
                o,h,l,c=b
                if s['state']=='in':
                    if X is not None:
                        S=s['ref']*(1-X); fill=None
                        if mode=='orders':
                            if o<=S: fill=o; ev['stop_gap']+=1
                            elif l<=S: fill=S
                        elif c<=S: fill=c
                        if fill is not None:
                            s['cash']=s['sh']*fill*(1-C); s['state']='out'; s['sellp']=fill; ev['stops']+=1; ev['loss_at_stop']+=fill/s['ref']-1
                    if s['state']=='in': s['px']=c
                elif s['state']=='out' and rebuy:
                    fill=None
                    if mode=='orders':
                        if o>=s['sellp']: fill=o
                        elif h>=s['sellp']: fill=s['sellp']
                    elif c>=s['sellp']: fill=c
                    if fill is not None:
                        s['sh']=s['cash']*(1-C)/fill; s['state']='in'; s['ref']=fill; s['px']=c; ev['rebuys']+=1; ev['rebuy_slip']+=fill/s['sellp']-1
                    else: s['px']=c
                tot+=s['sh']*s['px'] if s['state']=='in' else s['cash']
            nav.append((d,tot)); peak=max(peak,tot); mdd=min(mdd,tot/peak-1)
        if log is not None:
            endv={t:(s['sh']*s['px'] if s['state'] in('in','monthly') else s['cash']) for t,s in slots.items()}
            log.append(dict(signal=d0,end=d1,picks=picks,stock_ret=[endv[t]/start_val[t]-1 for t in picks],port_ret=(nav[-1][1]/E0-1) if nav else 0,nav=nav[-1][1],scores=[v for _,v in R.rank(m,'M6')[:K]]))
    # final liquidation
    tot=cash+sum((s['sh']*s['px']*(1-C) if s['state'] in('in','monthly') else s['cash']) for s in slots.values())
    yrs=(dt.date.fromisoformat(ME[i0+n])-dt.date.fromisoformat(ME[i0])).days/365.25
    if daily is not None: daily.extend(nav)
    # month-end MDD
    me=[1.0]+[v for d,v in nav if d in IDX]; pk=0;mm=0
    for v in me: pk=max(pk,v); mm=min(mm,v/pk-1)
    return dict(CAGR=tot**(1/yrs)-1,mult=tot,daily_MDD=mdd,monthend_MDD=mm,ev=ev,months=n)
if __name__=='__main__':
    out=[]
    for lab,a,b in (('2015 → 2026-11-13','2014-12-31',ME[-1]),('2007 → 2014','2006-12-29','2014-12-31')):
        i0=IDX[a]; n=IDX[b]-i0; print(f'== {lab} ({n} 个月)   年化 / 按日最大回撤 / 月末最大回撤 / 止损次数 / 买回次数')
        tests=[('不止损 (原规则)',dict())]
        for X in (0.05,0.08,0.10,0.15,0.20,0.30): tests.append((f'止损{int(X*100)}% + 回到卖出价买回 (挂单)',dict(X=X)))
        tests+=[('止损10% + 买回, 只看收盘价',dict(X=0.10,mode='close')),('止损10% + 买回, 跨月不重置',dict(X=0.10,carry=True)),('止损10%, 不买回 (等月底)',dict(X=0.10,rebuy=False)),('止损20%, 不买回 (等月底)',dict(X=0.20,rebuy=False))]
        for name,kw in tests:
            r=sim(i0,n,**kw); e=r['ev']
            print(f'  {name:34s} {r["CAGR"]*100:6.1f}% / {r["daily_MDD"]*100:5.1f}% / {r["monthend_MDD"]*100:5.1f}% / {e["stops"]:4d} / {e["rebuys"]:4d}'+(f'  | 跳空成交 {e["stop_gap"]}, 平均止损亏损 {e["loss_at_stop"]/e["stops"]*100:.1f}%, 买回平均比卖出价高 {e["rebuy_slip"]/max(1,e["rebuys"])*100:.1f}%' if e['stops'] else f'  | 月度数据持仓 {e["monthly_only"]}, 退市转现金 {e["delisted_cash_out"]}'))
            out.append(dict(period=lab,test=name,CAGR=round(r['CAGR'],4),multiple=round(r['mult'],2),daily_MDD=round(r['daily_MDD'],4),monthend_MDD=round(r['monthend_MDD'],4),stops=e['stops'],rebuys=e['rebuys'],gap_fills=e['stop_gap']))
    w=csv.DictWriter(open(os.path.join(HERE,'stop_rebuy_results.csv'),'w',newline=''),fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
