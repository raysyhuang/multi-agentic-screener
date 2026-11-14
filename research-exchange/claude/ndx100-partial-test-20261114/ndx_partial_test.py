#!/usr/bin/env python3
"""Claude: the post's rule on Nasdaq-100 point-in-time members -- PARTIAL test.
Membership: ndx100_membership_month_end.csv (from jmccarrell/n100tickers). Prices: the frozen S&P 500 pack's month-end rank prices,
which cover only ~85% of Nasdaq-100 member-months (non-S&P names such as MELI, PDD, MRVL, ZM, PTON, APP, MSTR, ASML have no prices here).
Members without a price are simply left out, so this is NOT a full Nasdaq-100 backtest. Engine: month-end proxy, price only.
Usage: python3 ndx_partial_test.py"""
import os,csv,collections,statistics as st
H=os.path.dirname(os.path.abspath(__file__)); exec(open(os.path.join(H,'..','ndx-top3-post-check-20261114','sp500_analog.py')).read().split("for dec,(P,first)")[0].replace("H=os.path.dirname(os.path.abspath(__file__))","H=os.path.join(H,'..','ndx-top3-post-check-20261114')"))
HERE=os.path.dirname(os.path.abspath(__file__)); ALIAS={'PCLN':'BKNG'}
# pack rank prices that collapse in the month a cash-and-stock takeover closes (ESRX 101.47 -> next month-end far lower; same pattern as APC): cash out at last good price
BAD_TAIL={'ESRX':'2018-11-30','APC':'2019-07-31'}
mem={r['month_end']:[ALIAS.get(t,t) for t in r['tickers'].split()] for r in csv.DictReader(open(os.path.join(HERE,'ndx100_membership_month_end.csv')))}
P,first=PACKS['2015-2025']; out=[]; miss=collections.Counter()
for var,ex,fee in (('all names',(),0.0025),('no NVDA',('NVDA',),0.0025),('all names, zero cost',(),0.0)):
    C=fee; E=E2(P,first,True,ex); cov=[]
    for t,last in BAD_TAIL.items(): E.px[t]={d:v for d,v in E.px[t].items() if d<=last}
    for d in E.ME:
        have=[t for t in mem[d] if d in E.px.get(t,{})]; cov.append(len(have)/len(mem[d])); E.el[d]=have
        if var=='all names' and E.ME[E.i0]<=d<E.ME[E.i0+120]:
            for t in mem[d]:
                if d not in E.px.get(t,{}): miss[t]+=1
    b,_=E.tranche('ALL',None,1,0); bc,bm,by=stats(b)
    print(f'== {var}: covered members per month avg {st.mean(cov[E.i0:E.i0+120])*100:.0f}% | EW covered Nasdaq-100 members: CAGR {bc*100:.1f}% / MDD {bm*100:.0f}%')
    for sig,lab in (('M6avg','post rule (mean of 6 monthly returns)'),('W12','12-month return')):
        for K in (1,3,5,10):
            p,to=E.tranche(sig,K,1,0); c,m,y=stats(p); a,t,d=active_t(p,b)
            print(f'   {lab:38s} Top{K:<2d}: CAGR {c*100:5.1f}% / MDD {m*100:4.0f}% / worst yr {min(y)*100:4.0f}% | vs EW {a*100:+5.1f} pp/yr t={t:+.2f} | worst month {min(p[j+1]/p[j]-1 for j in range(120))*100:+.0f}%')
            out.append(dict(variant=var,signal=sig,K=K,CAGR=c,monthend_MDD=m,worst_year=min(y),active_vs_EW=a,t_stat=t,EW_CAGR=bc))
            if K==3 and sig=='M6avg':
                print('      yearly 2015..2024:',' '.join('%+.0f%%'%(x*100) for x in y),'| EW:',' '.join('%+.0f%%'%(x*100) for x in by))
                if var=='all names':
                    cnt=collections.Counter(t for m_ in range(E.i0,E.i0+120) for t in E.ranking(m_,sig)[:3]); print('      most-held (months of 120):',cnt.most_common(12))
                    for m_ in range(E.i0,E.i0+120):
                        for t in E.ranking(m_,sig)[:3]:
                            r=E.ret(t,E.ME[m_],E.ME[m_+1])
                            if r is not None and abs(r)>0.45: print('      extreme month:',E.ME[m_],t,'%+.0f%%'%(r*100))
w=csv.DictWriter(open(os.path.join(HERE,'ndx_partial_results.csv'),'w',newline=''),fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
w=csv.writer(open(os.path.join(HERE,'missing_prices_request.csv'),'w',newline='')); w.writerow(['ticker','member_months_without_price_2015_2024'])
for t,n in miss.most_common(): w.writerow([t,n])
print('members with no price (2015-2024):',len(miss),'tickers,',sum(miss.values()),'member-months')
