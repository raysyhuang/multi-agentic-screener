#!/usr/bin/env python3
"""Claude: export the monthly holdings table (CSV) and the data block used by the page.  Usage: python3 make_page_data.py"""
import os,json,csv
H0=os.path.dirname(os.path.abspath(__file__)); exec(open(os.path.join(H0,'stop_rebuy_sim.py')).read().split("if __name__=='__main__':")[0].replace("HERE=os.path.dirname(os.path.abspath(__file__))","HERE=H0"))
i0=IDX['2014-12-31']; n=IDX[ME[-1]]-i0
lg=[]; d0=[]; base=sim(i0,n,log=lg,daily=d0); ls=[]; d1=[]; stp=sim(i0,n,X=0.10,log=ls,daily=d1)
def qq(d):
    b=bars('QQQ'); return b[d][3]
q0=qq(ME[i0]); rows=[]; prev=set()
w=csv.writer(open(os.path.join(H0,'holdings_by_month.csv'),'w',newline='')); w.writerow(['signal_date','hold_until','pick1','pick2','pick3','ret1','ret2','ret3','portfolio_return_after_cost','nav','nav_with_10pct_stop_and_rebuy','qqq_indexed','new_names'])
for x,y in zip(lg,ls):
    new=[t for t in x['picks'] if t not in prev]; prev=set(x['picks'])
    r=dict(s=x['signal'],e=x['end'],p=x['picks'],r=[round(v*100,1) for v in x['stock_ret']],pr=round(x['port_ret']*100,1),nav=round(x['nav'],4),navs=round(y['nav'],4),q=round(qq(x['end'])/q0,4),new=new,sr=round(y['port_ret']*100,1))
    rows.append(r); w.writerow([r['s'],r['e'],*r['p'],*r['r'],r['pr'],r['nav'],r['navs'],r['q'],' '.join(new)])
# daily drawdown series (weekly sampled) for base and stop
def dd(d):
    pk=1;o=[]
    for i,(dt_,v) in enumerate(d):
        pk=max(pk,v)
        if i%5==0 or i==len(d)-1: o.append([dt_,round((v/pk-1)*100,1)])
    return o
res=list(csv.DictReader(open(os.path.join(H0,'stop_rebuy_results.csv'))))
json.dump(dict(rows=rows,dd=dd(d0),dds=dd(d1),res=res,base=dict(cagr=base['CAGR'],mdd=base['daily_MDD'],mult=base['mult']),stop=dict(cagr=stp['CAGR'],mdd=stp['daily_MDD'],mult=stp['mult'],stops=stp['ev']['stops'],rebuys=stp['ev']['rebuys'])),open(os.path.join(H0,'page_data.json'),'w'),ensure_ascii=False,separators=(',',':'))
import collections
c=collections.Counter(t for r in rows for t in r['p']); print(len(rows),'months',len(c),'distinct tickers',c.most_common(12)); print('changes/month avg',sum(len(r['new']) for r in rows[1:])/(len(rows)-1)); print(base['CAGR'],base['daily_MDD'],stp['CAGR'],stp['daily_MDD'],rows[-1])
