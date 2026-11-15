#!/usr/bin/env python3
"""Claude: (1) does requiring a name to STAY in the ranking help?  (2) the Top3 rule against the market (QQQ) over the same months.
Nasdaq-100, score = mean of the last 6 monthly returns, month-end close, 25 bp one-way, price only. My engine + membership, Queen v1.1 prices;
QQQ month-end closes from Queen's raw Massive daily files (QQQQ before 2011-03).
Variants
  base        Top3, rebalanced to equal weight every month
  confirm     hold a name only if it is in the Top3 this month AND was in the Top3 last month; empty slots stay in cash
  confirm+QQQ same, empty slots hold QQQ
  confirm5    in the Top3 now and in the Top5 last month; empty slots cash
  sticky5/10  keep a held name while it stays in the Top5 / Top10; fill free slots with the best-ranked names not held
  skip-new    Top3, but a brand-new entrant gets half weight in its first month (other half cash)
Usage: python3 persistence_and_market.py"""
import os,csv,json,math,datetime as dt,statistics as st,bisect,collections
H0=os.path.dirname(os.path.abspath(__file__)); S=os.path.join(H0,'..','ndx-top3-holdings-and-stop-20261115')
exec(open(os.path.join(S,'stop_rebuy_sim.py')).read().split("def sim(")[0].replace("HERE=os.path.dirname(os.path.abspath(__file__))","HERE=S"))
qb=dict(bars('QQQQ')); qb.update(bars('QQQ')); qd=sorted(qb); QM=[qb[qd[bisect.bisect_right(qd,d)-1]][3] for d in ME]
R=Run()
def rr(t,m): return QM[m+1]/QM[m]-1 if t=='QQQ' else ret(t,ME[m],ME[m+1])
def run(i0,n,rule,R=R):
    hold={}; cash=1.0; eq=[1.0]; prev_top=[]; held=[]; to=0.0; expo=[]; rets=[]
    for m in range(i0,i0+n):
        rk=[t for t,_ in R.rank(m,'M6')]; top3=rk[:3]; w={}
        last3=[t for t,_ in R.rank(m-1,'M6')[:3]] if m>0 else []; last5=[t for t,_ in R.rank(m-1,'M6')[:5]] if m>0 else []
        if rule=='base': w={t:1/3 for t in top3}
        elif rule in('confirm','confirm+QQQ'):
            w={t:1/3 for t in top3 if t in last3}
            if rule=='confirm+QQQ' and len(w)<3: w['QQQ']=1-len(w)/3
        elif rule=='confirm5': w={t:1/3 for t in top3 if t in last5}
        elif rule.startswith('sticky'):
            K=int(rule[6:]); keep=[t for t in held if t in rk[:K]]
            for t in rk:
                if len(keep)>=3: break
                if t not in keep: keep.append(t)
            held=keep; w={t:1/3 for t in keep}
        elif rule=='skip-new': w={t:(1/3 if t in last3 else 1/6) for t in top3}
        E0=cash+sum(hold.values()); Ep=E0
        for _ in range(200):
            new=E0-C*(sum(abs(Ep*x-hold.get(t,0.0)) for t,x in w.items())+sum(v for t,v in hold.items() if t not in w))
            if abs(new-Ep)<1e-14: break
            Ep=new
        to+=(sum(abs(Ep*x-hold.get(t,0.0)) for t,x in w.items())+sum(v for t,v in hold.items() if t not in w))/E0/2
        hold={t:Ep*x for t,x in w.items()}; cash=Ep*(1-sum(w.values())); expo.append(sum(w.values()))
        for t in list(hold):
            r=rr(t,m)
            if r is None: cash+=hold.pop(t)
            else: hold[t]*=1+r
        eq.append(cash+sum(hold.values())); rets.append(eq[-1]/eq[-2]-1)
    pk=0;mdd=0
    for v in eq: pk=max(pk,v); mdd=min(mdd,v/pk-1)
    yrs=(dt.date.fromisoformat(ME[i0+n])-dt.date.fromisoformat(ME[i0])).days/365.25
    return dict(cagr=eq[-1]**(1/yrs)-1,mdd=mdd,to=to/yrs,expo=st.mean(expo),rets=rets,eq=eq,yrs=yrs)
out=[]; PER=(('2015 → 2026-11-13','2014-12-31',ME[-1]),('2007 → 2014','2006-12-29','2014-12-31'))
print('=== 1. 要求「留在榜上」 ===   年化 / 月末最大回撤 / 平均仓位 / 年换手')
for lab,a,b in PER:
    i0=IDX[a]; n=IDX[b]-i0; q=[QM[m+1]/QM[m]-1 for m in range(i0,i0+n)]; yrs=(dt.date.fromisoformat(b)-dt.date.fromisoformat(a)).days/365.25
    qe=[1.0]
    for x in q: qe.append(qe[-1]*(1+x))
    pk=0;qm=0
    for v in qe: pk=max(pk,v); qm=min(qm,v/pk-1)
    print(f'-- {lab}   QQQ: {(qe[-1]**(1/yrs)-1)*100:.1f}% / {qm*100:.0f}%')
    for rule,name in (('base','原规则 前3名'),('confirm','连续两个月在前3才买, 空位放现金'),('confirm+QQQ','连续两个月在前3才买, 空位放QQQ'),('confirm5','本月前3且上月前5才买, 空位放现金'),('sticky5','买入后只要还在前5就不换'),('sticky10','买入后只要还在前10就不换'),('skip-new','新进的第一个月只买一半')):
        r=run(i0,n,rule); print(f'   {name:30s} {r["cagr"]*100:6.1f}% / {r["mdd"]*100:4.0f}% / {r["expo"]*100:4.0f}% / {r["to"]*100:5.0f}%'); out.append(dict(period=lab,rule=name,CAGR=round(r['cagr'],4),monthend_MDD=round(r['mdd'],4),avg_exposure=round(r['expo'],3),turnover_per_year=round(r['to'],2)))
print('-- 稳健性: 「买入后只要还在前K名就不换」, K 从 3 (=原规则) 到 20;  年化 / 月末回撤 / 年换手')
rob=[]
for lab,a,b in PER:
    i0=IDX[a]; n=IDX[b]-i0
    for tag,RR in (('全部',R),('去掉NVDA',Run(excl=('NVDA',)))):
        line=[]
        for K in (3,4,5,6,8,10,15,20):
            r=run(i0,n,'sticky%d'%K,RR); line.append(f'K{K}: {r["cagr"]*100:.1f}%/{r["mdd"]*100:.0f}%/{r["to"]*100:.0f}%'); rob.append(dict(period=lab,universe=tag,K=K,CAGR=round(r['cagr'],4),monthend_MDD=round(r['mdd'],4),turnover_per_year=round(r['to'],2)))
        print(f'   {lab} {tag}: '+'  '.join(line))
w=csv.DictWriter(open(os.path.join(H0,'sticky_robustness.csv'),'w',newline=''),fieldnames=list(rob[0])); w.writeheader(); w.writerows(rob)
w=csv.DictWriter(open(os.path.join(H0,'persistence_results.csv'),'w',newline=''),fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print('\n=== 2. 对比同期大盘 (QQQ, 只算价格) ===')
mk=[]; yearly=[]
for lab,a,b in PER:
    i0=IDX[a]; n=IDX[b]-i0; r=run(i0,n,'base')['rets']; q=[QM[m+1]/QM[m]-1 for m in range(i0,i0+n)]
    mq=st.mean(q); mr=st.mean(r); beta=sum((x-mq)*(y-mr) for x,y in zip(q,r))/sum((x-mq)**2 for x in q); corr=beta*st.pstdev(q)/st.pstdev(r); alpha=(mr-beta*mq)*12
    up=[(x,y) for x,y in zip(q,r) if x>0]; dn=[(x,y) for x,y in zip(q,r) if x<=0]
    res=[y-beta*x for x,y in zip(q,r)]; ta=(st.mean(res)-0)/ (st.stdev(res)/math.sqrt(n))
    print(f'-- {lab}: 相关系数 {corr:.2f}, beta {beta:.2f}, 月波动 规则 {st.pstdev(r)*100:.1f}% / QQQ {st.pstdev(q)*100:.1f}%')
    print(f'   QQQ 上涨的 {len(up)} 个月: 规则平均 {st.mean(y for _,y in up)*100:+.1f}%, QQQ {st.mean(x for x,_ in up)*100:+.1f}%  | QQQ 下跌的 {len(dn)} 个月: 规则平均 {st.mean(y for _,y in dn)*100:+.1f}%, QQQ {st.mean(x for x,_ in dn)*100:+.1f}%')
    print(f'   跑赢 QQQ 的月份 {sum(y>x for x,y in zip(q,r))}/{n}; 扣掉大盘影响后的超额 (alpha) {alpha*100:+.1f}%/年, t={ta:.2f}')
    worst=sorted(zip(q,r,range(n)))[:5]; print('   QQQ 最差的 5 个月:',[(ME[i0+i+1][:7],'QQQ %+.0f%%'%(x*100),'规则 %+.0f%%'%(y*100)) for x,y,i in worst])
    mk.append(dict(period=lab,corr=round(corr,3),beta=round(beta,3),alpha_per_year=round(alpha,4),alpha_t=round(ta,2),up_months=len(up),rule_in_up=round(st.mean(y for _,y in up),4),qqq_in_up=round(st.mean(x for x,_ in up),4),down_months=len(dn),rule_in_down=round(st.mean(y for _,y in dn),4),qqq_in_down=round(st.mean(x for x,_ in dn),4),months_beating=sum(y>x for x,y in zip(q,r)),months=n))
    ys=collections.OrderedDict()
    for j in range(n): ys.setdefault(ME[i0+j+1][:4],[]).append((r[j],q[j]))
    for y,v in ys.items():
        a_=math.prod(1+x for x,_ in v)-1; b_=math.prod(1+x for _,x in v)-1; yearly.append(dict(year=y,rule=round(a_,4),qqq=round(b_,4)))
    print('   逐年 (规则 / QQQ):',' '.join(f'{y}:{(math.prod(1+x for x,_ in v)-1)*100:+.0f}/{(math.prod(1+x for _,x in v)-1)*100:+.0f}' for y,v in ys.items()))
    print(f'   规则跑赢 QQQ 的年份: {sum(1 for y,v in ys.items() if math.prod(1+x for x,_ in v)>math.prod(1+x for _,x in v))}/{len(ys)}')
w=csv.DictWriter(open(os.path.join(H0,'market_comparison.csv'),'w',newline=''),fieldnames=list(mk[0])); w.writeheader(); w.writerows(mk)
w=csv.DictWriter(open(os.path.join(H0,'yearly_rule_vs_qqq.csv'),'w',newline=''),fieldnames=list(yearly[0])); w.writeheader(); w.writerows(yearly)
