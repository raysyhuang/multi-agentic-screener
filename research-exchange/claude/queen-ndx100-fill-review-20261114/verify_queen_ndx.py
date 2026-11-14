#!/usr/bin/env python3
"""Claude: independent re-run of Queen's Nasdaq-100 results with MY month-end engine, MY membership export and QUEEN's month-end prices.
Inputs: ../../queen/ndx100-fill-20261114/out/prices/queen_rank_prices_month_end_all_members.csv (Queen's corrected split-adjusted closes; no extra
spin factor applied here), ndx100_membership_month_end_2006_2026.csv (my export of jmccarrell/n100tickers @16eaf93).
Engine: signal and trade at the same month-end close, 25 bp one-way on traded value (fixed point), price only, a name with no next month-end price
is cashed out at its last month-end price (renamed tickers continue under the successor symbol), final liquidation charged.
Usage: python3 verify_queen_ndx.py  -> verify_stdout (stdout) + CSVs"""
import os,csv,gzip,collections,statistics as st,sys
H=os.path.dirname(os.path.abspath(__file__)); Q=os.path.join(H,'..','..','queen','ndx100-fill-20261114')
px=collections.defaultdict(dict)
for r in csv.DictReader(open(os.path.join(Q,'out','prices','queen_rank_prices_month_end_all_members.csv'))): px[r['engine_ticker']][r['month_end']]=float(r['rank_price'])
MEM={r['month_end']:r['tickers'].split() for r in csv.DictReader(open(os.path.join(H,'ndx100_membership_month_end_2006_2026.csv')))}
ME=sorted(MEM); IDX={d:i for i,d in enumerate(ME)}
SUCC={'FB':'META','PCLN':'BKNG','CTRP':'TCOM','SYMC':'NLOK','ERTS':'EA','WFMI':'WFM','HANS':'MNST','LINTA':'QVCA','QVCA':'QRTEA','SUNW':'JAVA','KFT':'MDLZ','NWSA':'FOXA','FISV':'FI'}  # ticker renames (facts; same list Queen stitches)
# lookback for renamed tickers: Queen's export gives a successor only part of its predecessor history (e.g. MNST starts 2012-01, HANS ends 2012-01).
# Where both symbols have a price on a common month-end, extend the successor backwards with the predecessor scaled at the latest common date.
STITCHED=[]
ORIG={t:set(v) for t,v in px.items()}  # dates Queen's file really has for each symbol; stitched predecessor history is for lookback only
for p_,s_ in list(SUCC.items()):
    common=sorted(set(px[p_])&set(px[s_]))
    if common:
        k=px[s_][common[-1]]/px[p_][common[-1]]; add=[d for d in px[p_] if d not in px[s_] and d<common[-1]]
        for d in add: px[s_][d]=px[p_][d]*k
        if add: STITCHED.append((p_,s_,common[-1],round(k,4),len(add)))
def ret(t,a,b):
    for s in (t,SUCC.get(t)):
        if s and a in px[s] and b in px[s]: return px[s][b]/px[s][a]-1
    return None
class Run:
    def __init__(s,C=0.0025,excl=(),mask=None):
        s.C=C; s.excl=set(excl); s.mask=mask; s.cache={}
    def univ(s,i):
        d=ME[i]; return [t for t in MEM[d] if d in ORIG.get(t,()) and t not in s.excl and (s.mask is None or t in s.mask.get(d,()))]
    def rank(s,i,sig):
        k=(i,sig)
        if k in s.cache: return s.cache[k]
        out=[]
        for t in s.univ(i):
            if sig=='M6':
                p=[px[t].get(ME[j]) for j in range(i-6,i+1)] if i>=6 else [None]
                if None not in p: out.append((-sum(p[j+1]/p[j]-1 for j in range(6))/6,t))
            else:
                if i>=sig and ME[i-sig] in px[t]: out.append((-(px[t][ME[i]]/px[t][ME[i-sig]]-1),t))
        s.cache[k]=[(t,-v) for v,t in sorted(out)]; return s.cache[k]
    def path(s,sig,K,Hh,start,n,log=None):
        hold={}; cash=1.0; eqs=[1.0]; C=s.C
        for m in range(start,start+n):
            if (m-start)%Hh==0:
                names=[t for t,_ in s.rank(m,sig)[:K]] if K else s.univ(m); E0=cash+sum(hold.values()); Ep=E0
                for _ in range(200):
                    T=Ep/len(names); new=E0-C*(sum(abs(T-hold.get(t,0.0)) for t in names)+sum(v for t,v in hold.items() if t not in names))
                    if abs(new-Ep)<1e-14: break
                    Ep=new
                hold={t:Ep/len(names) for t in names}; cash=0.0
            before=cash+sum(hold.values()); pr={}
            for t in list(hold):
                r=ret(t,ME[m],ME[m+1]); pr[t]=r
                if r is None: cash+=hold.pop(t)
                else: hold[t]*=1+r
            eqs.append(cash+sum(hold.values()))
            if log is not None: log.append((ME[m],ME[m+1],dict(pr),eqs[-1]/eqs[-2]-1))
        eqs[-1]=cash+sum(hold.values())*(1-C)
        pk=0;mdd=0
        for v in eqs: pk=max(pk,v); mdd=min(mdd,v/pk-1)
        return eqs,eqs[-1]**(12/n)-1,mdd
P=lambda x:'%.1f%%'%(x*100)
out=[]
def chk(item,mine,queen,tol=0.006,fmt=P):
    ok=abs(mine-queen)<=tol; out.append(dict(item=item,claude=mine,queen=queen,diff=mine-queen,verdict='一致' if ok else '不一致'))
    print(f'  {"一致  " if ok else "不一致"} {item}: Claude {fmt(mine)} | Queen {fmt(queen)} | 差 {(mine-queen)*100:+.2f}pp')
# ---------- 1. post rule
print('predecessor lookback stitched:',STITCHED)
print('== 1. 帖子规则 (M6 均值, 月度调仓, 25bp, 只算价格; 回撤为月末口径)')
i15=IDX['2014-12-31']; i07=IDX['2006-12-29']
R=Run(); Rn=Run(excl=('NVDA',)); R0=Run(C=0.0)
QN={'2015':{1:(.253,-.649),3:(.277,-.438),5:(.233,-.429),10:(.183,-.391)},'2007':{1:(.079,-.697),3:(.091,-.420),5:(.152,-.535),10:(.107,-.615)}}
logs={}
for lab,i0,n in (('2015',i15,120),('2007',i07,96)):
    for K in (1,3,5,10):
        lg=[]; e,c,m=R.path('M6',K,1,i0,n,lg); logs[(lab,K)]=lg
        chk(f'{lab}起 Top{K} 年化',c,QN[lab][K][0]); chk(f'{lab}起 Top{K} 月末最大回撤',m,QN[lab][K][1],tol=0.012)
    e,c,m=Rn.path('M6',3,1,i0,n); chk(f'{lab}起 Top3 去掉NVDA 年化',c,.184 if lab=='2015' else .108)
    e,c,m=R.path('M6',None,1,i0,n); print(f'     成分股等权(我扣成本): {P(c)}  | Queen 不扣成本 {"11.4%" if lab=="2015" else "10.0%"}'); e,c,m=R0.path('M6',None,1,i0,n); chk(f'{lab}起 成分股等权(不计成本)',c,.114 if lab=='2015' else .100)
e,c,m=R0.path('M6',3,1,i15,120); chk('2015起 Top3 不计成本',c,.311)
for K,q in ((1,.172),(3,.191),(5,.196),(10,.148)):
    e,c,m=R.path('M6',K,1,i07,216); chk(f'2007-2024 全期 Top{K}',c,q)
chk('QQQ 只算价格 2015-2024 (FMP 103.25→511.23)',(511.23/103.25)**0.1-1,.173); chk('QQQ 只算价格 2007-2014 (FMP 43.16→103.25)',(103.25/43.16)**(1/8)-1,.114)
# worst months / top holdings
lg=logs[('2015',3)]; wm=min(lg,key=lambda x:x[3]); print('     Top3 最差月:',wm[0],P(wm[3]),{k:(None if v is None else round(v,3)) for k,v in wm[2].items()})
cnt=collections.Counter(t for x in lg for t in x[2]); print('     Top3 2015-2024 持有月数:',cnt.most_common(8))
w=csv.writer(open(os.path.join(H,'claude_top3_picks_full_universe.csv'),'w',newline='')); w.writerow(['period','signal','end','picks','stock_rets','gross_month_ret'])
for lab in ('2007','2015'):
    for a,b,pr,g in logs[(lab,3)]: w.writerow([lab,a,b,' '.join(pr),' '.join('NA' if v is None else '%+.4f'%v for v in pr.values()),'%.6f'%g])
# compare picks with Queen's picks_by_month
qp={(r['period'],r['signal']):r['picks'].split() for r in csv.DictReader(open(os.path.join(Q,'out','picks_by_month.csv'))) if r['variant']=='Top3'}
dif=[(lab,a,list(pr),qp.get((per,a))) for lab,per in (('2007','2007-2014'),('2015','2015-2024')) for a,b,pr,g in logs[(lab,3)] if qp.get((per,a)) is not None and set(pr)!=set(qp[(per,a)])]
print(f'     Top3 逐月持仓 vs Queen picks_by_month: 比较 {sum(1 for lab,per in (("2007","2007-2014"),("2015","2015-2024")) for x in logs[(lab,3)] if (per,x[0]) in qp)} 个月, 不同 {len(dif)} 个月',dif[:6])
# ---------- 1b. why 38.6 -> 27.7
print('== 1b. 84% 股票池 → 完整股票池')
sys.path.insert(0,H)
B=os.path.join(H,'..','next-steps-20261114'); g={'B':B}
exec(open(os.path.join(B,'breadth_study.py')).read().split("rows=[]; yearly=[]")[0].replace("HERE=os.path.dirname(os.path.abspath(__file__))","HERE=B"),g)
E=g['Era'](g['PACKS']['2015-2025'][0],'2014-12-31',True,())
mask={d:{t for t in MEM[d] if d in E.px.get({'PCLN':'BKNG'}.get(t,t),{})} for d in ME if d in E.mi}
Rm=Run(mask=mask); lgm=[]; e,cm,mm=Rm.path('M6',3,1,i15,120,lgm); chk('我的84%股票池 + Queen价格 Top3',cm,.382); print(f'     (我原来用数据包价格: 38.6%)  覆盖率 {st.mean(len(mask[ME[i]])/len(MEM[ME[i]]) for i in range(i15,i15+120))*100:.1f}%')
att=collections.Counter(); nm=collections.Counter(); nd=0
for (a,b,pf,gf),(a2,b2,pm_,gm) in zip(logs[('2015',3)],lgm):
    new=[t for t in pf if t not in mask[a]]
    if set(pf)!=set(pm_): nd+=1
    for t in new: att[t]+=(gf-gm)/len(new); nm[t]+=1
print(f'     Top3 持仓变化的月份: {nd}/120 (Queen: 64)')
srt=sorted(att.items(),key=lambda x:x[1]); print('     拖累最大:',[(t,'%+.0fpp'%(v*100),nm[t]) for t,v in srt[:9]]); print('     帮忙最大:',[(t,'%+.0fpp'%(v*100),nm[t]) for t,v in srt[-4:]])
w=csv.writer(open(os.path.join(H,'attribution_added_names.csv'),'w',newline='')); w.writerow(['ticker','sum_monthly_return_diff_full_minus_partial','months_held']); [w.writerow([t,'%.4f'%v,nm[t]]) for t,v in srt]
QA={'BIDU':-.22,'PDD':-.14,'LCID':-.12,'MELI':-.10,'PTON':-.09,'ZM':.18}
for t,q in QA.items(): chk(f'归因 {t}',att[t],q,tol=0.03)
yr=lambda lg:[(lambda s:s)(__import__('math').prod(1+x[3] for x in lg[12*k:12*k+12])-1) for k in range(10)]
print('     逐年(毛收益) 84%池:',' '.join('%+.0f%%'%(x*100) for x in yr(lgm))); print('     逐年(毛收益) 完整池:',' '.join('%+.0f%%'%(x*100) for x in yr(logs[('2015',3)])))
# ---------- 2. grid
print('== 2. 网格 6,912 条路径')
def grid(run,i0,n,tag):
    cell={}; rows=[]
    for W in range(1,13):
        for Hh in range(1,13):
            for K in range(1,5):
                cs=[]
                for ph in range(1,13):
                    e,c,m=run.path(W,K,Hh,i0+ph-1,n); cs.append(c); rows.append((W,Hh,K,ph,round(e[-1],4),round(c,6),round(m,4)))
                cell[(W,Hh,K)]=cs
    w=csv.writer(open(os.path.join(H,f'grid_paths_{tag}.csv'),'w',newline='')); w.writerow(['W','H','K','phase','final_multiple','CAGR','monthend_MDD']); w.writerows(rows)
    return cell
def summ(cell,K): 
    med={k:st.median(v) for k,v in cell.items() if k[2]==K}
    return dict(med=st.median(med.values()),hi=st.median([v for k,v in med.items() if k[0]>=4]),lo=st.median([v for k,v in med.items() if k[0]<=3]),n173=sum(1 for v in med.values() if v>.173),n132=sum(1 for v in med.values() if v>.132),lose=sum(1 for k,v in cell.items() if k[2]==K for c in v if c<0),best=max(med.items(),key=lambda x:x[1]),med_cells=med)
QG={'full2015':[(.292,111,.322,.095),(.271,121,.292,.152),(.253,112,.268,.157),(.224,112,.235,.153)],'full2015_noNVDA':[(.122,43,.140,.059),(.161,58,.181,.086),(.153,51,.173,.110),(.146,32,.157,.115)],'oos2008':[(.012,9,-.008,.049),(.097,33,.085,.136),(.117,51,.106,.175),(.123,55,.115,.171)]}
cells={}
for tag,run,i0,n in (('full2015',R,i15,120),('full2015_noNVDA',Rn,i15,120),('oos2008',R,IDX['2007-12-31'],84)):
    cells[tag]=grid(run,i0,n,tag)
    for K in range(1,5):
        s=summ(cells[tag],K); q=QG[tag][K-1]
        chk(f'{tag} Top{K} 144格中位数',s['med'],q[0]); chk(f'{tag} Top{K} W4-12 中位数',s['hi'],q[2],tol=0.008); chk(f'{tag} Top{K} W1-3 中位数',s['lo'],q[3],tol=0.008)
        nn=s['n132'] if tag=='oos2008' else s['n173']; print(f'     高于{"13.2%" if tag=="oos2008" else "17.3%"}的格数: Claude {nn} | Queen {q[1]} | 最好格 W{s["best"][0][0]}/H{s["best"][0][1]} {P(s["best"][1])} | 亏钱路径 {s["lose"]}')
# rank correlation of cell medians between decades
def spearman(a,b):
    ra={k:i for i,k in enumerate(sorted(a,key=a.get))}; rb={k:i for i,k in enumerate(sorted(b,key=b.get))}; n=len(a); return 1-6*sum((ra[k]-rb[k])**2 for k in a)/(n*(n*n-1))
for K in range(1,5):
    a=summ(cells['full2015'],K)['med_cells']; b=summ(cells['oos2008'],K)['med_cells']; print(f'     Top{K}: 两段 144 格中位数的秩相关 = {spearman(a,b):+.2f}')
w=csv.writer(open(os.path.join(H,'grid_cell_medians.csv'),'w',newline='')); w.writerow(['run','W','H','K','median_CAGR','min','max'])
for tag,c in cells.items():
    for (W,Hh,K),v in sorted(c.items()): w.writerow([tag,W,Hh,K,round(st.median(v),6),round(min(v),6),round(max(v),6)])
# ---------- 3. 2026 live
print('== 3. 2026 年帖子数字与实际表现')
post={'SNDK':47.75,'MU':31.37,'INTC':24.55,'STX':24.37,'WDC':23.68,'ARM':20.62,'LITE':20.33,'AMD':19.17,'MRVL':17.26,'LRCX':13.62}
rk=R.rank(IDX['2026-05-29'],'M6')[:10]; print('     2026-05-29 前十:',[(t,round(v*100,2)) for t,v in rk]); print(f'     与帖子截图逐项相同(名次与数值±0.01): {all(t==pt and abs(v*100-pv)<0.011 for (t,v),(pt,pv) in zip(rk,post.items()))}')
print('     2026-04-30 前四:',[(t,round(v*100,2)) for t,v in R.rank(IDX['2026-04-30'],'M6')[:4]])
i26=IDX['2026-04-30']; n26=len(ME)-1-i26
lg0=[]; e0,_,_=R0.path('M6',3,1,i26,n26,lg0); lg1=[]; e1,_,_=R.path('M6',3,1,i26,n26,lg1)
for a,b,pr,g in lg0: print(f'     {a}→{b} {" ".join(f"{t} {v*100:+.1f}%" for t,v in pr.items())} | 组合 {g*100:+.2f}%')
chk('5月组合 (不计成本)',lg0[0][3],.3274,tol=0.0002,fmt=lambda x:'%.2f%%'%(x*100)); chk('2026-04-30→11-13 不计成本',e0[-1]-1,-.089); 
# Queen's 25bp convention: no liquidation on the last (open) period -> report both
e1b=e1[-1]/(1-0.0025) if True else 0
chk('2026-04-30→11-13 单边25bp (期末不平仓)',e1[-2]*(1+lg1[-1][3])-1 if False else e1b-1,-.103); print(f'     (若期末平仓再扣25bp: {P(e1[-1]-1)})')
chk('QQQ 2026-04-30→11-13 (FMP 667.74→679.89)',679.89/667.74-1,.018)
w=csv.DictWriter(open(os.path.join(H,'verdicts.csv'),'w',newline=''),fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(f'合计 {len(out)} 项: 一致 {sum(1 for o in out if o["verdict"]=="一致")}, 不一致 {sum(1 for o in out if o["verdict"]!="一致")}')
for o in out:
    if o['verdict']!='一致': print('  不一致:',o['item'],'%+.2fpp'%(o['diff']*100))
