#!/usr/bin/env python3
"""Claude: (a) cell-by-cell comparison of my grid with Queen's grid files, (b) the known data disagreements.  Usage: python3 check_disagreements.py (after verify_queen_ndx.py)"""
import os,csv,gzip,json,collections,statistics as st,datetime as dt
H=os.path.dirname(os.path.abspath(__file__)); Q=os.path.join(H,'..','..','queen','ndx100-fill-20261114'); PK=os.path.join(H,'..','..','shared','inputs-pack-full-20261109')
print('== (a) 网格逐格比较 (每格 12 个起始月的年化中位数)')
mine=collections.defaultdict(dict)
for r in csv.DictReader(open(os.path.join(H,'grid_cell_medians.csv'))): mine[r['run']][(int(r['W']),int(r['H']),int(r['K']))]=float(r['median_CAGR'])
for tag,f in (('full2015','cells_full_2015.csv'),('full2015_noNVDA','cells_full_2015_noNVDA.csv'),('oos2008','cells_full_oos_2008.csv')):
    rows=list(csv.DictReader(open(os.path.join(Q,'out','grid',f)))); col=[c for c in rows[0] if 'median' in c.lower()][0]
    d={(int(r['W']),int(r['H']),int(r['K'])):mine[tag][(int(r['W']),int(r['H']),int(r['K']))]-float(r[col]) for r in rows}
    a=[abs(x) for x in d.values()]; big=sorted(d.items(),key=lambda x:-abs(x[1]))[:4]
    print(f'  {tag}: 576 格, |差| 中位数 {st.median(a)*100:.2f}pp, 平均 {st.mean(a)*100:.2f}pp, >1pp 的格 {sum(1 for x in a if x>0.01)}, >3pp 的格 {sum(1 for x in a if x>0.03)}, 最大 {[(k,"%+.1fpp"%(v*100)) for k,v in big]}')
    for K in range(1,5):
        lo=[mine[tag][k] for k in mine[tag] if k[2]==K and k[0]<=3]; ql=[float(r[col]) for r in rows if int(r['K'])==K and int(r['W'])<=3]
        if K==1: print(f'     Top1 W1-3 36格中位数: 我 {st.median(lo)*100:.1f}% | 用 Queen 的格文件重算 {st.median(ql)*100:.1f}%')
print('== (b) 已知分歧')
qp=collections.defaultdict(dict)
for r in csv.DictReader(open(os.path.join(Q,'out','prices','queen_rank_prices_month_end_all_members.csv'))): qp[r['engine_ticker']][r['month_end']]=float(r['rank_price'])
pp=collections.defaultdict(dict)
for r in csv.DictReader(gzip.open(os.path.join(PK,'rank_prices_month_end.csv.gz'),'rt')):
    if r['rank_price']: pp[r['engine_ticker']][r['month_end']]=float(r['rank_price'])
F={}
for r in csv.DictReader(open(os.path.join(PK,'corporate_actions_applied.csv'))):
    if r['kind'] in ('spin','special_div') and r['factor_F']: F[(r['engine_ticker'],r['ex_date'])]=float(r['factor_F'])
def pret(t,a,b):
    f=1.0
    for (tt,e),x in F.items():
        if tt==t and a<e<=b: f*=x
    return pp[t][b]/(pp[t][a]*f)-1
def daily(t):
    out={}
    for r in csv.DictReader(open(os.path.join(Q,'raw','adj',t+'.csv'))): out[(dt.datetime.fromtimestamp(int(r['t'])/1000,dt.timezone.utc)-dt.timedelta(hours=4)).date().isoformat()]=float(r['c'])
    return out
d=daily('ESRX'); last=max(d); print(f'  ESRX: 2018-11-30 收盘 {d["2018-11-30"]}, 最后交易日 {last} 收盘 {d[last]} → {d[last]/d["2018-11-30"]-1:+.2%}; 交易条款 48.75 现金 + 0.2434 股 CI')
for t,a,b in (('WDC','2025-01-31','2025-02-28'),('EXC','2022-01-31','2022-02-28'),('BIIB','2017-01-31','2017-02-28'),('HON','2018-09-28','2018-10-31'),('HON','2025-09-30','2025-10-31'),('GOOG','2014-03-31','2014-04-30'),('GOOGL','2014-03-31','2014-04-30'),('DISCA','2014-07-31','2014-08-29'),('FOX','2019-02-28','2019-03-29'),('FOXA','2019-02-28','2019-03-29')):
    q=qp[t][b]/qp[t][a]-1 if a in qp[t] and b in qp[t] else None; p=pret(t,a,b) if a in pp[t] and b in pp[t] else None
    print(f'  {t} {a}→{b}: Queen {"NA" if q is None else "%+.2f%%"%(q*100)} | 数据包(含F) {"NA" if p is None else "%+.2f%%"%(p*100)} | 数据包原始价 {pp[t].get(a)}→{pp[t].get(b)} | Queen价 {qp[t].get(a)}→{qp[t].get(b)}')
fm={r['date']:r['close'] for r in json.load(open(os.path.join(Q,'raw','fmp','HON.json')))}; hd=daily('HON')
for x in ('2024-12-31','2025-01-31','2025-10-29','2025-10-30','2026-05-29','2026-06-26','2026-06-29','2026-06-30','2026-07-31'):
    if x in fm and x in hd: print(f'  HON {x}: Massive {hd[x]:.2f} | FMP {fm[x]:.2f} | 比 {hd[x]/fm[x]:.4f}')
