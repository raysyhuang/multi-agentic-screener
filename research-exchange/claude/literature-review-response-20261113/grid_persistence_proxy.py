#!/usr/bin/env python3
"""Claude: does the W x H surface persist across decades?  PROXY test, stdlib only.
Top1 for every W1-12 x H1-12 x 12 phases in both decades, using ONLY month-end rank prices from each pack:
buy at the signal month-end close, hold H months, 0.25% one-way fee, spin/special-dividend F applied, a name whose
price series ends (deal) is carried at its last price. This is NOT the exact engine (which trades next-day closes with
full corporate-action handling); it is validated below against Queen's exact 2015-2025 answers.
Usage: python3 grid_persistence_proxy.py"""
import csv,gzip,os,collections,statistics as st
HERE=os.path.dirname(os.path.abspath(__file__)); R=os.path.join(HERE,'..','..'); C=0.0025
def load(P):
    rd=lambda f:list(csv.DictReader(gzip.open(os.path.join(P,f),'rt') if f.endswith('.gz') else open(os.path.join(P,f))))
    px=collections.defaultdict(dict)
    for r in rd('rank_prices_month_end.csv.gz'):
        if r['rank_price']: px[r['engine_ticker']][r['month_end']]=float(r['rank_price'])
    el=collections.defaultdict(list)
    for r in rd('eligibility_by_signal.csv.gz'):
        if r['eligible_final']=='True': el[r['signal_date']].append(r['engine_ticker'])
    ca=collections.defaultdict(list)
    for r in rd('corporate_actions_applied.csv'):
        if r['kind'] in('spin','special_div') and r['factor_F']: ca[r['engine_ticker']].append((r['ex_date'],float(r['factor_F'])))
    ME=sorted({d for v in px.values() for d in v}); return px,el,ca,ME
def F(ca,t,a,b):
    f=1.0
    for e,x in ca.get(t,()):
        if a<e<=b: f*=x
    return f
def grid(P,first_signal):
    px,el,ca,ME=load(P); mi={d:i for i,d in enumerate(ME)}; sig=sorted(el); out={}
    top={}
    def top1(S,W):
        if (S,W) in top: return top[(S,W)]
        B=ME[mi[S]-W]; best=None
        for t in el[S]:
            if S in px[t] and B in px[t]:
                r=px[t][S]/(px[t][B]*F(ca,t,B,S))-1
                if best is None or r>best[0] or (r==best[0] and t<best[1]): best=(r,t)
        top[(S,W)]=best[1]; return best[1]
    i0=mi[first_signal]
    for W in range(1,13):
        for H in range(1,13):
            res=[]
            for ph in range(12):
                s=i0+ph; e=s+120
                if e>=len(ME) or ME[s+((119)//H)*H] not in el: continue
                eq=1.0; held=None; k=s
                while k<e:
                    t=top1(ME[k],W); nk=min(k+H,e)
                    if t!=held: eq*= (1-C)/(1+C) if held else 1/(1+C)
                    a=ME[k]; ds=[d for d in ME[k+1:nk+1] if d in px[t]]; b=ds[-1] if ds else a
                    eq*=px[t][b]/(px[t][a]*F(ca,t,a,b)) if b!=a else 1.0
                    held=t if (ds and ds[-1]==ME[nk]) else None      # series ended -> treated as cash
                    k=nk
                eq*=(1-C) if held else 1.0
                res.append(eq**0.1-1)
            out[(W,H)]=res
    return out
def spearman(x,y):
    def rk(v):
        o=sorted(range(len(v)),key=lambda i:v[i]); r=[0]*len(v)
        for j,i in enumerate(o): r[i]=j
        return r
    a,b=rk(x),rk(y); n=len(x); return 1-6*sum((p-q)**2 for p,q in zip(a,b))/(n*(n*n-1))
g15=grid(os.path.join(R,'shared','inputs-pack-full-20261109'),'2014-12-31')
g05=grid(os.path.join(R,'queen','w10h2-w12h4-oos-2005-2015-20261112','pack'),'2004-12-31')
exact=collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(R,'queen','answer-key-full-20261109','paths_summary.csv'))):
    if r['K']=='1' and r['status']!='UNKNOWN': exact[(int(r['W']),int(r['H']))].append(float(r['CAGR']))
cells=[(W,H) for W in range(1,13) for H in range(1,13) if g15[(W,H)] and g05[(W,H)]]
m15=[st.median(g15[c]) for c in cells]; m05=[st.median(g05[c]) for c in cells]
vc=[c for c in cells if len(exact[c])>=6]; ex=[st.median(exact[c]) if exact[c] else float('nan') for c in cells]
print('cells compared',len(cells),'; cells with >=6 exact K1 paths for validation',len(vc))
print('phases per cell 2015-25:',{len(v) for v in g15.values()},' 2005-15:',{len(v) for v in g05.values()})
print('VALIDATION proxy vs Queen exact, 2015-2025 K1 cell medians: Spearman %.3f; mean abs diff %.2f pp; W10H2 proxy %.1f%% exact %.1f%%; W12H4 proxy %.1f%% exact %.1f%%'%(spearman([m15[cells.index(c)] for c in vc],[ex[cells.index(c)] for c in vc]),st.mean(abs(m15[cells.index(c)]-ex[cells.index(c)]) for c in vc)*100,st.median(g15[(10,2)])*100,st.median(exact[(10,2)])*100,st.median(g15[(12,4)])*100,st.median(exact[(12,4)])*100))
print('2005-2015 proxy: W10H2 %.1f%% (Queen exact -4.8%%), W12H4 %.1f%% (Queen exact -10.4%%)'%(st.median(g05[(10,2)])*100,st.median(g05[(12,4)])*100))
print('CROSS-DECADE Spearman of cell medians (2005-15 vs 2015-25): %.3f'%spearman(m05,m15))
top15=sorted(cells,key=lambda c:-m15[cells.index(c)])[:14]; r05=sorted(cells,key=lambda c:-m05[cells.index(c)])
print('top-decile cells of 2015-25 and their rank (of 144) in 2005-15:',[(c,r05.index(c)+1) for c in top15])
print('2005-15: cells with positive median CAGR %d of 144; best %s %.1f%%; median cell %.1f%%; W>=10 cells median %.1f%%, W<=3 cells median %.1f%%'%(sum(1 for v in m05 if v>0),r05[0],m05[cells.index(r05[0])]*100,st.median(m05)*100,st.median(m05[cells.index(c)] for c in cells if c[0]>=10)*100,st.median(m05[cells.index(c)] for c in cells if c[0]<=3)*100))
print('2015-25: W>=10 cells median %.1f%%, W<=3 cells median %.1f%%'%(st.median(m15[cells.index(c)] for c in cells if c[0]>=10)*100,st.median(m15[cells.index(c)] for c in cells if c[0]<=3)*100))
w=csv.writer(open(os.path.join(HERE,'grid_persistence_proxy.csv'),'w',newline='')); w.writerow(['W','H','proxy_median_CAGR_2005_2015','proxy_median_CAGR_2015_2025','queen_exact_median_CAGR_2015_2025'])
for c,a,b,e in zip(cells,m05,m15,ex): w.writerow([c[0],c[1],a,b,e])
