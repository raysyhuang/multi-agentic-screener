"""Queen re-run of Claude's NDX grid (W1-12 x H1-12 x K1-4 x 12 phases), same path logic as Claude's ndx_grid.py
(W-month total return signal, rebalance every H months, 25bp one-way with fixed-point cost, final liquidation cost, month-end MDD),
but on Queen's Massive split-adjusted prices.  Universes: claude_partial (Claude-pack availability mask), full.
Usage: python3 grid.py"""
import engine, sys, statistics as st, collections
from engine import *
from claude_mask import build_mask
C=0.0025
def prep(mask,exclude):
    engine.MASK=mask
    el={}; at={}
    for i,d in enumerate(ME):
        if d not in MEM: continue
        el[i]=[]
        for t,k in members(d,exclude):
            if PX.at(k,d): el[i].append((t,k))
    engine.MASK=None
    return el
_at={}
def A(k,i):
    c=(k,i)
    if c not in _at: _at[c]=PX.at(k,ME[i]) if i>=0 else None
    return _at[c]
_mr={}
def mret(k,i):
    """return over month i->i+1 and alive flag"""
    c=(k,i)
    if c not in _mr:
        r=PX.hold_ret(k,ME[i],ME[i+1]); a0=A(k,i); a1=A(k,i+1)
        alive = a1 is not None and (k not in PX.k or a1[2]==a0[2])
        _mr[c]=(r[0] if r else 0.0, alive)
    return _mr[c]
def rank(el,i,W,cache):
    if (i,W) not in cache:
        v=[]
        for t,k in el[i]:
            a,b=A(k,i-W),A(k,i)
            if a and b and (k not in PX.k or a[2]==b[2]): v.append((-(b[0]/a[0]-1),t,k))
        cache[(i,W)]=[(t,k) for _,t,k in sorted(v)]
    return cache[(i,W)]
def path(el,cache,s,n,W,Hh,K):
    hold={}; cash=1.0; pk=0.0; mdd=0.0; picks=collections.Counter()
    for m in range(s,s+n):
        if (m-s)%Hh==0:
            names=rank(el,m,W,cache)[:K] if K else el[m]
            E0=cash+sum(hold.values()); Ep=E0; tn={t for t,k in names}
            for _ in range(100):
                T=Ep/len(names); new=E0-C*(sum(abs(T-hold.get(k,0.0)) for t,k in names)+sum(v for k,v in hold.items() if k not in {kk for _,kk in names}))
                if abs(new-Ep)<1e-13: break
                Ep=new
            hold={k:Ep/len(names) for t,k in names}; cash=0.0
            for t,k in names: picks[t]+=1
        eq=cash+sum(hold.values()); pk=max(pk,eq); mdd=min(mdd,eq/pk-1)
        for k in list(hold):
            r,alive=mret(k,m); hold[k]*=1+r
            if not alive: cash+=hold.pop(k)
    eq=cash+sum(hold.values())*(1-C); mdd=min(mdd,eq/max(pk,eq)-1)
    return eq,eq**(12/n)-1,mdd,picks
def qqq_cagr(s,n,ser):
    return (ser[ME[s+n]]/ser[ME[s]])**(12/n)-1
def run_grid(tag,mask,exclude,start,n):
    el=prep(mask,exclude); cache={}; s0=IDX[start]
    rows=[]; cell=collections.defaultdict(list); pk=collections.defaultdict(collections.Counter)
    for W in range(1,13):
        for Hh in range(1,13):
            for K in range(1,5):
                for ph in range(1,13):
                    eq,c,m,p=path(el,cache,s0+ph-1,n,W,Hh,K)
                    rows.append(dict(W=W,H=Hh,K=K,phase=ph,start=ME[s0+ph-1],end=ME[s0+ph-1+n],final_multiple=round(eq,4),CAGR=round(c,6),monthend_MDD=round(m,4)))
                    cell[(W,Hh,K)].append((c,m)); pk[(W,Hh,K)].update(p)
    ewc=[path(el,cache,s0+ph-1,n,1,1,None)[1] for ph in range(1,13)]
    P,TR=qqq(); qp=[qqq_cagr(s0+ph-1,n,P) for ph in range(1,13)]; qt=[qqq_cagr(s0+ph-1,n,TR) for ph in range(1,13)]
    cr=[]
    for (W,Hh,K),v in cell.items():
        cs=[c for c,_ in v]
        cr.append(dict(W=W,H=Hh,K=K,median_CAGR=round(st.median(cs),6),min_CAGR=round(min(cs),6),max_CAGR=round(max(cs),6),worst_MDD=round(min(m for _,m in v),4),
                       phases_beating_QQQ_price=sum(1 for c,q in zip(cs,qp) if c>q),top_picks=' '.join(f'{t}:{n}' for t,n in pk[(W,Hh,K)].most_common(5))))
    with open(f'out/grid/paths_{tag}.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(f'out/grid/cells_{tag}.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(cr[0])); w.writeheader(); w.writerows(cr)
    summ=[]; thr=st.median(qp)
    for K in range(1,5):
        ck=sorted([c for c in cr if c['K']==K],key=lambda c:-c['median_CAGR']); allm=[c['median_CAGR'] for c in ck]
        summ.append(dict(run=tag,K=K,median_of_144_cells=st.median(allm),cells_beating_QQQ_price_median=sum(1 for x in allm if x>thr),
                         cells_beating_17_3=sum(1 for x in allm if x>0.173),best_cell='W%d/H%d'%(ck[0]['W'],ck[0]['H']),best_cell_median=ck[0]['median_CAGR'],
                         best_cell_range='%.1f..%.1f%%'%(ck[0]['min_CAGR']*100,ck[0]['max_CAGR']*100),best_cell_worstMDD=ck[0]['worst_MDD'],
                         worst_cell='W%d/H%d'%(ck[-1]['W'],ck[-1]['H']),worst_cell_median=ck[-1]['median_CAGR'],
                         paths_losing=sum(1 for r in rows if r['K']==K and r['CAGR']<0),
                         QQQ_price_median=thr,QQQ_TR_median=st.median(qt),EW_median=st.median(ewc),
                         band_W4_12_median=st.median([c for W in range(4,13) for Hh in range(1,13) for c,_ in cell[(W,Hh,K)]]),
                         band_W1_3_median=st.median([c for W in range(1,4) for Hh in range(1,13) for c,_ in cell[(W,Hh,K)]])))
        print(tag,'K%d median %.1f%% | beat QQQ(%.1f%%) %d/144 | best %s %.1f%% (%s, MDD %.0f%%) | worst %s %.1f%% | EW %.1f%% QQQ_TR %.1f%%'%(K,summ[-1]['median_of_144_cells']*100,thr*100,summ[-1]['cells_beating_QQQ_price_median'],summ[-1]['best_cell'],ck[0]['median_CAGR']*100,summ[-1]['best_cell_range'],ck[0]['worst_MDD']*100,summ[-1]['worst_cell'],ck[-1]['median_CAGR']*100,st.median(ewc)*100,st.median(qt)*100),flush=True)
    for K in (1,3):
        print('  K=%d median CAGR %%, rows W1..W12, cols H1..H12'%K)
        for W in range(1,13): print('    W%-2d '%W+' '.join('%4.0f'%(st.median([c for c,_ in cell[(W,Hh,K)]])*100) for Hh in range(1,13)))
    return summ
if __name__=='__main__':
    os.makedirs('out/grid',exist_ok=True)
    mask,_,_=build_mask(); allsum=[]
    jobs=[('claude_partial_2015',mask,(),'2014-12-31',120),('claude_partial_2015_noNVDA',mask,('NVDA',),'2014-12-31',120),
          ('full_2015',None,(),'2014-12-31',120),('full_2015_noNVDA',None,('NVDA',),'2014-12-31',120),
          ('full_oos_2008',None,(),'2007-12-31',84),('full_oos_2008_noNVDA',None,('NVDA',),'2007-12-31',84)]
    for j in jobs: allsum+=run_grid(*j)
    with open('out/grid/grid_summary.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allsum[0])); w.writeheader(); w.writerows(allsum)
