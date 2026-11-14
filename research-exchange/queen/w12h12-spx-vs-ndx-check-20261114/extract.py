# Queen check of Claude bb2c29e (W12/H12/Top3 x 12 phases, S&P 500 vs Nasdaq-100). Run from /workspace.
import csv,statistics as st
def get(f,**k): return [r for r in csv.DictReader(open(f)) if all(r[a]==str(b) for a,b in k.items())]
src={'SPX_v18_answer_key':'momentum432-answer-key-full-20261109/cells_12phase_medians.csv',
     'SPX_v18_noNVDA':'no-nvda-grid-20261114/cells_12phase_medians_no_nvda.csv',
     'NDX_v1.1_full_2015':'ndx100-fill-20261114-v1.1/out/grid/cells_full_2015.csv',
     'NDX_v1.1_full_2015_noNVDA':'ndx100-fill-20261114-v1.1/out/grid/cells_full_2015_noNVDA.csv'}
out=[]
for n,f in src.items():
    r=get(f,W=12,H=12,K=3)[0]; out.append(dict(source=n,file=f,median_CAGR=r['median_CAGR'],min_CAGR=r['min_CAGR'],max_CAGR=r['max_CAGR'],worst_MDD=r['worst_MDD'],n_known=r.get('n_known','12')))
w=csv.DictWriter(open('gh-sync-20261114/w12h12-spx-vs-ndx-check-20261114/w12h12k3_queen.csv','w',newline=''),fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
for n,f in [('SPX',src['SPX_v18_answer_key']),('NDX',src['NDX_v1.1_full_2015'])]:
    v=[float(r['median_CAGR']) for r in csv.DictReader(open(f)) if r['K']=='3' and r['median_CAGR'] not in ('','nan','None')]
    print(n,'Top3 cells',len(v),'median of cell medians %.4f'%st.median(v))
for o in out: print(o)
