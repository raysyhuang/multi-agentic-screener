"""Cross-check Queen's Massive month-end closes (split-adjusted) against FMP historical-price-eod (sample of 15 symbols)."""
import json,glob,os,csv
from engine import *
rows=[]
for f in sorted(glob.glob('raw/fmp/*.json')):
    t=os.path.basename(f)[:-5]; js=json.load(open(f))
    fm={r['date']:(r.get('price') or r.get('close')) for r in js}
    m=load_daily('raw/adj/%s.csv'%t); fx=PX.D.get(t,{})
    for d in ME:
        if d in fm and d in m and d>=min(fm):
            diff=m[d]/fm[d]-1
            dfx=fx[d]/fm[d]-1 if d in fx else None
            rows.append(dict(ticker=t,month_end=d,massive_raw_file=m[d],massive_after_queen_fixes=fx.get(d),fmp=fm[d],pct_diff_raw=round(diff*100,3),pct_diff_fixed=round(dfx*100,3) if dfx is not None else None,flag_gt_1pct_raw=abs(diff)>0.01,flag_gt_1pct_fixed=dfx is not None and abs(dfx)>0.01))
with open('out/fmp_crosscheck.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
from collections import Counter
n=Counter(r['ticker'] for r in rows); bad=Counter(r['ticker'] for r in rows if r['flag_gt_1pct_raw']); badf=Counter(r['ticker'] for r in rows if r['flag_gt_1pct_fixed'])
print('compared',len(rows),'month-ends; flagged >1% raw:',sum(bad.values()),'after fixes:',sum(badf.values()))
for t in n: print(t,'n=%d raw_flags=%d fixed_flags=%d'%(n[t],bad[t],badf[t]), [ (r['month_end'],r['pct_diff_fixed']) for r in rows if r['ticker']==t and r['flag_gt_1pct_fixed']][:3])
