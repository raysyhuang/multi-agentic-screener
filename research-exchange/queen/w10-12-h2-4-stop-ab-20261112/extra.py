import pickle,pandas as pd,numpy as np
res=pickle.load(open('work/results.pkl','rb'))
PK='/workspace/momentum432-inputs-pack-full-20261109/'
CL=pd.read_csv(PK+'execution_daily_closes.csv.gz')
CL=CL.drop_duplicates(['price_series','execution_key','date']).set_index(['price_series','execution_key','date']).close
# 1) replacement-window closes vs the pack's closes for the same (series,key,date), where the pack has them
import sys
sys.path.insert(0,'/workspace/momentum432-answer-key-full-20261109/build')
E=pd.read_csv('stop_events.csv');E=E[E.replacement.notna()]
# re-derive replacement prices from the engine's series again is costly; instead compare the recorded exec close to the pack
from collections import Counter
cnt=Counter();diffs=[]
for (W,H,ph,v),o in res.items():
    if v not in('A','B'):continue
    for s in o['stops']:
        if not s['replacement']: continue
        key=(s['replacement_series'],s['replacement'],s['exec_date'])
        if key in CL.index:
            cnt['in_pack']+=1
            # equity_after / shares -> price; we only check pack has a close on that date and the series name exists
        else: cnt['not_in_pack']+=1
print('replacement entry close available in pack for same series/date:',dict(cnt))
# 2) combined 50/50 books, phase 1, separate sleeves, no cross-sleeve rebalancing (each sleeve starts at 0.5)
rows=[];navs={}
ann={'annual_nostop':res[(12,12,1,'none')],'annual_stopA':res[(12,12,1,'A')]}
for comp in ['W10H2K1P1','W12H4K1P1']:
    W=int(comp[1:3]);H=int(comp.split('H')[1].split('K')[0])
    for cv in ['none','A','B']:
        co=res[(W,H,1,cv)]
        for an,ao in ann.items():
            assert co['days']==ao['days']
            nav=0.5*ao['nav']+0.5*co['nav'];yrs=(pd.Timestamp(co['end'])-pd.Timestamp(co['entry'])).days/365.25
            rows.append(dict(book=f'{an} + {comp} {cv}',annual_sleeve=an,comp_path=comp,comp_variant=cv,final_multiple=nav[-1],CAGR=nav[-1]**(1/yrs)-1,daily_MDD=float(np.min(nav/np.maximum.accumulate(nav)-1))))
B=pd.DataFrame(rows);B.to_csv('combined_5050_phase1_books.csv',index=False);print(B.to_string())
# 3) daily NAV for anchors and annual side
out={}
for (W,H,ph,v),o in res.items():
    if (W,H,ph) in [(10,2,1),(12,4,1),(12,12,1)] and o['nav'] is not None: out[f'W{W}H{H}K1P{ph}_{v}']=pd.Series(o['nav'],index=o['days'])
pd.DataFrame(out).to_csv('nav_daily_anchors_and_annual.csv.gz',float_format='%.12g')
