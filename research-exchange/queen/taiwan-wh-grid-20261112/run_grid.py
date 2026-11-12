import os,sys,time; exec(open('/workspace/taiwan-wh-grid-20261112/engine.py').read())
TAG=FPMODE; t0=time.time()
def period_nc(picks,s,e,col):  # no cache (memory)
    r=PC.pop((picks,s,e,col),None)
    return r if r else period(picks,s,e,col)
SUM=[];LED=[];CUR=[]
for univ in ('TW50','TW150','MC100'):
    for W in range(1,13):
        for H in range(1,13):
            for M in range(1,13):
                rows=run_cell(univ,W,H,M); PC.clear()
                SUM+=summarize(rows)
                for r in rows:
                    if r['censored']: CUR.append({k:v for k,v in r.items() if not k.startswith('_')})
                    if TAG=='lo': LED.append({k:v for k,v in r.items() if not k.startswith('_')})
        print(univ,W,round(time.time()-t0),flush=True)
pd.DataFrame(SUM).to_csv(f'{OUT}/grid_summary_{TAG}.csv',index=False)
pd.DataFrame(CUR).to_csv(f'{OUT}/current_censored_picks_{TAG}.csv',index=False)
if TAG=='lo':
    pd.DataFrame(LED).to_csv(f'{OUT}/period_ledger_lo.csv.gz',index=False)
    D=pd.DataFrame(DROPS,columns=['univ','W','signal','ticker','reason']).drop_duplicates()
    D.to_csv(f'{OUT}/dropped_names_log.csv',index=False)
print('done',time.time()-t0)
