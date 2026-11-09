"""Write daily NAV per path (known paths only) as 12 wide gz CSVs, one per W."""
import pickle,os,pandas as pd
B='/workspace/momentum432-answer-key-full-20261109/'
NAV=pickle.load(open('/workspace/m432full-scratch/NAV.pkl','rb'))
S=pd.read_csv('/workspace/momentum432-inputs-pack-full-20261109/trading_sessions.csv').session.tolist()
P=pd.read_csv(B+'paths_summary.csv').set_index('path_id');os.makedirs(B+'nav_daily',exist_ok=True)
for W in range(1,13):
    cols={}
    for pid in P.index:
        if pid.startswith(f'W{W}H') and pid in NAV:
            i=S.index(P.loc[pid,'entry']);v=NAV[pid];cols[pid]=pd.Series(v,index=S[i:i+len(v)])
    df=pd.DataFrame(cols).sort_index();df.index.name='date'
    df.to_csv(B+f'nav_daily/nav_daily_W{W:02d}.csv.gz',float_format='%.12g',compression='gzip')
print('ok')
