"""Benchmarks computed this run from FMP daily files pulled 2026-11-13 (momentum-etf-check-20261113/raw): tr = dividend-adjusted adjClose, px = split-adjusted close."""
import pandas as pd
R='/workspace/momentum-etf-check-20261113/raw/';rows=[]
for t in ['SPY','SPMO','MTUM','QQQ','PDP','XMMO']:
    for kind in ('tr','px'):
        try: d=pd.read_csv(R+f'{t}_{kind}_daily.csv')
        except FileNotFoundError: continue
        col=[c for c in d.columns if c!='date'][0] if 'adjClose' not in d.columns and 'close' not in d.columns else ('adjClose' if 'adjClose' in d.columns else 'close')
        s=d.set_index('date')[col].astype(float).sort_index()
        for a,b in (('2015-01-02','2025-01-02'),('2005-01-03','2015-01-02')):
            z=s[(s.index>=a)&(s.index<=b)]
            if not len(z): continue
            yrs=(pd.Timestamp(z.index[-1])-pd.Timestamp(z.index[0])).days/365.25
            rows.append(dict(ticker=t,kind=kind,start=z.index[0],end=z.index[-1],full_window=(z.index[0]==a and z.index[-1]==b),years=round(yrs,3),CAGR=(z.iloc[-1]/z.iloc[0])**(1/yrs)-1,MDD=float((z/z.cummax()-1).min()),col=col))
D=pd.DataFrame(rows);D.to_csv('results/benchmarks.csv',index=False);print(D.to_string())
