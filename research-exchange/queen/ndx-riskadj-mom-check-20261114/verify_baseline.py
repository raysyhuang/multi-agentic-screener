# sanity: this runner reproduces Queen's verified post-rule baseline (month-end close, 25 bps/side, 2015-01..2026-11-13)
from riskadj import *
s=stats(run(('mean6',),3,IDX['2014-12-31'],IDX['2026-11-13'],0.0025,lag=0))
print('mean6 Top3 25bp close-exec: CAGR(months/12) %.4f%%  CAGR(cal days) %.4f%%  MDD_daily %.4f%%  multiple x%.3f'%(s['CAGR_months12']*100,s['CAGR']*100,s['MDD_daily']*100,s['total']))
s0=E.stats(E.run(3,IDX['2014-12-31'],IDX['2026-11-13'],(),0.0025))
print('engine.run reference:            CAGR(months/12) %.4f%%  MDD_daily %.4f%%  multiple x%.3f'%(s0['CAGR']*100,s0['MDD_daily']*100,s0['total']))
