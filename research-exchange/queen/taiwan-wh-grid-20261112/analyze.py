import pandas as pd,numpy as np
pd.set_option('display.width',300);pd.set_option('display.max_rows',500);pd.set_option('display.max_columns',50)
S=pd.read_csv('grid_summary_lo.csv');H_=pd.read_csv('grid_summary_hi.csv')
K=['univ','W','H','M','window']
S['excess_log']=np.log(S.tr_mult/S.b0050_tr_mult)
# month sensitivity
rows=[]
for (u,W,H,w),g in S.groupby(['univ','W','H','window']):
    g=g.sort_values('M'); d=dict(univ=u,W=W,H=H,window=w)
    for _,r in g.iterrows(): d[f'M{r.M:02d}_tr']=round(r.tr_mult,3); d[f'M{r.M:02d}_0050']=round(r.b0050_tr_mult,3)
    d.update(min_tr=g.tr_mult.min(),median_tr=g.tr_mult.median(),max_tr=g.tr_mult.max(),
             months_beat_0050=int(g.beats_0050.sum()),best_month=int(g.loc[g.tr_mult.idxmax(),'M']),worst_month=int(g.loc[g.tr_mult.idxmin(),'M']),
             median_excess_log=g.excess_log.median(),min_excess_log=g.excess_log.min(),
             dec_tr=float(g[g.M==12].tr_mult.iloc[0]),dec_0050=float(g[g.M==12].b0050_tr_mult.iloc[0]),dec_beats=bool(g[g.M==12].beats_0050.iloc[0]))
    d['month_fragile']=d['months_beat_0050']<9
    rows.append(d)
MS=pd.DataFrame(rows); MS.to_csv('month_sensitivity_summary_lo.csv',index=False)
# hi flips
m=S.merge(H_[K+['tr_mult','price_mult','beats_0050']],on=K,suffixes=('','_hi'))
flip=m[m.beats_0050!=m.beats_0050_hi]; print('lo/hi beats-0050 flips:',len(flip)); print(flip[K+['tr_mult','tr_mult_hi','b0050_tr_mult']].to_string())
print('max |tr lo-hi| rel',(m.tr_mult_hi/m.tr_mult-1).abs().max())
dec=S[S.M==12]
print('\n=== OFFICIAL Dec headlines ===')
for (W,H) in [(12,12),(6,6)]:
    print(dec[(dec.W==W)&(dec.H==H)][['univ','W','H','window','n','first_signal','last_end','price_mult','tr_mult','tr_cagr','tr_maxdd','b0050_tr_mult','b0050_cagr','pct_beat_0050','worst_tr','n_flagged']].round(3).to_string())
print('\n=== Dec-phase: # of 144 (W,H) beating 0050 TR, per univ/window ===')
print(dec.groupby(['univ','window']).beats_0050.sum().unstack())
allw=dec.pivot_table(index=['univ','W','H'],columns='window',values='beats_0050',aggfunc='first')
allw['all3']=allw.all(axis=1); print(allw.groupby('univ').all3.sum())
print('\n=== MS: (W,H) robust (>=9/12 months beat) per univ/window ===')
print(MS.groupby(['univ','window']).apply(lambda g:pd.Series(dict(robust=(g.months_beat_0050>=9).sum(),all12=(g.months_beat_0050==12).sum(),dec_beats=g.dec_beats.sum(),dec_beats_but_fragile=(g.dec_beats&g.month_fragile).sum()))))
for u in ['TW50','TW150','MC100']:
    for w in ['5y','10y','full']:
        x=dec[(dec.univ==u)&(dec.window==w)].pivot(index='W',columns='H',values='excess_log')
        print(f'\n{u} {w} Dec-phase log(TR mult / 0050 mult):');print(x.round(2).to_string())

# distinct phases: months with identical in-window period sets give identical metrics
def phase_stats(g):
    u=g.drop_duplicates(['n','first_signal','tr_mult'])
    return pd.Series(dict(distinct_phases=len(u),phases_beat=int(u.beats_0050.sum()),phase_median_excess=u.excess_log.median()))
PS=S.groupby(['univ','W','H','window']).apply(phase_stats).reset_index()
MS=MS.merge(PS,on=['univ','W','H','window']); MS['phase_beat_share']=MS.phases_beat/MS.distinct_phases
MS.to_csv('month_sensitivity_summary_lo.csv',index=False)
print('\n=== share of distinct phases beating 0050, averaged over (W,H) ===')
print(MS.groupby(['univ','window']).phase_beat_share.mean().unstack().round(2))
print(MS[MS.H>=2].groupby(['univ','window']).apply(lambda g:pd.Series(dict(cells=len(g),robust_ge75=(g.phase_beat_share>=0.75).sum(),all_phases=(g.phase_beat_share==1).sum(),dec_beats=g.dec_beats.sum(),dec_beats_and_ge75=((g.phase_beat_share>=0.75)&g.dec_beats).sum()))))
# Descriptive: Dec-phase cells, min excess over 3 windows
D=dec.pivot_table(index=['univ','W','H'],columns='window',values='excess_log').reset_index()
D['min_ex']=D[['5y','10y','full']].min(axis=1)
D=D.merge(MS[MS.window=='full'][['univ','W','H','phases_beat','distinct_phases','months_beat_0050']],on=['univ','W','H']).merge(MS[MS.window=='10y'][['univ','W','H','phases_beat','distinct_phases']],on=['univ','W','H'],suffixes=('_full','_10y'))
D.to_csv('dec_phase_cells_ranked_lo.csv',index=False)
for u in ['TW50','TW150','MC100']:
    print(f'\n{u} top Dec cells by min log-excess across 3 windows');print(D[D.univ==u].sort_values('min_ex',ascending=False).head(10).round(2).to_string(index=False))
# neighbourhood: W7-9 x H3-6 block
for u in ['TW150','MC100','TW50']:
    b=MS[(MS.univ==u)&MS.W.between(7,9)&MS.H.between(3,6)]
    print(u,'W7-9xH3-6 block: dec beats',b.groupby('window').dec_beats.sum().to_dict(),'mean phase share',b.groupby('window').phase_beat_share.mean().round(2).to_dict())
