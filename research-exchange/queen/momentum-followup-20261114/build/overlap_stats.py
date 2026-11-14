import pickle,collections,json,pandas as pd
R=pickle.load(open('scratch/ranks_1525.pkl','rb'))
MS=sorted({s for s,_ in R})
def top(S,W,ex):
    return [t for t in R[(S,W)]['top'] if t not in ex]
rows=[]
for label,win in (('A: signals 2014-12-31..2024-11-29 (trade months 2015-01..2024-12)',[m for m in MS if '2014-12-01'<=m<='2024-11-30']),
                  ('B: signals 2015-01-30..2024-12-31',[m for m in MS if '2015-01-01'<=m<='2024-12-31'])):
  for mode,ex in (('with NVDA',set()),('no NVDA',{'NVDA'})):
    n=len(win);miss=sum(1 for S in win for W in range(1,13) if R[(S,W)]['miss'])
    def agree(a,b): return sum(top(S,a,ex)[0]==top(S,b,ex)[0] for S in win)/n
    def ov3(a,b): return sum(len(set(top(S,a,ex)[:3])&set(top(S,b,ex)[:3]))/3 for S in win)/n
    d=dict(window=label,mode=mode,n_months=n,signals_with_unrankable=miss)
    for a,b in ((10,11),(11,12),(10,12),(6,12),(1,12)): d[f'W{a}vW{b}_top1']=agree(a,b);d[f'W{a}vW{b}_top3ov']=ov3(a,b)
    d['W10_11_12_same_top1']=sum(len({top(S,w,ex)[0] for w in (10,11,12)})==1 for S in win)/n
    i0=MS.index(win[0]);d['W12_top1_same_as_prev_month']=sum(top(S,12,ex)[0]==top(MS[MS.index(S)-1],12,ex)[0] for S in win)/n
    c=collections.Counter(top(S,w,ex)[0] for S in win for w in (10,11,12));tot=sum(c.values())
    d['top1_share_W10_12']=json.dumps({k:round(v/tot,4) for k,v in c.most_common(8)});d['top5_sum']=sum(v for _,v in c.most_common(5))/tot
    d['adj_top1_W10_11_12_mean']=(agree(10,11)+agree(11,12))/2;d['adj_top3ov_W10_11_12_mean']=(ov3(10,11)+ov3(11,12))/2
    d['adj_top1_W1_12_mean']=sum(agree(w,w+1) for w in range(1,12))/11;d['adj_top3ov_W1_12_mean']=sum(ov3(w,w+1) for w in range(1,12))/11
    rows.append(d)
D=pd.DataFrame(rows);D.to_csv('results/overlap_stats.csv',index=False)
for r in rows:
    print('\n',r['window'],r['mode'])
    for k,v in r.items():
        if k not in('window','mode'): print(' ',k,round(v,4) if isinstance(v,float) else v)
