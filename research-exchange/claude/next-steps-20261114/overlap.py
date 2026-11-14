#!/usr/bin/env python3
"""Claude: how much do the top-ranked stocks overlap across lookback lengths?  2015-2025 frozen pack, stdlib only.
Usage: python3 overlap.py   (runs Claude's ranker twice: all names, and NVDA excluded)"""
import os,sys,subprocess,pickle,collections,statistics as st,csv
HERE=os.path.dirname(os.path.abspath(__file__)); A=os.path.join(HERE,'..','input-audit-20261112'); PACK=os.path.join(HERE,'..','..','shared','inputs-pack-full-20261109')
rows=[]
for label,excl in (('with NVDA',''),('without NVDA','NVDA')):
    pk=os.path.join(HERE,'_rank.pkl'); subprocess.check_call([sys.executable,os.path.join(A,'independent_ranker_claude.py'),PACK],env=dict(os.environ,EXCLUDE_TICKERS=excl,RANK_OUT=pk),stdout=subprocess.DEVNULL)
    m=pickle.load(open(pk,'rb')); os.remove(pk)
    S_all=sorted({k[0] for k in m if '2014-12-31'<=k[0]<='2024-11-29'}); ok=lambda S,W:(S,W) in m and not m[(S,W)]['bad']
    print('==',label,'| month-end signals',len(S_all))
    for a,b in ((10,11),(11,12),(10,12),(6,12),(3,12),(1,12)):
        z=[S for S in S_all if ok(S,a) and ok(S,b)]
        t1=sum(m[(S,a)]['top'][0][0]==m[(S,b)]['top'][0][0] for S in z)/len(z)
        j3=st.mean(len({t for t,_ in m[(S,a)]['top'][:3]}&{t for t,_ in m[(S,b)]['top'][:3]})/3 for S in z)
        rows.append(dict(universe=label,lookback_a=a,lookback_b=b,same_top1_share=round(t1,4),top3_shared_share=round(j3,4)))
        print('  W%d vs W%d: same Top1 on %.0f%% of month-ends; Top3 names shared %.0f%%'%(a,b,t1*100,j3*100))
    z=[S for S in S_all if all(ok(S,W) for W in (10,11,12))]
    c=collections.Counter(m[(S,W)]['top'][0][0] for S in z for W in (10,11,12)); tot=sum(c.values())
    print('  W10/W11/W12 all pick the same Top1 on %.0f%% of month-ends'%(100*sum(len({m[(S,W)]['top'][0][0] for W in (10,11,12)})==1 for S in z)/len(z)))
    print('  distinct Top1 names (W10-12): %d; top 5 names take %.0f%% of picks: %s'%(len(c),100*sum(v for _,v in c.most_common(5))/tot,[(t,'%.0f%%'%(100*v/tot)) for t,v in c.most_common(6)]))
    z=[S for S in S_all if ok(S,12)]
    print('  W12 Top1 unchanged from the previous month: %.0f%% of months'%(100*sum(m[(a,12)]['top'][0][0]==m[(b,12)]['top'][0][0] for a,b in zip(z,z[1:]))/(len(z)-1)))
w=csv.DictWriter(open(os.path.join(HERE,'overlap_by_lookback.csv'),'w',newline=''),fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
