#!/usr/bin/env python3
"""Claude check of Queen's 2005-2015 cross-period packet. Runs Claude's own ranker and executor (written from SPEC.md,
in ../input-audit-20261112/) on Queen's PROVISIONAL 2005-2015 pack and compares the 25 paths with her results_paths.csv.
Usage (stdlib only):  python3 check_queen_2005_2015.py      -> writes claude_vs_queen_25_paths.csv and stdout"""
import os,sys,csv,subprocess,statistics as st
HERE=os.path.dirname(os.path.abspath(__file__)); A=os.path.join(HERE,'..','input-audit-20261112')
Q=os.path.join(HERE,'..','..','queen','w10h2-w12h4-oos-2005-2015-20261112'); PACK=os.path.join(Q,'pack')
subprocess.check_call([sys.executable,os.path.join(A,'independent_ranker_claude.py'),PACK],stdout=subprocess.DEVNULL)
os.environ['BASE_YEAR']='2005'
src=open(os.path.join(A,'independent_execution_claude.py')).read().split("ref={r['path_id']")[0]
sys.argv=['x',PACK]; g={'__file__':os.path.join(A,'independent_execution_claude.py')}; exec(compile(src,'exec','exec'),g)
run,Fail=g['run'],g['Fail']
ref={r['path']:r for r in csv.DictReader(open(os.path.join(Q,'results','results_paths.csv')))}
rows=[]; mx=0
for pid,q in ref.items():
    W,H,K,ph=int(q['W']),int(q['H']),int(q['K']),int(q['phase'])
    try:
        f,c,m=run(W,H,K,ph); mine=('KNOWN','',f,c,m)
    except Fail as e: mine=('UNKNOWN',str(e),'','','')
    qk='UNKNOWN' if q['status']=='UNKNOWN' else 'KNOWN'
    ok=mine[0]==qk and (qk=='UNKNOWN' and mine[1]==q['fail_reason'] or qk=='KNOWN' and max(abs(mine[2]/float(q['final_equity'])-1),abs(mine[3]-float(q['CAGR'])),abs(mine[4]-float(q['MDD'])))<1e-9)
    if qk=='KNOWN' and mine[0]=='KNOWN': mx=max(mx,abs(mine[3]-float(q['CAGR'])),abs(mine[4]-float(q['MDD'])))
    rows.append(dict(path=pid,claude_status=mine[0],claude_fail=mine[1],claude_final=mine[2],claude_CAGR=mine[3],claude_MDD=mine[4],queen_status=q['status'],queen_fail=q['fail_reason'],queen_final=q['final_equity'],queen_CAGR=q['CAGR'],queen_MDD=q['MDD'],match=ok))
w=csv.DictWriter(open(os.path.join(HERE,'claude_vs_queen_25_paths.csv'),'w',newline=''),fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('paths',len(rows),'match',sum(r['match'] for r in rows),'max abs diff CAGR/MDD %.2e'%mx)
for fam in ('W10H2K1','W12H4K1'):
    v=[r['claude_CAGR'] for r in rows if r['path'].startswith(fam+'P') and r['claude_status']=='KNOWN']
    d=[r['claude_MDD'] for r in rows if r['path'].startswith(fam+'P') and r['claude_status']=='KNOWN']
    print(fam,'known',len(v),'median CAGR %.2f%% min %.2f%% max %.2f%% median MDD %.2f%%'%(st.median(v)*100,min(v)*100,max(v)*100,st.median(d)*100))
r=[r for r in rows if r['path']=='W12H12K1P1'][0]; print('W12H12K1P1 CAGR %.2f%% MDD %.2f%% final %.4f'%(r['claude_CAGR']*100,r['claude_MDD']*100,r['claude_final']))
os.remove(os.path.join(A,'rank3.pkl'))
