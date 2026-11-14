# NDX post rule (v1.1 engine, imported read-only): Top3 25bp 2014-12-31..2026-11-13, plus a variant restricted each month to
# NDX members that were also S&P 500 PIT members (S&P snapshot from the v17 legacy membership file via spx.pkl).
import sys,os,pickle,bisect
B='/workspace/ndx-vs-spx-overlap-20261114/build'
V='/workspace/ndx100-fill-20261114-v1.1'; sys.path.insert(0,V); os.chdir(V)
import engine as E
os.chdir(B)
CAN={'FB':'META','PCLN':'BKNG','YHOO':'AABA'}
s=pickle.load(open(B+'/spx.pkl','rb')); md=[d for d,_ in s['sp_mem']]; sm=[set(x) for _,x in s['sp_mem']]
def sp_at(d): return sm[bisect.bisect_right(md,d)-1]
i0,i1=E.IDX['2014-12-31'],E.IDX['2026-11-13']
out={}
res=E.run(3,i0,i1,(),0.0025); out['top3']=dict(rec=res['rec'],navs=res['navs'],dnav=res['dnav'],contrib=res['contrib'])
mask={E.ME[i]:{t for t in E.MEM[E.ME[i]] if CAN.get(t,t) in sp_at(E.ME[i])} for i in range(i0,i1)}
E.MASK=mask; E._SIG.clear()
res2=E.run(3,i0,i1,(),0.0025); out['top3_spx_members_only']=dict(rec=res2['rec'],navs=res2['navs'],dnav=res2['dnav'],contrib=res2['contrib'])
E.MASK=None
P,TR=E.qqq()
out['mem']={E.ME[i]:list(E.MEM[E.ME[i]]) for i in range(i0,i1)}
out['qqq']={k:v for k,v in P.items() if '2014-12-01'<=k}
out['days']=[E.ME[i0]]+E.days_between(E.ME[i0],E.ME[i1]); out['ME']=E.ME[i0:i1+1]
pickle.dump(out,open(B+'/ndx.pkl','wb'))
print('top3',E.stats(res)); print('spx-only',E.stats(res2))
