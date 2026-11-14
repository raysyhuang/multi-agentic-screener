"""Run Claude's own engine (repo @4547367, read-only) and dump its Top3 monthly picks/returns for the 2015-2024 partial test, to diff with Queen's."""
import os,csv,collections
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),'repo/research-exchange/claude/ndx100-partial-test-20261114')
src=open(os.path.join(D,'ndx_partial_test.py')).read().split("P,first=PACKS['2015-2025']")[0]
__file__=os.path.join(D,'ndx_partial_test.py'); exec(src.replace("H=os.path.dirname(os.path.abspath(__file__))","H=D",1))
P,first=PACKS['2015-2025']
E=E2(P,first,True,())
for t,last in BAD_TAIL.items(): E.px[t]={d:v for d,v in E.px[t].items() if d<=last}
for d in E.ME: E.el[d]=[t for t in mem[d] if d in E.px.get(t,{})]
out=[]
for m in range(E.i0,E.i0+120):
    names=E.ranking(m,'M6avg')[:3]; rs=[E.ret(t,E.ME[m],E.ME[m+1]) for t in names]
    out.append(dict(signal=E.ME[m],picks=' '.join(names),rets=' '.join('NA' if r is None else '%+.4f'%r for r in rs)))
w=csv.DictWriter(open('out/claude_engine_top3_picks_2015_2024.csv','w',newline=''),fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
