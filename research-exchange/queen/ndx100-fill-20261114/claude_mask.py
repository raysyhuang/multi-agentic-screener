import gzip,csv,os
from engine import MEM,ME
H=os.path.dirname(os.path.abspath(__file__))
def load_pack():
    px={}
    for r in csv.DictReader(gzip.open('/workspace/momentum432-inputs-pack-full-20261109/rank_prices_month_end.csv.gz','rt')):
        px.setdefault(r['month_end'],set()).add(r['engine_ticker'])
    return px
# WBA and BKR have pack prices but Claude's engine still counted them as missing (its missing_prices_request lists them: 112+24 member-months)
CLAUDE_ENGINE_DROPS={'WBA','BKR'}
def build_mask():
    pack=load_pack(); A={'PCLN':'BKNG'}
    cm={r['month_end']:r['tickers'].split() for r in csv.DictReader(open(os.path.join(H,'repo/research-exchange/claude/ndx100-partial-test-20261114/ndx100_membership_month_end.csv')))}
    mask={}; diff=[]
    for d,ts in cm.items():
        mask[d]={t for t in ts if A.get(t,t) in pack.get(d,()) and t not in CLAUDE_ENGINE_DROPS}
        if d in MEM and set(ts)!=set(MEM[d]): diff.append((d,sorted(set(ts)-set(MEM[d])),sorted(set(MEM[d])-set(ts))))
    return mask,diff,pack
if __name__=='__main__':
    m,diff,_=build_mask()
    print('membership diffs vs mine:',len(diff),diff[:5])
    import statistics as st
    ds=[d for d in ME if '2014-12-31'<=d<'2024-12-31']
    print('avg cov',st.mean(len(m[d])/len(MEM[d]) for d in ds))
