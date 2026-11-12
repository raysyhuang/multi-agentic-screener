import sys; exec(open('/workspace/taiwan-wh-grid-20261112/engine.py').read())
print('missing price tickers:',MISSING, 'LAST',LAST, 'master days',nD)
for (W,H) in [(6,6),(12,12)]:
    rows=run_cell('TW150',W,H,12)
    for d in summarize(rows): print(W,H,{k:(round(v,4) if isinstance(v,float) else v) for k,v in d.items()})
