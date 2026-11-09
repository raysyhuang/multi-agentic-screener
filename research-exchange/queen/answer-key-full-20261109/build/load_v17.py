# Load the frozen v17 strict4 engine read-only (same as w10h2-w12h4-ledgers-20261109/load_engine.py). No files written in V17.
import os,sys,time
from pathlib import Path
V17=Path('/workspace/second-sleeve-v17-targeted-ca-20261108')
def load():
    os.chdir(V17/'engine');sys.path.insert(0,str(V17/'engine'))
    ns={'__name__':'v18_dedupe'}
    t=time.time();exec(compile(Path('engine432_strict4.py').read_text(),'engine432_strict4.py','exec'),ns)
    print('engine loaded',round(time.time()-t,1),'s',flush=True)
    return ns
