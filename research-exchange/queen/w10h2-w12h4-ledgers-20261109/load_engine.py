# Load the frozen v17 strict4 engine (read-only) into a namespace, exactly as run_actual_annual.py does.
import os,sys,time
from pathlib import Path
V17=Path('/workspace/second-sleeve-v17-targeted-ca-20261108')
def load():
    cwd=os.getcwd();os.chdir(V17/'engine');sys.path.insert(0,str(V17/'engine'))
    ns={'__name__':'annual_attribution_AD'}
    t=time.time();exec(compile(Path('engine432_strict4.py').read_text(),'engine432_strict4.py','exec'),ns)
    os.chdir(V17/'engine')  # engine expects relative paths during runs
    print('engine loaded',round(time.time()-t,1),'s',flush=True)
    return ns
