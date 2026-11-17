"""Run the extended ablation cells (experiments/ablation/c*.py) as one script.

The cell files are the single source of truth: tools/build_notebook.py inserts
the same files into the notebook. Usage (from the repository root):

    python experiments/run_ablation.py                 # all cells
    python experiments/run_ablation.py c00 c01 c04     # selected cells (c00 and c01 always run; they define shared objects)

Environment: MCF_ABL_DIR (default results/ablation), MCF_REPEATS (5), MCF_JOBS (-1).
"""
import glob
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.environ.setdefault('MCF_ABL_DIR', 'results/ablation')
pd.set_option('display.width', 250, 'display.max_columns', 30, 'display.max_colwidth', 60)

cells = sorted(glob.glob('experiments/ablation/c*.py'))
wanted = sys.argv[1:]
if wanted:
    cells = [c for c in cells if os.path.basename(c)[:3] in ('c00', 'c01')
             or any(os.path.basename(c).startswith(w) for w in wanted)]

ns = {'display': lambda obj: print(obj.to_string() if hasattr(obj, 'to_string') else obj),
      '__name__': '__main__'}
plt.show = lambda *a, **k: plt.close('all')
for cell in cells:
    t0 = time.time()
    print(f'\n######## {cell} ########', flush=True)
    exec(compile(open(cell).read(), cell, 'exec'), ns)
    print(f'######## {cell} done in {time.time() - t0:.0f}s', flush=True)
