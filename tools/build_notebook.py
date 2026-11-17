"""Insert the extended ablation (experiments/ablation/c*.py) into the notebook.

1. Runs tools/fix_original_notebook.py (code/text corrections to the v4 notebook).
2. Inserts the extended ablation cells as Section 17B after Section 17, plus
   revision notes.
3. Executes the WHOLE notebook top to bottom with nbclient (cwd = notebook/,
   ablation results cached in results/ablation), so every output matches the code.

Usage (repository root):  python tools/build_notebook.py [--no-exec]
"""
import copy
import glob
import os
import sys

import nbformat
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_NB = os.path.join(ROOT, 'notebook', '_fixed_base.ipynb')   # produced by tools/fix_original_notebook.py
OUT_DIR = os.path.join(ROOT, 'notebook')
OUT_NB = os.path.join(OUT_DIR, 'MultiCardioFusion_IEEE_2_extended_ablation.ipynb')
STANDALONE = os.path.join(OUT_DIR, 'extended_ablation_standalone.ipynb')

INTRO = """## 17B. EXTENDED ABLATION STUDY AND EDGE CASES

The single-split ablation above (Section 17) uses one 61-patient hold-out, no uncertainty estimates
and preprocessing fitted on all 303 rows. Following the reviewers' request (action-plan row 16),
this section re-tests every claim under **one controlled protocol**:

| Item | Setting |
|---|---|
| Data | Raw UCI Cleveland files; missing values kept and imputed **inside** each training fold |
| Splits | Repeated stratified 10-fold CV (5 × 10 = 50 folds), identical for every variant |
| Pipeline | median imputation → ECG-proxy features → StandardScaler → resampling (train only) → classifier |
| Hyper-parameters | Fixed Section-15 values for every variant (identical tuning budget) |
| Metrics | AUC, accuracy, sensitivity, specificity, F1, Brier score, calibration slope |
| Uncertainty | 95% CI from the Nadeau–Bengio corrected resampled t-test; paired ΔAUC; Holm correction |

Sub-sections: 17B-1 component ablation · 17B-2 control experiments and permutation test ·
17B-3 imbalance strategies · 17B-4 protocol and leakage · 17B-5 robustness and edge cases ·
17B-6 encoding and calibration · 17B-7 external validation on three other hospitals ·
17B-8 SHAP stability · 17B-9 cross-attention · 17B-10 PTB-XL ECG branch · 17B-11 verdict.

Results are written to `outputs/ablation/` (set `MCF_ABL_DIR` to change); completed runs are cached,
so re-running a cell is fast. Set `MCF_REPEATS=3` for a quicker run.
"""

CELL_NOTES = {
    'c00': None,
    'c01': '### 17B-1. Component ablation\nOne component removed at a time; ΔAUC is paired by fold against the full system.',
    'c02': '### 17B-2. Control experiments\nDoes the ECG-proxy block beat equally sized blocks of non-ECG features, pure noise, or a row-shuffled copy of itself?',
    'c03': '### 17B-3. Class-imbalance strategies\nResampling is done inside training folds only. Also checks how realistic SMOTE patients are.',
    'c04': '### 17B-4. Evaluation protocol and leakage\nHow much do leaky preprocessing, SMOTE-before-CV, tuning choices and the choice of hold-out split change the numbers?',
    'c05': '### 17B-5. Robustness and edge cases\nLabel noise, missing values, measurement noise, smaller training sets, unavailable tests, and patient subgroups.',
    'c06': '### 17B-6. Ordinal encoding and probability calibration',
    'c07': '### 17B-7. External validation (Hungary, Switzerland, VA Long Beach)\nTrained on Cleveland only; the other sites have different prevalence and many missing tests.',
    'c08': '### 17B-8. SHAP stability and agreement with permutation importance',
    'c09': '### 17B-9. Dual-branch network: does cross-attention help?',
    'c10': '### 17B-10. PTB-XL ECG branch (D2)\nRequires the PTB-XL records (downloaded from PhysioNet if missing; ~25 MB for 1,212 records).',
    'c11': '### 17B-11. Verdict: which claims survive?',
}

LEGACY_NOTES = {
    # cell index in the original notebook → note prepended to its markdown/code
    36: ('\n\n> **Note (revision):** this single-split ablation is kept for transparency. It uses one 61-patient '
         'hold-out without confidence intervals, so it cannot support claims about individual components. '
         'See **Section 17B** for the controlled, repeated-CV version.'),
    41: ('\n\n> **Note (revision):** with p = 1.00 (t-test) and p = 0.72 (McNemar) the fused model is **not** '
         'better than the clinical model. Section 17B-1/17B-4 confirm this with 50-fold paired tests.'),
}


def cell_sources():
    for path in sorted(glob.glob(os.path.join(ROOT, 'experiments', 'ablation', 'c*.py'))):
        tag = os.path.basename(path)[:3]
        yield tag, open(path).read().rstrip() + '\n'


def build_standalone():
    nb = nbformat.v4.new_notebook()
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb.cells.append(nbformat.v4.new_markdown_cell(INTRO))
    for tag, src in cell_sources():
        if CELL_NOTES.get(tag):
            nb.cells.append(nbformat.v4.new_markdown_cell(CELL_NOTES[tag]))
        nb.cells.append(nbformat.v4.new_code_cell(src))
    return nb


def execute(nb):
    os.environ.setdefault('MCF_ABL_DIR', '../results/ablation')
    os.environ['MPLBACKEND'] = 'module://matplotlib_inline.backend_inline'
    NotebookClient(nb, timeout=7200, kernel_name='python3',
                   resources={'metadata': {'path': OUT_DIR}}).execute()
    strip_stderr(nb)
    return nb


def strip_stderr(nb):
    """Drop stderr streams (TensorFlow start-up logs) from executed cells."""
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = [o for o in cell.outputs if not (o.output_type == 'stream' and o.get('name') == 'stderr')]


def merge(executed):
    nb = nbformat.read(SRC_NB, as_version=4)
    for idx, note in LEGACY_NOTES.items():
        nb.cells[idx].source = nb.cells[idx].source.rstrip() + note
    insert_at = 38                       # after the Section-17 ablation code cell
    new_cells = [copy.deepcopy(c) for c in executed.cells]
    nb.cells[insert_at:insert_at] = new_cells
    nb.cells.insert(1, nbformat.v4.new_markdown_cell(REVISION_NOTE))
    return nb


REVISION_NOTE = """> ### Revision note (v5): read Section 17B first
> The abstract above describes the original (v4) analysis. Its headline numbers (">96% accuracy, AUC > 0.97",
> later 0.9535 AUC / 88.52% accuracy, "+7.04 pp fusion gain") come from a single 61-patient hold-out split with
> preprocessing fitted on all patients. **Section 17B** (inserted after Section 17) re-tests every claim under a
> controlled, leakage-free protocol: 5×10 repeated CV, identical splits and tuning budget, corrected 95% CIs,
> 94 configurations, external validation on three further hospitals, SHAP stability, cross-attention and PTB-XL ablations.
>
> Main results: full system AUC ≈ 0.90; clinical-only AUC ≈ 0.91; the ECG-proxy block gives **no** measurable gain
> (ΔAUC CI includes 0) and performs like added noise; the seed-42 split is at the 95th percentile of 200 random splits.
> Sections 18–25 are kept unchanged for transparency; notes in Sections 17 and 19 point to the new results.
> The revised paper is `paper/MultiCardioFusion_Revised_v5.pdf`.
"""


if __name__ == '__main__':
    import subprocess
    os.makedirs(OUT_DIR, exist_ok=True)
    subprocess.check_call([sys.executable, os.path.join(ROOT, 'tools', 'fix_original_notebook.py')])
    full = merge(build_standalone())          # corrected original + Section 17B (unexecuted)
    nbformat.validate(full)
    if '--no-exec' not in sys.argv:
        full = execute(full)                  # execute the WHOLE notebook top to bottom
    nbformat.write(full, OUT_NB)
    # stand-alone copy of Section 17B with the outputs of the full run
    start = next(i for i, c in enumerate(full.cells) if c.source.startswith('## 17B.'))
    end = next(i for i, c in enumerate(full.cells) if c.source.startswith('## 18.'))
    sa = nbformat.v4.new_notebook(); sa.metadata = full.metadata; sa.cells = full.cells[start:end]
    nbformat.write(sa, STANDALONE)
    print(f'wrote {OUT_NB} ({len(full.cells)} cells) and {STANDALONE}')
