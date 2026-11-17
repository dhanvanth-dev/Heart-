# MultiCardioFusion: worklog

This file tracks everything done to the project in this repository, newest entries last.
Each entry says what was done, why, and where the result lives.

## Goal (from the professor's review)

Expand the ablation study with more experiments and edge cases so that the paper's claims
(especially "the ECG-proxy layer / fusion improves on clinical-only prediction") can be
validated or refuted under a rigorous protocol. Update the notebook and the paper to match.

Reviewer action plan: `original/multicardio_fusion_paper_draft.docx`. The rows this work
addresses are 16 (ablation), 12/15 (resampling), 8 (leakage), 13 (calibration),
14 (cross-attention), 6/18 (SHAP stability), 17 (non-significant results).

## Repository layout

| Path | Contents |
|------|----------|
| `original/` | Files as received: notebook (Kaggle key redacted), v4 paper PDF, reviewer action plan (.docx) |
| `data/uci/` | UCI heart-disease `processed.*.data` for Cleveland, Hungary, Switzerland, VA Long Beach |
| `data/ptbxl/` | PTB-XL metadata (`ptbxl_database.csv`, `scp_statements.csv`) |
| `experiments/ablation/` | Source of the new notebook cells (single source of truth) |
| `experiments/run_ablation.py` | Runs the ablation cells as a script, writes `results/` |
| `results/ablation/` | CSV tables and figures produced by the extended ablation |
| `notebook/` | Updated notebook with the extended ablation section |
| `paper/` | Revised paper (v5) source and PDF |

## Log

### 1. Repository setup
- Copied the three uploaded files into `original/`.
- **Security:** notebook cell 10 contained a hard-coded Kaggle API key. It was replaced with
  `os.environ.get('KAGGLE_KEY')` before the first commit. The key was exposed in the file you
  shared, so **rotate it on kaggle.com** (Account → API → Expire token).
- Downloaded UCI data for all four sites (Cleveland is the training cohort; the other three
  are used as external test sites) and PTB-XL metadata from PhysioNet.

### 2. Issues found while reading the notebook and paper
These shape the ablation design; each one is tested explicitly.

1. **Protocol mismatch in the headline claim.** The "+7.04 pp" gain compares the fused model's
   *hold-out* accuracy (0.8852, 61 patients, seed 42) with the clinical model's *10-fold CV*
   accuracy (0.8148). Under the same 10-fold protocol the notebook itself reports
   clinical XGB = fused XGB = 0.8148 (t = 0, p = 1).
2. **The cross-attention dual-branch network has no code.** The paper describes MCF-BiModNet
   with multi-head cross-attention; notebook cell 31 trains a clinical-only MLP. With a
   sequence length of 1, softmax over a single key is always 1, so the attention block is
   mathematically a linear projection of the clinical branch.
3. **Coding bug:** UCI `slope` is 1 = upsloping, 2 = flat, 3 = downsloping, but the proxy
   feature `downslope = (slope == 2)` flags flat slopes.
4. **Five of the 23 "ECG-proxy" columns are exact copies of clinical columns** (restecg,
   oldpeak, slope, thalach, exang), so the fused matrix duplicates them.
5. **Leakage:** median imputation and StandardScaler are fitted on all 303 rows before
   the split / CV.
6. **SMOTE after feature engineering** creates synthetic rows whose derived features are
   inconsistent (e.g. synthetic `oldpeak_sq` is not the square of synthetic `oldpeak`).
7. Table IV labels 10-fold CV numbers as hold-out results; the "MCF-BiModNet XGB" and
   "Soft-Voting" rows are identical.
8. Class counts: 164 healthy (54.1%) / 139 disease (45.9%); the paper text assigns the
   percentages the other way round.
