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

### 3. Extended ablation suite (`experiments/ablation/c00`–`c11`)
Protocol (identical for every variant): raw UCI Cleveland data; imputation, ECG-proxy
features, scaling and resampling fitted **inside** each training fold; 5×10 repeated
stratified CV (50 folds; 30 for neural-network and robustness runs); fixed Section-15
hyper-parameters; 95% CIs from the Nadeau–Bengio corrected resampled t-test; Holm-adjusted
p-values. Run with `python experiments/run_ablation.py` (results are cached in
`results/ablation/cache/`, which is not committed).

| Cell | Experiment |
|------|------------|
| c01 | Component ablation (21 variants: feature groups, SMOTE, each ensemble member, hard voting, stacking) |
| c02 | Controls (non-ECG features, Gaussian noise, row-shuffled proxy block) + 100-permutation label test |
| c03 | Imbalance strategies (none, class weights, ROS, SMOTE pre/post, SMOTE-NC, Borderline, ADASYN) + SMOTE realism + class counts |
| c04 | Leakage (global preprocessing, SMOTE before CV), nested tuning, 200 random hold-out splits |
| c05 | 16 robustness scenarios (label noise, MCAR, measurement noise, training size, missing tests) + subgroups |
| c06 | Ordinal vs one-hot encoding, Platt/isotonic calibration |
| c07 | External validation on Hungary, Switzerland, VA Long Beach; chol = 0 edge case |
| c08 | SHAP stability across folds/seeds, permutation-importance agreement, dependence and local plots |
| c09 | Dual-branch DL: concat vs cross-attention (L=1) vs linear vs token-level attention |
| c10 | PTB-XL ECG branch: leads, tsfresh set, 12-lead, VarianceThreshold leakage, sample size, signal quality |
| c11 | Claim-by-claim verdict table |

### 4. Key findings (numbers from `results/ablation/tables/`)
- Full system AUC 0.902 [0.858, 0.945]; clinical-only AUC 0.906 [0.864, 0.947];
  ΔAUC +0.004 [−0.009, +0.017] in favour of clinical-only. **RQ1: the ECG-proxy block does not improve prediction.**
- No feature group changes AUC by more than 0.001. Correcting the downslope bug changes nothing.
- 18 Gaussian-noise columns (AUC 0.899) or 18 non-ECG engineered columns (0.903) do as well as the proxy block.
- The permutation test confirms real signal (AUC 0.907 vs 0.497 null, p = 0.0099).
- SMOTE: −SMOTE ΔAUC −0.002 (n.s.); all eight imbalance strategies are within ±0.005.
  100% of SMOTE rows have non-integer categorical codes; 13.6% have inconsistent derived features.
- Leakage: SMOTE before CV inflates AUC to 0.916; global preprocessing has a negligible effect here.
- Hold-out: over 200 random 80/20 splits, fused accuracy is 0.833 (95% range 0.754–0.918). The paper's seed-42
  split is at the 95th percentile (accuracy) and 99th (AUC). Fused beats clinical on only 41% of splits.
- Robustness: across 16 perturbations ΔAUC (fused − clinical) lies between −0.014 and +0.004, and none is significant,
  including the "no angiography/thallium" edge case.
- Subgroups: no subgroup benefits; AUC is lowest for symptomatic chest pain (0.80) and age ≥ 55 (0.85).
- Calibration: soft voting is under-confident (slope 1.37); Platt scaling helps (1.13); isotonic over-corrects (0.56).
- Ensemble: soft voting ≈ SVM alone ≈ logistic regression; it beats RF/GBM alone; hard voting is worse (−0.032).
