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
- PTB-XL ECG branch (D2): Lead II Efficient tsfresh AUC 0.819; V5 equal (0.819); Lead I 0.724, V1 0.752;
  Minimal-10 features 0.666 (significant drop); adding 12-lead statistics to Lead II gives +0.004 (n.s.);
  1,212 vs 303 records does not raise AUC (0.825 vs 0.823; only accuracy and CI width improve);
  noisy recordings have AUC 0.780 vs 0.845 for clean ones.
- Cross-attention (L = 1) is provably a linear map (attention weights all exactly 1, query has no effect). DL AUC:
  L=1 attention 0.887 ≈ linear 0.885 ≈ concatenation 0.878; token-level attention 0.893 = clinical MLP 0.893.
- SHAP: fold-to-fold ρ = 0.90, seed ρ = 0.95, SHAP vs permutation importance ρ = 0.27; cp/ca/thal are in the top 5 in
  100% of folds; the proxy block receives ~25% of |SHAP| while adding 0 AUC (credit redistributed from oldpeak etc.).
- External: Hungary 0.889, Switzerland 0.772, VA 0.709 (fused ≈ clinical everywhere). Sensitivity collapses because
  missing ca/thal are imputed with "healthy" medians; training without them restores sensitivity to 0.77–0.85.
- **Verdict (E11): none of the nine quantitative v4 claims is supported.** 94 configurations were evaluated.

### 5. Deliverables
- `notebook/MultiCardioFusion_IEEE_2_extended_ablation.ipynb`: the original notebook plus a revision note (cell 2),
  notes in Sections 17 and 19, and the new **Section 17B** (executed, outputs embedded).
  `notebook/extended_ablation_standalone.ipynb` has the same section on its own (runs without TF/PTB-XL for 17B-1…8).
  Rebuild with `python tools/build_notebook.py`.
- `paper/MultiCardioFusion_Revised_v5.pdf` (IEEEtran, 17 pages): retitled and reframed per reviewer row 0, new
  Section VI (Extended Ablation Study, 10 subsections, 20+ tables generated from `results/`), and revised SHAP,
  discussion and conclusion. Rebuild with `paper/build.sh`. Reviewer rows are mapped in `paper/RESPONSE_TO_REVIEWERS.md`.

### 6. Open items for the author
- Add the four reviewer-supplied XAI references (row 6). Their details were not in the action plan.
- Extend Table I with the rigor columns (row 7). This needs facts from each cited paper.
- Confirm reference [23] (replaced with the Lundberg et al. 2018 Nat. Biomed. Eng. paper).
- Rotate the Kaggle API key that was in the notebook.

### 7. Fact-check as a journal referee (see `paper/REFEREE_REPORT.md`)
- **Citations** were checked against the published papers and indexing records.
  - **Corrected:** Arabasadi (Z-Alizadeh Sani dataset, 93.85%), Mohan (88.7%), Latha & Jeeva (85.48%),
    Haq (89% with LR + Relief; "AUC 0.950" unsupported) and Attia (ECG-only, LV dysfunction).
  - **Removed:** Khan (not PTB-XL), Zhang 2022 and Bi 2021 (could not be found), Hossain 2023 (wrong authors and
    journal; replaced by Kligfield 2007), and 9 uncited entries.
  - **Added:** Hanley & McNeil, Riley 2020, TRIPOD+AI, the PTB-XL v1.0.3 DOI, scikit-learn, imbalanced-learn and
    FT-Transformer.
- **Overclaims in my own v5 draft were fixed:**
  - Non-significant ensemble effects were described as real.
  - The calibration claim did not match the results.
  - The SHAP dependence description was wrong for `ca` and `thal`.
  - The provenance of `ca` and `thal` was overstated.
  - The model is now stated to be a post-test classifier that is not validated for clinical use.
- **Errors in the original notebook** were fixed in `tools/fix_original_notebook.py` (40 patches), and the notebook
  was re-executed end to end:
  - the protocol-mismatched "+7.04 pp";
  - cross-validation and hyper-parameter tuning on SMOTE-augmented data;
  - **hold-out leakage** (a model from the last CV fold scored on the hold-out patients: AUC 0.998 → 0.931 once fixed);
  - the downslope code;
  - pandas imputation that silently did nothing;
  - mislabelled models and ablation variants;
  - unsupported literature claims in cells 44 and 51.
- **Reproducibility:** all 94 configurations re-run from an empty cache give identical tables
  (`paper/REPRODUCIBILITY.md`).
- **Corrected notebook, same model, seed-42 split:** fused − clinical = +3.3 pp on the hold-out but −3.0 pp under
  10-fold CV of the same patients.
