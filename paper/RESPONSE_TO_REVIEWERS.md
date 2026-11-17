# Response to the reviewer action plan (v4 → v5)

Row numbers follow `original/multicardio_fusion_paper_draft.docx`. "§" refers to sections of
`paper/MultiCardioFusion_Revised_v5.pdf`; "17B-n" refers to notebook sections in
`notebook/MultiCardioFusion_IEEE_2_extended_ablation.ipynb`.

| Row | Severity | Issue | Status | Where |
|-----|----------|-------|--------|-------|
| 0 | Critical | Title and claims over-promise multimodal fusion | **Done.** Title changed to the suggested wording. The abstract, introduction, contributions and conclusion are reframed as ECG-proxy-enhanced clinical prediction, and the fusion claim is tested and reported as not supported | Title, Abstract, §I, §IX |
| 1 | Critical | Distinguish true fusion / proxy fusion / ECG-only benchmarking | **Done.** Three-category taxonomy added, with this work placed in categories (ii) and (iii) | §I |
| 2 | Critical | Reframe novelty and contributions | **Done.** Five reframed contributions, each delivered in the paper | §I |
| 6 | Critical | SHAP caveats, stability, permutation importance | **Done** (caveats, fold/seed stability, permutation importance, duplicated-feature attribution). **Author action:** add the four XAI references the reviewer supplied (Patlatzoglou et al. 2025; Taleban et al. 2026; the arXiv arrhythmia-XAI paper; Comput. Biol. Med. 2024). Their full details were not in the action plan, so they were not invented | §II-D, §VII, 17B-8 |
| 8 | Critical | Target definition, class counts, imputation leakage, ordinal coding, demographics | **Done.** Angiographic definition, corrected counts (164/54.1% no CAD, 139/45.9% CAD), in-fold imputation, ordinal vs one-hot experiment, demographics table | §III-A, Table demo, §VI-F |
| 12 | Critical | SMOTE on mixed data; compare strategies | **Done.** Eight strategies compared (none, class weights, ROS, SMOTE pre/post, SMOTE-NC, Borderline, ADASYN), plus a SMOTE realism check | §VI-C, 17B-3 |
| 16 | Critical | Rigorous ablation with CIs, same splits and budget | **Done.** 21-variant table with Acc, AUC, Sens, Spec, F1, Brier, ΔAUC, 95% CI and Holm p, plus controls, robustness, external validation and a claim-by-claim verdict | §VI, 17B-1…11 |
| 3 | Medium | Justify single lead | **Done.** Justification added and quantified with a lead/12-lead ablation | §III-C, §VI-J |
| 4 | Medium | PTB-XL label semantics; Ribeiro/Wagner | **Done** | §II-B, §III-C |
| 5 | Medium | True vs proxy fusion in Related Work | **Done** | §II-C |
| 7 | Medium | Extend Table I with rigor columns | **Partly done.** Caption states that protocols differ, the attribution to [12] is corrected and the own row is updated. **Author action:** the extra columns (validation protocol, CI reported, leakage prevention, etc.) need facts from each cited paper that could not be verified here | Table I |
| 9 | Medium | Dataset → model mapping table | **Done** | Table mapping |
| 10 | Medium | Split the overloaded algorithm | **Done.** Three algorithms; SHAP is explicitly post-hoc | §IV-B |
| 11 | Minor | Justify tsfresh on 10-s segments | **Done** | §III-D |
| 13 | Medium | Calibration, voting weights, tuning parity | **Done.** Calibration slope and Brier reported, Platt/isotonic compared, stacking as learned weights, fixed shared budget plus nested tuning | §IV-E, §VI-D, §VI-F |
| 14 | Medium | Cross-attention with sequence length 1 | **Done.** Derived and numerically verified that it is a linear map; six-network ablation; cross-attention removed from the proposed system | §IV-C, §VI-I, 17B-9 |
| 15 | Medium | Class-count table; recall/spec/Brier/calibration | **Done** | Table counts, §VI-C, §VI-F |
| 17 | Medium | Discussion: protocol-aware comparison, clinical meaning, non-significant tests, RQ wrap-up | **Done** | §VIII |
| 18 | Medium | Beeswarm, dependence, local TP/FN/FP/TN, stability, permutation importance | **Done** (dependence, local explanations, stability, permutation importance; the v4 beeswarm is in the notebook, §18) | §VII |
| 19–21 | Minor | Formatting / tracking notes | Re-typeset in IEEEtran. References renumbered after the fact-check removed fabricated, unverifiable and uncited entries (see `REFEREE_REPORT.md`) | — |

## Corrections to v4 found during the revision
- The "+7.04 pp" fusion gain compared a hold-out score with a CV score; under one protocol the gain is 0.
- The cross-attention MCF-BiModNet described in v4 had no code. The notebook's "DL dual-branch" was a clinical-only MLP.
- Table IV labelled CV results as hold-out results, and its two "★" rows were the same model.
- The v4 unimodal baselines were computed by CV on the SMOTE-augmented training set (synthetic rows in test folds).
- The `downslope` proxy feature tested `slope == 2` (flat) instead of 3 (downsloping).
- Several v4 literature values and references were wrong or could not be found. The full list is in `REFEREE_REPORT.md` §1:
  Arabasadi (different dataset), Mohan, Latha & Jeeva, Haq ("AUC 0.950" unsupported), Khan (not PTB-XL),
  Zhang 2022 and Bi 2021 (not found), Hossain 2023 (wrong authors/journal), Attia (ECG-only, LV dysfunction).
- The v4 notebook scored a "hold-out" XGBoost with the model from the last CV fold, which had seen most hold-out
  patients (AUC 0.998 on re-execution).
- Reference [23] was listed as a Lundberg et al. 2018 *npj Digit. Med.* sepsis paper. It was replaced with the
  Lundberg et al. 2018 *Nat. Biomed. Eng.* hypoxaemia paper. **Please confirm** which paper you intended.
- A Kaggle API key was hard-coded in the notebook. It has been removed. **Rotate it.**
