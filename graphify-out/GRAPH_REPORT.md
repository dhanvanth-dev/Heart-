# Graph Report - heart-  (2026-11-17)

## Corpus Check
- Corpus is ~42,384 words - fits in a single context window. You may not need a graph.

## Summary
- 349 nodes · 483 edges · 28 communities (21 shown, 7 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 44 edges (avg confidence: 0.86)
- Token cost: 273,579 input · 0 output

## Community Hubs (Navigation)
- v5 Paper Findings & Citations
- Ablation Engine & Statistics
- Pipeline & Model Factory
- Literature Baselines
- Ablation Runner & Notebook Build
- Dual-Branch Network Code
- Robustness Perturbations
- v4 Defects & Corrections
- Paper Table Generator
- Worklog Ablation Cells
- Project Overview & Datasets
- v4 ECG Branch & Future Work
- Imbalance & Dependencies
- v4 Fusion Architectures
- Reviewer Reframing Requests
- v4 Datasets & SMOTE
- Soft-Voting Ensemble Members
- SHAP & XGBoost (v4)
- SHAP Caveats & XAI Refs
- Verdict Code
- Calibration Findings
- Cross-Attention L=1 Issue
- Class Count Correction

## God Nodes (most connected - your core abstractions)
1. `MultiCardioFusion Framework` - 33 edges
2. `MultiCardioFusion (v5 revision)` - 27 edges
3. `Extended ablation suite (experiments/ablation c00-c11)` - 15 edges
4. `Response to reviewer action plan (v4 to v5)` - 14 edges
5. `MultiCardioFusion Reviewer Comment Action Plan` - 13 edges
6. `MCF-BiModNet Algorithm` - 10 edges
7. `ProxyFeatures` - 9 edges
8. `HardVoting` - 9 edges
9. `evaluate()` - 9 edges
10. `DualBranchNet` - 9 edges

## Surprising Connections (you probably didn't know these)
- `SMOTE after feature engineering yields inconsistent rows` --semantically_similar_to--> `SMOTE on mixed-type small clinical data synthesizes unrealistic patients`  [INFERRED] [semantically similar]
  WORKLOG.md → graphify-out/converted/multicardio_fusion_paper_draft_85d2d521.md
- `Response to reviewer action plan (v4 to v5)` --references--> `Hard-coded Kaggle API key (security issue)`  [INFERRED]
  paper/RESPONSE_TO_REVIEWERS.md → WORKLOG.md
- `Response to reviewer action plan (v4 to v5)` --references--> `Protocol mismatch in +7.04 pp headline claim`  [INFERRED]
  paper/RESPONSE_TO_REVIEWERS.md → WORKLOG.md
- `Response to reviewer action plan (v4 to v5)` --references--> `Cross-attention with sequence length 1 is a linear map`  [INFERRED]
  paper/RESPONSE_TO_REVIEWERS.md → WORKLOG.md
- `Response to reviewer action plan (v4 to v5)` --references--> `downslope proxy feature coding bug (slope==2)`  [INFERRED]
  paper/RESPONSE_TO_REVIEWERS.md → WORKLOG.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Reviewer critical concerns addressed by ablation cells** — graphify_out_converted_multicardio_fusion_paper_draft_85d2d521_rigorous_ablation_table, graphify_out_converted_multicardio_fusion_paper_draft_85d2d521_smote_mixed_data_concern, graphify_out_converted_multicardio_fusion_paper_draft_85d2d521_cross_attention_concern, graphify_out_converted_multicardio_fusion_paper_draft_85d2d521_shap_caveats, worklog_extended_ablation_suite [INFERRED 0.85]
- **Defects found in v4 notebook and paper** — worklog_protocol_mismatch, worklog_downslope_bug, worklog_duplicated_proxy_columns, worklog_preprocessing_leakage, worklog_smote_after_feature_engineering, worklog_class_count_error [EXTRACTED 1.00]
- **Leakage-free statistical evaluation protocol** — worklog_evaluation_protocol, worklog_nadeau_bengio_t_test, worklog_holm_adjustment, worklog_extended_ablation_suite [EXTRACTED 1.00]
- **Soft-Voting Ensemble Members** — paper_v4_extracted_text_xgboost, paper_v4_extracted_text_svm, paper_v4_extracted_text_gradient_boosting, paper_v4_extracted_text_random_forest, paper_v4_extracted_text_soft_voting_ensemble [EXTRACTED 1.00]
- **Three Fusion Architectures Compared** — paper_v4_extracted_text_feature_level_fusion, paper_v4_extracted_text_soft_voting_ensemble, paper_v4_extracted_text_cross_attention_fusion, paper_v4_extracted_text_multicardiofusion [EXTRACTED 1.00]
- **ECG Feature Extraction Pipeline** — paper_v4_extracted_text_ptb_xl_dataset, paper_v4_extracted_text_lead_ii, paper_v4_extracted_text_tsfresh, paper_v4_extracted_text_variance_threshold, paper_v4_extracted_text_standardscaler, paper_v4_extracted_text_dataset_d2 [EXTRACTED 1.00]
- **Controlled leak-free evaluation protocol** — paper_multicardiofusion_revised_v5_leakage_aware_pipeline, paper_multicardiofusion_revised_v5_repeated_stratified_cv, paper_multicardiofusion_revised_v5_nadeau_bengio_corrected_t, paper_multicardiofusion_revised_v5_holm_correction, paper_multicardiofusion_revised_v5_permutation_test [EXTRACTED 1.00]
- **Extended ablation study (94 configurations)** — paper_multicardiofusion_revised_v5_component_ablation, paper_multicardiofusion_revised_v5_control_experiments, paper_multicardiofusion_revised_v5_robustness_scenarios, paper_multicardiofusion_revised_v5_external_validation, paper_multicardiofusion_revised_v5_holdout_sensitivity [EXTRACTED 1.00]
- **Soft-voting ensemble members** — paper_multicardiofusion_revised_v5_xgboost, paper_multicardiofusion_revised_v5_rbf_svm, paper_multicardiofusion_revised_v5_random_forest, paper_multicardiofusion_revised_v5_gradient_boosting [EXTRACTED 1.00]

## Communities (28 total, 7 thin omitted)

### Community 0 - "v5 Paper Findings & Citations"
Cohesion: 0.05
Nodes (48): Dominant predictors cp, ca, thal, Chawla et al. 2002 - SMOTE [17], Chen & Guestrin 2016 - XGBoost [28], Christ et al. 2018 - tsfresh [10], Component ablation study (Table VII), Control experiments: noise, shuffled proxy, non-ECG features (Table VIII), D3 36-column matrix (13 clinical + 23 ECG-proxy), Detrano et al. 1989 - probability algorithm for CAD [1] (+40 more)

### Community 1 - "Ablation Engine & Statistics"
Cohesion: 0.07
Nodes (16): _cache_key(), calibration_slope(), corrected_t(), ecg_proxy_frame(), evaluate(), _fit_predict(), fmt_table(), fold_metrics() (+8 more)

### Community 2 - "Pipeline & Model Factory"
Cohesion: 0.06
Nodes (7): HardVoting, make_base(), make_clf(), make_pipe(), ProxyFeatures, _shap_fold(), ecg_pipe()

### Community 3 - "Literature Baselines"
Cohesion: 0.11
Nodes (21): Alizadehsani et al. [18], Arabasadi et al. 2017 [2], Attia et al. 2019 [12], Bi et al. 2021 [24], Detrano et al. 1989 [1], Gao et al. 2021 [8], Hannun et al. 2019 [9], Haq et al. 2018 [3] (+13 more)

### Community 4 - "Ablation Runner & Notebook Build"
Cohesion: 0.16
Nodes (5): build_standalone(), cell_sources(), execute(), merge(), strip_stderr()

### Community 6 - "Robustness Perturbations"
Cohesion: 0.21
Nodes (8): flip_labels(), tf(), gauss_noise(), mcar(), mcar_train(), subsample(), unavailable(), unavailable_train()

### Community 7 - "v4 Defects & Corrections"
Cohesion: 0.15
Nodes (14): Attia et al. 2019 (ref [12]) attribution correction, Response to reviewer action plan (v4 to v5), Table I rigor columns (partly done), Split into three algorithms; SHAP post-hoc, Ablation cell c04: Leakage, nested tuning, 200 hold-out splits, downslope proxy feature coding bug (slope==2), Five ECG-proxy columns duplicate clinical columns, ECG-proxy feature block (23 columns) (+6 more)

### Community 8 - "Paper Table Generator"
Cohesion: 0.19
Nodes (5): build.sh script, _bin(), _cont(), macro(), pfmt()

### Community 9 - "Worklog Ablation Cells"
Cohesion: 0.18
Nodes (10): Ablation cell c02: Controls and permutation label test, Ablation cell c05: Robustness scenarios and subgroups, Ablation cell c07: External validation (Hungary, Switzerland, VA), Ablation cell c11: Claim-by-claim verdict table, Extended ablation suite (experiments/ablation c00-c11), External-site sensitivity collapse from healthy-median imputation of ca/thal, Holm-adjusted p-values, MultiCardioFusion worklog (+2 more)

### Community 10 - "Project Overview & Datasets"
Cohesion: 0.18
Nodes (11): Rigorous ablation table with 95% CI and delta AUC, Single-lead ECG justification, Extended statistically controlled ablation study, MultiCardioFusion (project README), PTB-XL ECG dataset v1.0.3 (PhysioNet), experiments/run_ablation.py (reproduce ablation), UCI Cleveland Heart Disease dataset (id 45), tsfresh (+3 more)

### Community 11 - "v4 ECG Branch & Future Work"
Cohesion: 0.18
Nodes (10): Christ et al. 2018 tsfresh [10], Federated Learning (future work), Self-supervised ECG Transformer Encoders (ST-MEM, MERL), Hossain et al. 2023 Lead-II [37], Khan 2021 MDCNN [5], Lead-II ECG Channel, PTB-XL ECG Dataset, tsfresh Feature Extraction (+2 more)

### Community 12 - "Imbalance & Dependencies"
Cohesion: 0.22
Nodes (10): SMOTE on mixed-type small clinical data synthesizes unrealistic patients, imbalanced-learn, Python requirements, scikit-learn, tensorflow-cpu, ucimlrepo, wfdb, xgboost (+2 more)

### Community 13 - "v4 Fusion Architectures"
Cohesion: 0.20
Nodes (7): Adam Optimizer, Ioffe & Szegedy 2015 BatchNorm [32], Kingma & Ba 2015 Adam [31], Multimodal Fusion Hold-out Results (Table IV), Srivastava et al. 2014 Dropout [33], StandardScaler Standardisation, Vaswani et al. 2017 Attention Is All You Need [14]

### Community 14 - "Reviewer Reframing Requests"
Cohesion: 0.25
Nodes (9): C. Rodriguez Rivero (reviewer), PTB-XL label semantics vs coronary narrowing, MultiCardioFusion Reviewer Comment Action Plan, Ribeiro et al. (12-lead ECG) and Wagner et al. (PTB-XL), Suggested title: Explainable ECG-Proxy-Enhanced Clinical Prediction Framework, True multimodal fusion vs proxy-feature fusion vs ECG-only benchmarking, tsfresh on 10-second ECG segments justification, Three-category taxonomy: true fusion / proxy fusion / ECG-only benchmarking (+1 more)

### Community 15 - "v4 Datasets & SMOTE"
Cohesion: 0.28
Nodes (7): Ablation Study, Chawla et al. 2002 SMOTE [17], D1 Clinical-Only Dataset (303x13), D2 PTB-XL tsfresh Dataset (303x568), D3 Aligned Multimodal Matrix (303x36), SMOTE Oversampling, Unimodal Baseline Results (Table III)

### Community 16 - "Soft-Voting Ensemble Members"
Cohesion: 0.25
Nodes (7): Breiman 2001 Random Forests [29], Cortes & Vapnik 1995 SVN [30], Gradient Boosting Machine, Pedregosa et al. 2011 scikit-learn [34], Random Forest, RandomizedSearchCV Hyperparameter Tuning, Support Vector Machine (RBF)

### Community 17 - "SHAP & XGBoost (v4)"
Cohesion: 0.29
Nodes (7): Chen & Guestrin 2016 XGBoost [28], Lundberg et al. 2018 TreeSHAP [16], Lundberg & Lee 2017 SHAP [15], SHAP Interaction Heatmap (Top 12 Features), Statistical Significance Testing (paired t-test, McNemar), TreeSHAP / SHAP Explainability, XGBoost

### Community 18 - "SHAP Caveats & XAI Refs"
Cohesion: 0.40
Nodes (6): Patlatzoglou et al. (npj Digit. Med. 2025), SHAP caveats: post-hoc, associational, unstable on small N, Taleban et al. (BSPC 2026), Four reviewer-supplied XAI references (author action), shap, Ablation cell c08: SHAP stability and permutation importance

### Community 20 - "Verdict Code"
Cohesion: 0.83
Nodes (3): add(), paired(), verdict()

### Community 21 - "Calibration Findings"
Cohesion: 0.50
Nodes (4): Probability calibration and tuning-budget parity, Ablation cell c06: Ordinal vs one-hot encoding and calibration, Soft voting under-confident; Platt scaling helps, Soft-voting ensemble

### Community 22 - "Cross-Attention L=1 Issue"
Cohesion: 0.50
Nodes (3): Cross-attention over sequence length 1 questionable, Ablation cell c09: Dual-branch DL attention ablation, MCF-BiModNet cross-attention dual-branch network

## Knowledge Gaps
- **63 isolated node(s):** `scikit-learn`, `xgboost`, `tensorflow-cpu`, `wfdb`, `ucimlrepo` (+58 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 140 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MultiCardioFusion Framework` connect `Literature Baselines` to `v4 ECG Branch & Future Work`, `v4 Fusion Architectures`, `v4 Datasets & SMOTE`, `Soft-Voting Ensemble Members`, `SHAP & XGBoost (v4)`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `Response to reviewer action plan (v4 to v5)` (e.g. with `Cross-attention with sequence length 1 is a linear map` and `downslope proxy feature coding bug (slope==2)`) actually correct?**
  _`Response to reviewer action plan (v4 to v5)` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `scikit-learn`, `xgboost`, `tensorflow-cpu` to the rest of the system?**
  _63 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `v5 Paper Findings & Citations` be split into smaller, more focused modules?**
  _Cohesion score 0.05387205387205387 - nodes in this community are weakly interconnected._
- **Why does `make_clf()` connect `Pipeline & Model Factory` to `Ablation Engine & Statistics`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._
- **Should `Ablation Engine & Statistics` be split into smaller, more focused modules?**
  _Cohesion score 0.07272727272727272 - nodes in this community are weakly interconnected._
- **Should `Pipeline & Model Factory` be split into smaller, more focused modules?**
  _Cohesion score 0.06006006006006006 - nodes in this community are weakly interconnected._