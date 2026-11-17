# Graph Report - heart-  (2026-11-17)

## Corpus Check
- Corpus is ~47,811 words - fits in a single context window. You may not need a graph.

## Summary
- 445 nodes · 623 edges · 34 communities (27 shown, 7 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 48 edges (avg confidence: 0.85)
- Token cost: 456,849 input · 0 output

## Community Hubs (Navigation)
- Ablation Engine & Statistics
- Runner, Notebook Build & Fixes
- Pipeline & Model Factory
- Referee Citation Check
- Reviewer Response Rows
- v5 SHAP & Clinical-AI Refs
- v4 Reference List (as printed)
- Paper Table Generator
- Project Overview & Datasets
- Dual-Branch Network Code
- Reviewer Action Plan Concerns
- v5 Core Findings & CAD Data
- Robustness Perturbations Code
- v5 Ensemble & Calibration Refs
- v5 Software & Reporting Refs
- Statistical Protocol
- v4 Ensemble Members
- v4 Fusion Architectures
- v4 Datasets & Literature
- Imbalance Strategies
- Dual-Branch & FT-Transformer
- Ablation Study & RQs
- ECG Feature Extraction
- Leakage & Hold-out Issues
- PTB-XL & Lead II
- v4 SMOTE & Baselines
- v4 ECG Branch & Future Work
- Verdict Code
- Misc: Logistic regression baseline

## God Nodes (most connected - your core abstractions)
1. `MultiCardioFusion v5 paper` - 51 edges
2. `MultiCardioFusion Framework` - 33 edges
3. `Citation check` - 17 edges
4. `Extended ablation suite (c00-c11)` - 16 edges
5. `Response to the reviewer action plan (v4 to v5)` - 14 edges
6. `MultiCardioFusion pipeline` - 13 edges
7. `MultiCardioFusion Reviewer Comment Action Plan` - 11 edges
8. `SMOTE Oversampling` - 11 edges
9. `MCF-BiModNet Algorithm` - 10 edges
10. `Referee report and fact-check (v5)` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Nadeau-Bengio corrected resampled t-test` --semantically_similar_to--> `Nadeau-Bengio correction`  [INFERRED] [semantically similar]
  WORKLOG.md → paper/REFEREE_REPORT.md
- `Protocol mismatch in +7.04 pp headline claim` --semantically_similar_to--> `+7.04 pp fusion gain artefact`  [INFERRED] [semantically similar]
  WORKLOG.md → paper/REFEREE_REPORT.md
- `Nadeau-Bengio corrected resampled t-test` --cites--> `[26] Nadeau and Bengio 2003 - Inference for the generalization error`  [EXTRACTED]
  WORKLOG.md → paper/MultiCardioFusion_Revised_v5.pdf
- `Nadeau-Bengio corrected resampled t-test` --cites--> `[27] Bouckaert and Frank 2004 - Evaluating the replicability of significance tests for comparing learning algorithms`  [EXTRACTED]
  WORKLOG.md → paper/MultiCardioFusion_Revised_v5.pdf
- `Hold-out leakage (last CV fold model)` --semantically_similar_to--> `Hold-out leakage`  [INFERRED] [semantically similar]
  WORKLOG.md → paper/REFEREE_REPORT.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **ECG Feature Extraction Pipeline** — paper_v4_extracted_text_ptb_xl_dataset, paper_v4_extracted_text_lead_ii, paper_v4_extracted_text_tsfresh, paper_v4_extracted_text_variance_threshold, paper_v4_extracted_text_standardscaler, paper_v4_extracted_text_dataset_d2 [EXTRACTED 1.00]
- **Soft-Voting Ensemble Members** — paper_v4_extracted_text_xgboost, paper_v4_extracted_text_svm, paper_v4_extracted_text_gradient_boosting, paper_v4_extracted_text_random_forest, paper_v4_extracted_text_soft_voting_ensemble [EXTRACTED 1.00]
- **Three Fusion Architectures Compared** — paper_v4_extracted_text_feature_level_fusion, paper_v4_extracted_text_soft_voting_ensemble, paper_v4_extracted_text_cross_attention_fusion, paper_v4_extracted_text_multicardiofusion [EXTRACTED 1.00]
- **External UCI validation sites** — paper_multicardiofusion_revised_v5_hungarian_institute_of_cardiology_budapest, paper_multicardiofusion_revised_v5_university_hospitals_zurich_and_basel, paper_multicardiofusion_revised_v5_va_medical_center_long_beach, paper_multicardiofusion_revised_v5_external_validation_on_three_uci_hospitals [EXTRACTED 1.00]
- **Class-imbalance strategies compared** — paper_v4_extracted_text_smote, paper_multicardiofusion_revised_v5_smote_nc, paper_multicardiofusion_revised_v5_borderline_smote, paper_multicardiofusion_revised_v5_adasyn, paper_multicardiofusion_revised_v5_class_imbalance_strategies [EXTRACTED 1.00]
- **Soft-voting ensemble members** — paper_multicardiofusion_revised_v5_xgboost_classifier, paper_multicardiofusion_revised_v5_rbf_svm_classifier, paper_multicardiofusion_revised_v5_gradient_boosting_classifier, paper_multicardiofusion_revised_v5_random_forest_classifier, paper_multicardiofusion_revised_v5_soft_voting_ensemble [EXTRACTED 1.00]
- **Rigorous ablation statistical protocol** — worklog_extended_ablation_suite, worklog_repeated_stratified_cv_protocol, worklog_nadeau_bengio_corrected_t_test, worklog_holm_correction [EXTRACTED 1.00]
- **v4 notebook errors corrected by fix script** — tools_fix_original_notebook, paper_referee_report_hold_out_leakage, paper_referee_report_fusion_gain_7_04pp, paper_referee_report_cv_on_smoted_data, paper_referee_report_pandas_fillna_inplace_noop [EXTRACTED 1.00]
- **Reproducible results pipeline** — experiments_run_ablation, paper_reproducibility_identical_tables_result, paper_make_tables, tools_build_notebook, paper_reproducibility_extended_ablation_notebook [INFERRED 0.85]

## Communities (34 total, 7 thin omitted)

### Community 0 - "Ablation Engine & Statistics"
Cohesion: 0.08
Nodes (15): _cache_key(), calibration_slope(), corrected_t(), ecg_proxy_frame(), evaluate(), _fit_predict(), fmt_table(), fold_metrics() (+7 more)

### Community 1 - "Runner, Notebook Build & Fixes"
Cohesion: 0.07
Nodes (21): Code review, CV on SMOTE-augmented data, +7.04 pp fusion gain artefact, Hold-out leakage, No evidence of data fabrication, pandas fillna inplace no-op, PTB-XL v1.0.3, MultiCardioFusion_IEEE_2_extended_ablation.ipynb (+13 more)

### Community 2 - "Pipeline & Model Factory"
Cohesion: 0.06
Nodes (7): HardVoting, make_base(), make_clf(), make_pipe(), ProxyFeatures, _shap_fold(), ecg_pipe()

### Community 3 - "Referee Citation Check"
Cohesion: 0.08
Nodes (30): Arabasadi et al. 2017, Attia et al. 2019, Items only the author can resolve, Bi et al. 2021 (not found), Citation check, Claims and their evidence, Detrano 1989, Gao et al. 2021 (+22 more)

### Community 4 - "Reviewer Response Rows"
Cohesion: 0.08
Nodes (27): Cross-attention model absent from code, Response to the reviewer action plan (v4 to v5), Four reviewer-supplied XAI references, Row 0: Title and claims reframed, Row 12: SMOTE on mixed data, strategy comparison, Row 13: Calibration, voting weights, tuning parity, Row 14: Cross-attention with sequence length 1, Row 16: Rigorous ablation with CIs (+19 more)

### Community 5 - "v5 SHAP & Clinical-AI Refs"
Cohesion: 0.13
Nodes (19): [10] Attia et al. 2019 - Screening for cardiac contractile dysfunction using an artificial intelligence-enabled electrocardiogram, [12] Lundberg and Lee 2017 - A unified approach to interpreting model predictions, [15] Ribeiro et al. 2020 - Automatic diagnosis of the 12-lead ECG using a deep neural network, [16] Lundberg et al. 2018 - Explainable machine-learning predictions for the prevention of hypoxaemia during surgery, [24] World Health Organization - Cardiovascular diseases (CVDs) key facts, [2] Arabasadi et al. 2017 - Computer aided decision making for heart disease detection using hybrid neural network-genetic algorithm, [32] Kumar et al. 2020 - Problems with Shapley-value-based explanations as feature importance measures, [33] Molnar et al. 2022 - General pitfalls of model-agnostic interpretation methods for machine learning models (+11 more)

### Community 6 - "v4 Reference List (as printed)"
Cohesion: 0.12
Nodes (19): Alizadehsani et al. [18], Attia et al. 2019 [12], Bi et al. 2021 [24], Hannun et al. 2019 [9], Latha & Jeeva 2019 [7], Litjens et al. 2017 [11], Lundberg et al. 2018 sepsis [23], Lundberg et al. 2018 TreeSHAP [16] (+11 more)

### Community 7 - "Paper Table Generator"
Cohesion: 0.13
Nodes (10): build.sh script, _bin(), _cont(), macro(), pfmt(), read(), Re-run of 94 configurations from empty cache, Environment (Python 3.13, scikit-learn 1.9.1, XGBoost 3.4.1, SHAP 0.54.1, TF 2.21, pandas 3.0.5) (+2 more)

### Community 8 - "Project Overview & Datasets"
Cohesion: 0.11
Nodes (18): Extended statistically controlled ablation study, MultiCardioFusion (project README), PTB-XL ECG dataset v1.0.3 (PhysioNet), experiments/run_ablation.py (reproduce ablation), UCI Cleveland Heart Disease dataset (id 45), imbalanced-learn, Python requirements, scikit-learn (+10 more)

### Community 10 - "Reviewer Action Plan Concerns"
Cohesion: 0.14
Nodes (15): C. Rodriguez Rivero (reviewer), Probability calibration and tuning-budget parity, Cross-attention over sequence length 1 questionable, Patlatzoglou et al. (npj Digit. Med. 2025), PTB-XL label semantics vs coronary narrowing, MultiCardioFusion Reviewer Comment Action Plan, Ribeiro et al. (12-lead ECG) and Wagner et al. (PTB-XL), Rigorous ablation table with 95% CI and delta AUC (+7 more)

### Community 11 - "v5 Core Findings & CAD Data"
Cohesion: 0.16
Nodes (15): [1] Detrano et al. 1989 - International application of a new probability algorithm for the diagnosis of coronary artery disease, [25] Janosi et al. 1989 - Heart Disease UCI Machine Learning Repository, [43] Riley et al. 2020 - Calculating the sample size required for developing a clinical prediction model, ca and thal dominant predictors, Coronary artery disease (CAD) label, D3 36-column matrix, ECG-proxy feature block, External validation on three UCI hospitals (+7 more)

### Community 12 - "Robustness Perturbations Code"
Cohesion: 0.21
Nodes (8): flip_labels(), tf(), gauss_noise(), mcar(), mcar_train(), subsample(), unavailable(), unavailable_train()

### Community 13 - "v5 Ensemble & Calibration Refs"
Cohesion: 0.14
Nodes (14): [13] Lundberg et al. 2018 - Consistent individualized feature attribution for tree ensembles, [17] Chen and Guestrin 2016 - XGBoost: A scalable tree boosting system, [18] Breiman 2001 - Random forests, [19] Cortes and Vapnik 1995 - Support-vector networks, [30] Van Calster et al. 2019 - Calibration: the Achilles heel of predictive analytics, Gradient boosting classifier, Hard voting, Probability calibration (+6 more)

### Community 14 - "v5 Software & Reporting Refs"
Cohesion: 0.18
Nodes (11): [23] Pedregosa et al. 2011 - Scikit-learn: Machine learning in Python, [31] Huang et al. 2020 - Fusion of medical imaging and electronic health records using deep learning: a systematic review, [41] Lemaitre, Nogueira and Aridas 2017 - Imbalanced-learn, [44] Collins et al. 2024 - TRIPOD+AI statement, imbalanced-learn, MIMIC-IV-ECG, MultiCardioFusion pipeline, Proxy-feature fusion (+3 more)

### Community 15 - "Statistical Protocol"
Cohesion: 0.18
Nodes (10): [26] Nadeau and Bengio 2003 - Inference for the generalization error, [27] Bouckaert and Frank 2004 - Evaluating the replicability of significance tests for comparing learning algorithms, [28] Holm 1979 - A simple sequentially rejective multiple test procedure, Holm multiple-comparison correction, Repeated stratified 10-fold cross-validation (5x10), Known limitations, Nadeau-Bengio correction, Review of new ablation code (imblearn Pipeline) (+2 more)

### Community 16 - "v4 Ensemble Members"
Cohesion: 0.18
Nodes (10): Breiman 2001 Random Forests [29], Chen & Guestrin 2016 XGBoost [28], Cortes & Vapnik 1995 SVN [30], Gradient Boosting Machine, Pedregosa et al. 2011 scikit-learn [34], Random Forest, RandomizedSearchCV Hyperparameter Tuning, Statistical Significance Testing (paired t-test, McNemar) (+2 more)

### Community 17 - "v4 Fusion Architectures"
Cohesion: 0.20
Nodes (7): Adam Optimizer, Ioffe & Szegedy 2015 BatchNorm [32], Kingma & Ba 2015 Adam [31], Multimodal Fusion Hold-out Results (Table IV), Srivastava et al. 2014 Dropout [33], StandardScaler Standardisation, Vaswani et al. 2017 Attention Is All You Need [14]

### Community 18 - "v4 Datasets & Literature"
Cohesion: 0.25
Nodes (8): Ablation Study, Arabasadi et al. 2017 [2], D3 Aligned Multimodal Matrix (303x36), Detrano et al. 1989 [1], Gao et al. 2021 [8], Haq et al. 2018 [3], Janosi et al. 1989 UCI Heart Disease [38], UCI Cleveland Heart Disease Dataset

### Community 19 - "Imbalance Strategies"
Cohesion: 0.29
Nodes (7): [14] Chawla et al. 2002 - SMOTE: Synthetic minority over-sampling technique, [35] He et al. 2008 - ADASYN: Adaptive synthetic sampling approach for imbalanced learning, [36] Han, Wang and Mao 2005 - Borderline-SMOTE, ADASYN, Borderline-SMOTE, Class-imbalance strategies, SMOTE-NC

### Community 20 - "Dual-Branch & FT-Transformer"
Cohesion: 0.29
Nodes (7): [20] Kingma and Ba 2015 - Adam: A method for stochastic optimization, [21] Ioffe and Szegedy 2015 - Batch normalization, [22] Srivastava et al. 2014 - Dropout: A simple way to prevent neural networks from overfitting, [37] Gorishniy et al. 2021 - Revisiting deep learning models for tabular data, Dual-branch neural network (MCF-BiModNet), FT-Transformer, Token-level cross-attention

### Community 21 - "Ablation Study & RQs"
Cohesion: 0.29
Nodes (7): [29] Ojala and Garriga 2010 - Permutation tests for studying classifier performance, Extended ablation study (94 configurations), Label-permutation test, Negative control experiments, Robustness and edge-case scenarios (16), RQ2 pipeline component contributions, RQ3 robustness

### Community 22 - "ECG Feature Extraction"
Cohesion: 0.29
Nodes (7): [9] Christ et al. 2018 - Time series feature extraction on basis of scalable hypothesis tests (tsfresh), Christ et al. 2018 tsfresh [10], Self-supervised ECG Transformer Encoders (ST-MEM, MERL), Hossain et al. 2023 Lead-II [37], Lead-II ECG Channel, tsfresh Feature Extraction, VarianceThreshold Filter

### Community 23 - "Leakage & Hold-out Issues"
Cohesion: 0.33
Nodes (6): [11] Vaswani et al. 2017 - Attention is all you need, [38] Vabalas et al. 2019 - Machine learning algorithm validation with a limited sample size, Cross-attention with sequence length 1, Data leakage (SMOTE before CV), MultiCardioFusion v4 manuscript, Seed-42 61-patient hold-out split

### Community 24 - "PTB-XL & Lead II"
Cohesion: 0.33
Nodes (6): [39] Kligfield et al. 2007 - Recommendations for the standardization and interpretation of the electrocardiogram Part I, [42] Wagner et al. 2022 - PTB-XL version 1.0.3, [5] Wagner et al. 2020 - PTB-XL, a large publicly available electrocardiography dataset, Lead II single-lead choice, Parallel ECG benchmarking, PTB-XL ECG dataset

### Community 25 - "v4 SMOTE & Baselines"
Cohesion: 0.40
Nodes (5): Chawla et al. 2002 SMOTE [17], D1 Clinical-Only Dataset (303x13), D2 PTB-XL tsfresh Dataset (303x568), SMOTE Oversampling, Unimodal Baseline Results (Table III)

### Community 26 - "v4 ECG Branch & Future Work"
Cohesion: 0.40
Nodes (4): Federated Learning (future work), Khan 2021 MDCNN [5], PTB-XL ECG Dataset, Wagner et al. 2020 PTB-XL [6]

### Community 28 - "Verdict Code"
Cohesion: 0.83
Nodes (3): add(), paired(), verdict()

## Knowledge Gaps
- **96 isolated node(s):** `tsfresh`, `Rigorous ablation table with 95% CI and delta AUC`, `Single-lead ECG justification`, `PTB-XL ECG dataset v1.0.3 (PhysioNet)`, `UCI Cleveland Heart Disease dataset (id 45)` (+91 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 173 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Extended ablation suite (c00-c11)` connect `Reviewer Response Rows` to `Project Overview & Datasets`, `Runner, Notebook Build & Fixes`, `Statistical Protocol`?**
  _High betweenness centrality (0.384) - this node is a cross-community bridge._
- **What connects `tsfresh`, `Rigorous ablation table with 95% CI and delta AUC`, `Single-lead ECG justification` to the rest of the system?**
  _96 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Ablation Engine & Statistics` be split into smaller, more focused modules?**
  _Cohesion score 0.07610993657505286 - nodes in this community are weakly interconnected._
- **Why does `MultiCardioFusion v5 paper` connect `v5 SHAP & Clinical-AI Refs` to `v5 Core Findings & CAD Data`, `v5 Ensemble & Calibration Refs`, `v5 Software & Reporting Refs`, `Statistical Protocol`, `Imbalance Strategies`, `Dual-Branch & FT-Transformer`, `Ablation Study & RQs`, `ECG Feature Extraction`, `Leakage & Hold-out Issues`, `PTB-XL & Lead II`?**
  _High betweenness centrality (0.352) - this node is a cross-community bridge._
- **Should `Runner, Notebook Build & Fixes` be split into smaller, more focused modules?**
  _Cohesion score 0.07396870554765292 - nodes in this community are weakly interconnected._
- **Should `Pipeline & Model Factory` be split into smaller, more focused modules?**
  _Cohesion score 0.06006006006006006 - nodes in this community are weakly interconnected._
- **Should `Referee Citation Check` be split into smaller, more focused modules?**
  _Cohesion score 0.08172043010752689 - nodes in this community are weakly interconnected._