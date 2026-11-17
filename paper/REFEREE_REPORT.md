# Referee report and fact-check: MultiCardioFusion (v5)

This review was written from the point of view of a reviewer for a cardiology or medical-AI journal, where the
readers are clinicians and errors can affect patient care. It covers four questions:

1. Is every claim backed by evidence?
2. Do the citations say what the paper says they say?
3. Are the medical statements correct?
4. Is the code correct?

Everything found has been fixed in the repository. Items that only the author can resolve are listed at the end.

---

## 1. Citation check

Each citation was checked against the published paper, its publisher page or abstract, or an indexing record
(Wiley, IEEE Xplore, DOAJ, PhysioNet). The **v4 manuscript and notebook contained several citation errors that
would be grounds for rejection**:

| Ref. (v4) | What v4 said | What the source says | Action |
|---|---|---|---|
| Arabasadi *et al.* 2017 | "Cleveland (303), 93.1%" | **Z-Alizadeh Sani** dataset (303 patients, 54 features), **93.85%** | Corrected and labelled as a different dataset |
| Mohan *et al.* 2019 | 88.4% | **88.7%** (HRFLM) | Corrected |
| Latha & Jeeva 2019 | 89.1%, AUC 0.892 | **85.48%** (majority voting); no AUC found | Corrected; AUC removed |
| Haq *et al.* 2018 | "SVM, 91.7%, **AUC 0.950**, highest validated AUC" | **89%** with logistic regression on Relief features (10-fold CV); **no AUC 0.950 found** | Corrected; the "best prior AUC" claim removed everywhere, including notebook cell 51 |
| Khan 2021 | "PTB-XL, MDCNN, 90.5%, AUC 0.924" | Published **2020**, IEEE Access vol. 8, reports **98.2%**; it is **not** a PTB-XL study | Removed (its dataset could not be verified, and the reference is not needed) |
| Gao *et al.* 2021 | "AUC not reported" (added by me in v5) | The abstract lists ROC among its metrics | My statement corrected |
| Zhang *et al.* 2022, IEEE JBHI | "multi-view ECG + clinical framework" | **No such paper found** | Removed (likely fabricated) |
| Bi *et al.* 2021, Diagnostics | "SHAP agrees with clinicians" | **No such paper found**; the title is about SVM on cardiac MRI and does not concern SHAP | Removed (likely fabricated) |
| Hossain *et al.* 2023, Healthcare | Lead-II heartbeat classification | The paper with this title is by **Issa *et al.*, Heliyon 2023**, with different authors and journal | Replaced with the AHA/ACCF/HRS ECG standardisation statement (Kligfield *et al.*, Circulation 2007) |
| Attia *et al.* 2019 | v4: "AF screening"; v5 draft: "ECG + demographics" | **12-lead ECG only**, LV systolic dysfunction (EF ≤35%), AUC 0.93 | Corrected |
| Lundberg *et al.* 2018 | "npj Digit. Med., sepsis" | The SHAP clinical paper from 2018 is *Nat. Biomed. Eng.*, **hypoxaemia** | Corrected (author please confirm) |
| Ribeiro *et al.* 2020 | "specialist-level" | 2.3 M ECGs, F1 > 80%, outperformed cardiology **residents** | Wording corrected |
| WHO fact sheet | dated 2023 | The page is updated periodically and its figures change | Cited with an access date |
| Uncited entries | Alizadehsani, Lyu, Sharma, Pouriyeh, Benjamin, Tison, Rossi, James, Litjens | Listed but never cited | Removed (journals do not accept uncited references) |

Added for claims that previously had no support: Kligfield 2007 (Lead II as rhythm lead), Hanley & McNeil 1983
(comparing correlated AUCs), Riley 2020 (sample size), TRIPOD+AI 2024 (reporting), the PTB-XL v1.0.3 dataset DOI,
scikit-learn, imbalanced-learn and FT-Transformer (methods actually used).

After the changes, every `\cite` resolves and every bibliography entry is cited.

## 2. Claims and their evidence

Every number in the results, discussion, abstract and conclusion is generated from the experiment output files
by `paper/make_tables.py`, so no number is typed by hand. Hand-written statements were checked one by one. Errors
found in **my own v5 draft** and fixed:

| Statement | Problem | Fix |
|---|---|---|
| "Soft voting outperforms RF and GBM alone" | Not significant after Holm correction (p = 0.51, 0.56) | Reworded, p-values given |
| "Hard voting is clearly worse" | Holm p = 0.067; partly an artefact (a 4-member vote has only five possible scores) | Reworded |
| "Probabilistic ensembling matters" (discussion, conclusion) | Not supported by significant results | Removed |
| "Risk rises with the number of diseased vessels" | Dependence plot shows a **step** between 0 and ≥1 vessels | Corrected |
| "…and with a reversible thallium defect" | Risk is raised for **fixed and reversible** defects | Corrected |
| "ca and thal come from the same work-up as the label" | thal is a separate nuclear stress test | Reworded: both are advanced tests from the diagnostic work-up, so the model is a post-test classifier |
| "Calibrated, explainable risk estimates" | The probabilities are under-confident (slope 1.37) | Removed; a "not ready for clinical use" statement added |
| "Discrimination is lower in women" | False: AUC 0.899 in women vs 0.883 in men | Corrected (lower in age ≥55 and symptomatic chest pain) |
| "Isotonic and Platt both correct calibration" | Isotonic over-corrects (slope 0.56) | Corrected |
| "More data helps the ECG branch" | AUC 0.823 → 0.800 → 0.825 for 303/606/1212 records | Corrected |
| "Gain ≤0.01 AUC would need several thousand patients [Vabalas]" | The citation does not support a power calculation | Reworded with Hanley & McNeil |
| "This error can explain implausibly high accuracies in the literature" | Speculation about other studies | Softened |
| "a typical training fold" (SMOTE realism) | Measured on one fold only | "the first training fold" |
| "derived from exercise-ECG measurements" | restecg is a resting ECG | Corrected |
| "Two ordinal variables have missing values" | thal is categorical | Corrected |

## 3. Medical statements checked

| Statement | Status |
|---|---|
| Cleveland label: angiographic >50% diameter narrowing (num > 0) | Correct (UCI documentation, Detrano 1989) |
| Variable coding (cp 1–4 with 4 = asymptomatic; restecg 0/1/2 = normal/ST-T/LVH; slope 1/2/3 = up/flat/down; thal 3/6/7 = normal/fixed/reversible) | Correct (UCI documentation) |
| Lead II lies near the normal mean QRS axis and is the standard rhythm-strip lead | Correct (Kligfield 2007) |
| Anterior ischaemia in V1–V4, inferior in II/III/aVF | Correct |
| Cholesterol 0 mg/dL is impossible (missing-value code at Switzerland/VA) | Correct |
| Higher CAD rate with asymptomatic chest pain | True **in this referral cohort** (73% CAD); the paper now says it is not a general clinical rule |
| PTB-XL labels are ECG diagnostic statements, not coronary anatomy | Correct |
| PTB-XL v1.0.3: 21,799 records, 18,869 patients | Correct (PhysioNet) |

## 4. Code review

### 4a. Errors in the original (v4) notebook, all fixed in `tools/fix_original_notebook.py`, with the notebook re-executed end to end

| Cell | Error | Effect on the paper |
|---|---|---|
| 35 | The hold-out row "Multimodal – XGB" used `m_xgb`, the model from the **last CV fold**, trained on ~90% of all 303 patients including most hold-out patients | Leakage: on re-execution this row shows hold-out AUC **0.998** (accuracy 0.984). It fed the v4 notebook's state-of-the-art table |
| 35 | `cl_xgb_acc = cv_cl_acc`: the fused **soft-voting hold-out** accuracy minus the clinical **XGBoost 10-fold CV** accuracy | **This is the source of the "+7.04 pp fusion gain"**: different models and different protocols |
| 26/28/30 | CV run on **already-SMOTEd** training data, so synthetic neighbours of test patients were in training | Optimistic baselines (v4 Table III, part of Table IV) |
| 33 | Hyper-parameter search CV on SMOTEd data | Optimistic tuning scores |
| 20 | `downslope = (slope == 2)`: code 2 is *flat* | Mislabelled feature (no measurable effect) |
| 7, 20 | `df[col].fillna(..., inplace=True)` is a silent no-op in pandas ≥ 3 | Imputation silently skipped on current pandas |
| 31 | A clinical-only MLP labelled "DL dual-branch fusion" | The **cross-attention model described in the paper did not exist in the code** |
| 37 | The "No Feature Engineering" variant built the **same** matrix as the full system; a "97% target" line | Mislabelled ablation |
| 42 | "Trend toward significance" printed for p = 1.0; accuracy reported as "AUC improvement" | Misleading interpretation |
| 44, 51 | Literature values above; "+x pp over best literature"; "first work fusing Cleveland + PTB-XL"; other studies labelled "Leaky CV" without evidence | Unsupported claims in figures |
| 10 | Hard-coded Kaggle API key | Security issue (key removed; **rotate it**) |

### 4b. Was anything manipulated?

I found **no evidence that data or outputs were fabricated**. The numbers in the v4 paper match the outputs stored
in the v4 notebook. The problems are of a different kind, though a journal would still treat them as serious:

- **An invalid comparison** in the code produced the headline gain.
- **Mislabelled tables**: cross-validation results were presented as hold-out results, and two "★" rows were the same model.
- **A described model with no code behind it** (cross-attention).
- **Literature values that do not match the cited papers**, and **at least two references that could not be found**.

Whether these were mistakes or not, they must not reach a journal. They are corrected here and documented in the
paper (Section V-C) and in `RESPONSE_TO_REVIEWERS.md`.

### 4c. Review of the new ablation code (`experiments/ablation/`)

- **Leakage:** imputation, feature derivation, scaling and resampling are all inside the training folds
  (imblearn `Pipeline`).
- **Pairing:** splits are identical across variants.
- **Statistics:**
  - Nadeau–Bengio correction applied correctly (test/train = 1/9).
  - Holm procedure checked by hand.
  - The Spearman stability matrices have the correct orientation (folds × folds, seeds × seeds).
- **Bugs found and fixed during the work:**
  - The fold imputer dropped all-missing columns (now `keep_empty_features=True`).
  - A matplotlib API change.
  - A summary step referenced a name defined in another cell.
- **Reproducibility:** all 94 configurations were **re-run from an empty cache** into a separate folder, and the
  tables were compared with the reported ones (result in Section 5).
- **Known limitations (stated in the paper):**
  - Hyper-parameters fixed from the v4 search (nested tuning only for XGBoost).
  - Subgroup analyses are exploratory.
  - Holm correction is applied within tables, not across the whole study.
  - PTB-XL records with no diagnostic statement are labelled abnormal (v4 rule).
  - One PTB-XL patient appears twice in the 303-record sample.

## 5. Reproducibility check

All 94 configurations were re-run from scratch, outside the cache, into `results/ablation_rerun/` and compared
table by table with `results/ablation/`. Result: see `REPRODUCIBILITY.md` in this folder.

## 6. Would I recommend publication? Questions I would still ask as a referee

**For a clinical cardiology journal: not in its current form.** The corrected manuscript is honest, but its main
finding is negative (ECG-proxy features add nothing). The model relies on fluoroscopy and thallium results that are
not available at the point where a screening tool would be used, and it has no clinical-utility analysis.
**For a methods, ML-in-medicine or reproducibility venue** it could be suitable as a careful negative-result and
evaluation-practice paper, after the author-side items below are resolved.

Questions the author should be ready to answer:

1. **Intended use.** At what point in the clinical pathway would the model be used? With `ca` and `thal` as inputs
   it can only apply *after* fluoroscopy and nuclear imaging, when the clinician already has most of the information.
2. **Clinical utility.** Is there a decision-curve or net-benefit analysis, and which threshold would be used?
   Sensitivity at the default 0.5 threshold is only about 0.77.
3. **Calibration in the target population.** Prevalence ranges from 36% to 94% across the four sites. How would the
   model be recalibrated before use?
4. **Contemporary validity.** The cohort comes from the 1980s, before contemporary CAD diagnostics such as
   high-sensitivity troponin and CT coronary angiography were in routine use. Is it still representative?
5. **Missing data at external sites.** `ca` and `thal` are missing for up to 99% of patients, most likely not at
   random. Is median imputation acceptable, or should the model simply not use them?
6. **Sample size.** With 139 events and 36 candidate features, does the study meet minimum sample-size criteria
   (Riley 2020)?
7. **Reporting.** Please provide a completed TRIPOD+AI checklist.
8. **Multiplicity.** Across about 90 comparisons, how is the family-wise error controlled beyond Holm within tables?
9. **Fairness.** Subgroups are small (97 women, 25 with CAD). Was the study powered to detect differential
   performance?
10. **PTB-XL branch.** Why label records with no diagnostic statement as abnormal? How are repeat recordings from the
    same patient handled?
11. **Data and code availability.** Are the repository and the exact package versions archived (e.g. with a DOI)?
12. **Ethics.** Public de-identified data: a statement saying so is still expected.

## 7. Items only the author can resolve

- Add the four XAI references the reviewer supplied (Patlatzoglou 2025, Taleban 2026, an arXiv arrhythmia-XAI
  paper, Comput. Biol. Med. 2024), with full and verified details.
- Confirm the intended Lundberg 2018 reference.
- Explain the origin of the Zhang 2022 and Bi 2021 references, which could not be found.
- Add the Table I rigor columns (row 7), which need facts from each cited paper.
- Rotate the Kaggle API key.
- Provide the ethics, data-availability and conflict-of-interest statements the target journal requires.
