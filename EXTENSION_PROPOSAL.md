# Extending MultiCardioFusion into a publishable study

## 1. Why the current paper is not enough

The v5 paper is now correct, but it offers no new contribution:

- **The ECG-proxy idea gives no gain, and for a mathematical reason.** The proxy features are functions of columns
  the model already has, so they cannot add information.
- **The evaluation itself has been published.** A literature search found:
  - leakage-free nested CV with calibration and decision curves on these cohorts
    ([PubMed 42342721](https://pubmed.ncbi.nlm.nih.gov/42342721/));
  - leave-one-cohort-out validation of the four UCI hospitals
    ([J Med Eng Technol 2026](https://www.tandfonline.com/doi/full/10.1080/03091902.2026.2759972));
  - leave-one-hospital-out validation with calibration on GitHub
    ([MHKhan50/Heart-Disease](https://github.com/MHKhan50/Heart-Disease)).
- **The real problem is the data.** Cleveland and PTB-XL are different patients, so "clinical + ECG fusion" could
  never be tested on them.

## 2. New data found and verified

| Dataset | Access | Paired clinical + ECG for the same patient? | CAD label | Use |
|---|---|---|---|---|
| **Z-Alizadeh Sani** (UCI id 412), 303 patients, Iran | Open (CC BY 4.0), downloaded | **Yes**: history, risk factors, symptoms, 7 resting-ECG findings, labs, echo | Angiography | **Main new cohort** |
| UCI Cleveland / Hungary / Switzerland / VA, 920 patients, US/Hungary/Switzerland | Open, already in repo | **Yes**: risk factors, resting ECG, exercise ECG, thallium, fluoroscopy | Angiography | Four-hospital internal–external validation |
| PTB-XL / PTB-XL+ | Open | Demographics + ECG only | ECG statements, not CAD | Not suitable for the CAD question |
| MIMIC-IV-ECG | ECG subset open; clinical linkage needs credentialed access | Only with credentialed MIMIC-IV | — | Future work (requires your PhysioNet credentialing) |
| CODE-15% | Zenodo (blocked here) | Age/sex only | Mortality / ECG labels | Different question |

**Together: 1,223 patients from 5 cohorts in 4 countries, all with angiographic CAD labels and patient-level clinical
and ECG data.** This is the true paired data the original project lacked.

## 3. Proposed study

**Working title:** *Where does the ECG add diagnostic information for obstructive coronary artery disease? A
stage-wise incremental-value analysis with internal–external validation across five angiography cohorts.*

### Clinical rationale

The diagnosis of suspected CAD follows a pathway. Guideline pre-test probability (ESC 2019; CAD Consortium) uses
**age, sex and symptoms**. Then come risk factors, a **resting ECG** (recommended for all patients), functional
testing (classically the **exercise ECG**), and finally imaging or angiography. Each stage costs time, money and
sometimes radiation. The clinically useful question is therefore not "how accurate is a model with every variable",
but **"how much does each test add over what is already known at that point?"** Moons et al. (Clin Chem 2012) set
out the methodology for measuring a test's added value, and Steyerberg describes the same approach for prediction
models.

### Hypotheses and why they are plausible

| # | Hypothesis | Theoretical or clinical basis |
|---|---|---|
| H1 | Resting-ECG findings (Q waves, ST-T changes) add diagnostic value over symptoms and risk factors when they are recorded in detail | Pathological Q waves and ST-T changes reflect prior infarction or ischaemia, so they relate causally to CAD. A coarse 3-level code (UCI `restecg`) may lose this |
| H2 | Exercise-ECG variables add value over resting information, and the gain transfers between hospitals | Exercise-induced ST depression and angina are the physiological signature of flow-limiting stenosis, the mechanism behind stress testing |
| H3 | Tests that are missing at the deployment site (fluoroscopy, thallium) harm transportability | Imputing a missing high-weight predictor with the training-site median biases predictions (shown in v5); this follows from prediction-time missingness theory (Fletcher Mercaldo & Blume 2020) |
| H4 | In high-prevalence referral cohorts, no stage model changes the decision at the guideline 15% testing threshold | Net benefit (Vickers & Elkin 2006) can only exceed "test everyone" if the model assigns many patients a risk below 15%. With 46–94% prevalence this is unlikely |

### Methods (all established, with references)

- **Nested stage models.** Logistic regression as the primary model, the standard for clinical prediction and as
  good as the ensembles in v5. Gradient boosting is a sensitivity analysis.
- **Internal validation:** repeated 10-fold CV with Nadeau–Bengio corrected CIs.
- **External validation:** leave-one-hospital-out internal–external cross-validation (Royston et al. 2004; Debray
  et al. 2013), pooling site AUCs and ΔAUCs with random-effects meta-analysis and reporting I² (Riley et al., BMJ
  2016).
- **Clinical usefulness:**
  - calibration (slope, intercept);
  - decision-curve analysis at the ESC 15% testing threshold and across 5–50%.
- **Missing tests:** median imputation versus pattern submodels (Fletcher Mercaldo & Blume 2020).
- **Guideline benchmark (pending, see §6):** apply the published CAD Consortium basic and clinical models unchanged
  to all five cohorts, and test whether any data-driven stage model beats them.
- **Reporting:** TRIPOD+AI checklist (Collins et al. 2024).

## 4. Pilot results (real data, `experiments/extension/pilot_stagewise.py`)

### A. Z-Alizadeh Sani (n = 303, 71% CAD), logistic regression, 5×10 CV

| Stage | AUC [95% CI] | ΔAUC vs previous stage | p |
|---|---|---|---|
| S1 age, sex, chest-pain type, dyspnoea | 0.883 [0.832, 0.935] | — | — |
| S2 + risk factors | 0.906 [0.863, 0.949] | +0.022 [−0.008, +0.052] | 0.14 |
| **S3 + resting-ECG findings** | 0.925 [0.890, 0.961] | **+0.020 [−0.001, +0.040]** | 0.065 |
| S4 + labs and vitals | 0.923 | −0.002 | 0.82 |
| S5 + echocardiography | 0.931 | +0.008 | 0.42 |

Net benefit at the 15% threshold is **0.662 for every stage, the same as testing everyone** (H4).

### B. Four UCI hospitals, leave-one-hospital-out (n = 920), paired ΔAUC pooled by random effects

| Increment | Pooled ΔAUC [95% CI] | I² |
|---|---|---|
| + risk factors | +0.006 [−0.003, +0.014] | 0.00 |
| + resting ECG (3-level `restecg`) | −0.001 [−0.003, +0.001] | 0.41 |
| **+ exercise ECG** | **+0.040 [+0.023, +0.058]** | **0.00** |
| + fluoroscopy/thallium | −0.008 [−0.022, +0.006] (VA: −0.014, significant) | 0.67 |

Pattern submodels performed the same as imputation for stages 1–4. At stage 5 they failed where no other site had
the test (Cleveland AUC 0.50), which is itself evidence for H3.

### What the pilot already shows

- **H2 is supported strongly.** Exercise ECG adds +0.04 AUC at external hospitals, consistently (I² = 0).
- **H1 is borderline.** Detailed resting-ECG findings add +0.02 (p = 0.065); the coarse UCI code adds nothing. This
  contrast (detailed vs coarse resting ECG) is a finding in itself, and a larger cohort would be needed to settle it.
- **H3 is supported.** Tests unavailable at the new site do not transfer.
- **H4 is supported.** In referral populations the models do not change testing decisions at the guideline threshold.

## 5. What would be new, and what would not

**New, as far as the literature search found:**

1. A stage-wise incremental-value analysis of the diagnostic pathway on these five public angiography cohorts. No
   published Z-Alizadeh Sani study reports a with-versus-without-ECG comparison.
2. Internal–external validation with random-effects pooling of *incremental* value, not only overall AUC.
3. An explicit test of guideline-threshold decision utility in referral cohorts.
4. A demonstration that the coarse resting-ECG coding in the UCI data hides information that detailed findings
   carry.

**Not new, and should not be claimed:**

- the individual methods;
- leave-one-hospital-out validation of the UCI cohorts by itself;
- "first multimodal fusion".

**Honest limits:**

- Historical referral cohorts (1980s US/Europe; 2010s Iran).
- Different variable definitions across cohorts: Z-Alizadeh Sani and UCI cannot be pooled into one model, only
  compared by stage.
- n = 303 in Z-Alizadeh Sani limits the power of H1.
- No contemporary tests such as CT angiography or troponin.

**Suitable venues:** clinical-methods and informatics journals, e.g. *Diagnostic and Prognostic Research*, *BMC
Medical Informatics and Decision Making*, *JAMIA Open*, *PLOS ONE*, *Frontiers in Cardiovascular Medicine*. It is not
a fit for high-impact clinical cardiology journals, which would expect contemporary prospective data.

## 6. Decisions needed from you

1. **Guideline benchmark.** Reading the published CAD Consortium coefficients and the ESC 2019 table was blocked by
   this session's permissions. Either paste Table 2 of Genders et al. (BMJ 2012;344:e3485) and Table 5 of Knuuti et
   al. (Eur Heart J 2020;41:407), or allow the fetch. I will not reconstruct coefficients from memory.
2. **Go-ahead** to turn this pilot into the full study (gradient-boosting sensitivity analysis, calibration, full
   decision curves, TRIPOD+AI checklist) and rewrite the paper around it. The v5 ablation work becomes a supporting
   section ("why engineered ECG proxies do not help").
3. **Optional:** PhysioNet credentialing (CITI course) would unlock MIMIC-IV clinical data linked to raw ECGs. That
   is the only open route to true raw-signal + clinical fusion at scale.
