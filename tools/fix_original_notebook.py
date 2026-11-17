"""Apply code and text corrections to the original MultiCardioFusion notebook.

Reads  original/MultiCardioFusion_IEEE_1.ipynb   (as received; never modified)
Writes notebook/_fixed_base.ipynb                 (input to tools/build_notebook.py)

Every change is listed in FIXES with the reason. Each replacement must match exactly
once, so the script fails loudly if the source changes.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'original', 'MultiCardioFusion_IEEE_1.ipynb')
OUT = os.path.join(ROOT, 'notebook', '_fixed_base.ipynb')

TAG = '# [v5 fix] '

# (cell index, old text, new text, reason[, expected number of matches])
FIXES = [
    # ── Abstract made factual ────────────────────────────────────────────
    (0, "Comprehensive evaluation on balanced datasets (SMOTE augmentation) with 10-fold stratified cross-validation demonstrates that the multimodal approach achieves **>96% accuracy** and **AUC > 0.97**, outperforming the best unimodal clinical model by several percentage points. SHAP-based explainability reveals clinically meaningful feature interactions. Statistical significance (paired t-test, McNemar test) confirms superiority over unimodal baselines.",
     "**[v5 correction]** The original abstract claimed >96% accuracy, AUC > 0.97 and statistically confirmed superiority over unimodal baselines. None of these claims is supported by the results in this notebook. Under a leakage-free, repeated cross-validation protocol (Section 17B) the full system reaches AUC ≈ 0.90, the clinical variables alone perform equally well, the ECG-proxy features give no measurable gain, and the paired t-test and McNemar test are not significant.",
     'Abstract claims contradicted by the notebook\'s own results'),
    # ── pandas: chained in-place fillna is a no-op under Copy-on-Write ────
    (7, "        X_cl[col].fillna(X_cl[col].median(), inplace=True)",
        "        X_cl[col] = X_cl[col].fillna(X_cl[col].median())   " + TAG.strip() + " chained inplace fillna is a silent no-op in pandas>=3",
     'pandas Copy-on-Write: imputation silently did nothing'),
    (20, "    df_raw[col].fillna(df_raw[col].median(), inplace=True)",
         "    df_raw[col] = df_raw[col].fillna(df_raw[col].median())   " + TAG.strip() + " pandas>=3 safe",
     'pandas Copy-on-Write'),
    # ── UCI coding: slope 1 = up, 2 = flat, 3 = down ─────────────────────
    (20, "ecg_eng['downslope']         = (df_raw['slope'].values  == 2).astype(int)",
         "ecg_eng['downslope']         = (df_raw['slope'].values  == 3).astype(int)   " + TAG.strip() + " UCI slope 3 = downsloping (2 = flat)",
     'downslope flag tested the flat-slope code'),
    # ── CV on SMOTE-augmented data → synthetic rows in test folds ───────
    (26, "    for tr, te in skf.split(X, y):\n        model.fit(X[tr], y[tr])",
         "    for tr, te in skf.split(X, y):\n        " + TAG + "resample INSIDE the training fold; v4 ran CV on already-SMOTEd data,\n        # so synthetic neighbours of test patients were in the training folds (optimistic scores).\n        X_fit, y_fit = SMOTE(random_state=SEED).fit_resample(X[tr], y[tr])\n        model.fit(X_fit, y_fit)",
     'CV leakage through SMOTE'),
    (26, "cv_evaluate(model, Xcl_tr_sm, y_tr_sm, label=name)", "cv_evaluate(model, Xcl_tr, y_tr, label=name)", 'CV on original rows'),
    (28, "cv_evaluate(model, Xecg_tr_sm, y_ecg_sm, label=name)", "cv_evaluate(model, Xecg_tr, y_ecg_tr, label=name)", 'CV on original rows'),
    (30, "cv_evaluate(model, Xfu_tr_sm, y_fu_sm, label=name)", "cv_evaluate(model, Xfu_tr, y_tr, label=name)", 'CV on original rows', 2),
    (31, 'print("== DL Dual-Branch Fusion ==")',
         'print("== Clinical-only MLP (no fusion; the dual-branch cross-attention model is evaluated in Section 17B-9) ==")',
     'mislabelled model'),
    # ── Hyper-parameter search ran CV on SMOTE-augmented rows ────────────
    (33, "xgb_grid.fit(Xfu_tr_sm, y_fu_sm)", "xgb_grid.fit(Xfu_tr, y_tr)   " + TAG.strip() + " tune on original rows (no synthetic rows in validation folds)", 'tuning leakage'),
    (33, "svm_grid.fit(Xfu_tr_sm, y_fu_sm)", "svm_grid.fit(Xfu_tr, y_tr)", 'tuning leakage'),
    (33, "gb_grid.fit(Xfu_tr_sm, y_fu_sm)", "gb_grid.fit(Xfu_tr, y_tr)", 'tuning leakage'),
    (33, "rf_grid.fit(Xfu_tr_sm, y_fu_sm)", "rf_grid.fit(Xfu_tr, y_tr)", 'tuning leakage'),
    # ── The "+7.04 pp" came from hold-out(fused voting) − CV(clinical XGB) ─
    (35, "cl_xgb_acc = cv_cl_acc   # use CV acc for final summary\ncl_xgb_auc = float(df_comparison.loc['Clinical – XGB', 'AUC'])",
     TAG + "v4 set cl_xgb_acc to the clinical XGB *10-fold CV* accuracy and later subtracted it from the\n"
     "# fused soft-voting *hold-out* accuracy (different model AND protocol) -> the reported '+7.04 pp'.\n"
     "# The clinical baseline now uses the SAME model (tuned soft voting) on the SAME split / CV folds.\n"
     "from sklearn.base import clone\n"
     "CL_ENSEMBLE = clone(BEST_ENSEMBLE).fit(np.nan_to_num(Xcl_tr_sm), y_tr_sm)\n"
     "yp_cl_vote  = CL_ENSEMBLE.predict(np.nan_to_num(Xcl_te))\n"
     "ypp_cl_vote = CL_ENSEMBLE.predict_proba(np.nan_to_num(Xcl_te))[:, 1]\n"
     "cl_xgb_acc = accuracy_score(y_te, yp_cl_vote)   # name kept for downstream cells: clinical soft voting, hold-out\n"
     "cl_xgb_auc = roc_auc_score(y_te, ypp_cl_vote)\n"
     "cv_cl_vote = []\n"
     "for tr, te in skf10_full.split(X_cl_clean, y_fused):\n"
     "    Xc_sm, yc_sm = smote_cv.fit_resample(np.nan_to_num(X_cl_clean[tr]), y_fused[tr])\n"
     "    cv_cl_vote.append(accuracy_score(y_fused[te], clone(BEST_ENSEMBLE).fit(Xc_sm, yc_sm).predict(np.nan_to_num(X_cl_clean[te]))))\n"
     "cv_cl_vote = np.array(cv_cl_vote)\n"
     "print(f\"\\n── Same-model, same-protocol comparison (soft voting) ──\")\n"
     "print(f\"  Hold-out accuracy : fused {acc_best:.4f} vs clinical {cl_xgb_acc:.4f}  (diff {(acc_best-cl_xgb_acc)*100:+.2f} pp)\")\n"
     "print(f\"  Hold-out AUC      : fused {auc_best:.4f} vs clinical {cl_xgb_auc:.4f}\")\n"
     "print(f\"  10-fold CV acc.   : fused {cv_acc_ensemble.mean():.4f} vs clinical {cv_cl_vote.mean():.4f}  (diff {(cv_acc_ensemble.mean()-cv_cl_vote.mean())*100:+.2f} pp)\")",
     'protocol-mismatched fusion gain'),
    (35, 'print(f"   Fusion gain (CV)           : +{(cv_acc_ensemble.mean() - cv_cl_acc)*100:.2f} pp")',
         'print(f"   (XGB vs soft voting are different models; see the same-model comparison above)")',
     'compared different models'),
    # ── Legacy single-split ablation: mislabelled variants, arbitrary target ─
    (37, "r['Variant'] = 'No Feature Engineering'; abl.append(r)",
         "r['Variant'] = 'Same features as full system (scaling check)'; abl.append(r)   " + TAG.strip() + " this matrix equals the full fused matrix",
     'mislabelled ablation variant'),
    (37, "abl.append({'Variant':  'No Calibration / No Stacking',", "abl.append({'Variant':  'Tuned single XGBoost (no ensemble)',", 'mislabelled variant'),
    (37, "ax.axhline(0.97, color='navy', ls='--', lw=1.5, label='97% target')\nax.legend()\n", "", 'arbitrary 97% target line'),
    (37, "ax.set_title('Figure 9: Ablation Study – Component Contributions', fontweight='bold')",
         "ax.set_title('Figure 9: Legacy single-split ablation (61 patients, no CIs) - see Section 17B', fontweight='bold')", 'context'),
    # ── Significance tests: no 'trend' for p = 1 ────────────────────────
    (42, "'◎ Trend toward significance (small N favours n.s.)'", "'not significant at α=0.05'", 'misleading wording'),
    (42, "'◎ Trend (note: McNemar requires c>>b for significance)'", "'not significant at α=0.05'", 'misleading wording'),
    (42, """  The multimodal system improves accuracy by
  +{(fu_scores.mean()-cl_scores.mean())*100:.2f} pp over clinical-only XGB across all 10 folds.
  With only 303 samples, paired tests are underpowered
  (Cohen's d ≈ {(fu_scores.mean()-cl_scores.mean())/cl_scores.std():.2f}).
  The AUC improvement (clinical {cl_scores.mean():.3f} → fused {fu_scores.mean():.3f}) and
  consistent fold-by-fold gains provide practical significance.""",
     """  Mean fold accuracy difference (fused − clinical XGB): {(fu_scores.mean()-cl_scores.mean())*100:+.2f} pp.
  Neither test shows a significant difference. These are accuracies, not AUCs.
  Section 17B repeats this comparison with 50 paired folds and corrected confidence intervals.""",
     'claimed practical significance for a zero difference; mislabelled accuracy as AUC'),
    # ── Summary cells ───────────────────────────────────────────────────
    (48, "    'Multimodal DL': [", "    'Clinical MLP (no fusion)': [", 'mislabelled model'),
    (48, "'Fusion Gain (Acc pp)'", "'Acc. difference vs clinical soft voting, same split (pp)'", 'clarify'),
    (50, "print(f'  Clinical XGB (unimodal)     :  Acc={cl_xgb_acc:.4f}  AUC={cl_xgb_auc:.4f}')",
         "print(f'  Clinical soft voting (same split) :  Acc={cl_xgb_acc:.4f}  AUC={cl_xgb_auc:.4f}')", 'clarify baseline'),
    (50, "print(f'  Multimodal DL  (fused)      :  Acc={accuracy_score(y_te,dl_pred):.4f}'",
         "print(f'  Clinical MLP (no fusion)    :  Acc={accuracy_score(y_te,dl_pred):.4f}'", 'mislabelled model'),
    (50, "print(f'  Fusion gain (Acc)           :  +{(acc_best-cl_xgb_acc)*100:.2f} pp')",
         "print(f'  Fused − clinical (same split):  {(acc_best-cl_xgb_acc)*100:+.2f} pp accuracy (single 61-patient split; see 17B)')", 'sign-aware, same protocol'),
    (50, "print('\\n✓ Notebook complete — ready for IEEE EMBC / BIBM submission.')",
         "print('\\n✓ Notebook complete. Use the Section 17B results (repeated CV with CIs) for any reported claim.')", 'remove unsupported readiness claim'),
]

CELL10 = '''# ══════════════════════════════════════════════════════════════════════════
# PTB-XL from PhysioNet (open access, no API key).                    [v5 fix]
# v4 downloaded the full dataset from Kaggle with a hard-coded API key, which
# must never be committed. Only the metadata and the sampled records are needed.
# ══════════════════════════════════════════════════════════════════════════
import os, urllib.request
PTB_URL = 'https://physionet.org/files/ptb-xl/1.0.3/'
PTB_BASE_DIR = next((d for d in ['../data/ptbxl', 'data/ptbxl', '/content/ptb-xl']
                     if os.path.exists(os.path.join(d, 'ptbxl_database.csv'))), '/content/ptb-xl')
os.makedirs(PTB_BASE_DIR, exist_ok=True)
for f in ('ptbxl_database.csv', 'scp_statements.csv'):
    if not os.path.exists(os.path.join(PTB_BASE_DIR, f)):
        urllib.request.urlretrieve(PTB_URL + f, os.path.join(PTB_BASE_DIR, f))
PTB_CSV     = os.path.join(PTB_BASE_DIR, 'ptbxl_database.csv')
SCP_CSV     = os.path.join(PTB_BASE_DIR, 'scp_statements.csv')
RECORDS_DIR = os.path.join(PTB_BASE_DIR, 'records100')   # 100 Hz version

def ensure_record(filename_lr):
    """Download one 100 Hz WFDB record (.hea + .dat) if it is not on disk yet."""
    base = os.path.join(PTB_BASE_DIR, filename_lr)
    for ext in ('hea', 'dat'):
        if not os.path.exists(f'{base}.{ext}'):
            os.makedirs(os.path.dirname(base), exist_ok=True)
            urllib.request.urlretrieve(f'{PTB_URL}{filename_lr}.{ext}', f'{base}.{ext}')

print(f"PTB-XL metadata: {PTB_CSV}  (exists={os.path.exists(PTB_CSV)})")
'''

CELL13_OLD = "        signal, fields = wfdb.rdsamp(full_path)"
CELL13_NEW = "        ensure_record(row['filename_lr'])          # [v5 fix] download on demand\n        signal, fields = wfdb.rdsamp(full_path)"

CELL44 = '''# ══════════════════════════════════════════════════════════════════════════
# 20. CONTEXT FROM THE LITERATURE                                      [v5 fix]
# v4 listed several values that do not match the cited papers (e.g. Arabasadi
# 93.1% on "Cleveland" - the paper used the Z-Alizadeh Sani dataset and reports
# 93.85%; Haq "AUC 0.950"; Khan "PTB-XL 90.5%"), drew a "97% benchmark" line and
# ranked results obtained under different protocols. The values below are
# checked against the original papers and are shown as context, not as a ranking.
# ══════════════════════════════════════════════════════════════════════════
lit = pd.DataFrame([
    ('Detrano et al. 1989 [1]',     'Cleveland (+3 sites)',     'Logistic probability algorithm', 77.0),
    ('Haq et al. 2018 [3]',         'Cleveland',                'LR + Relief, 10-fold CV',        89.0),
    ('Mohan et al. 2019 [4]',       'Cleveland',                'HRFLM',                          88.7),
    ('Latha & Jeeva 2019 [7]',      'Cleveland',                'Majority voting',                85.48),
    ('Gao et al. 2021 [8]',         'Cleveland',                'Bagging + PCA',                  98.6),
    ('Arabasadi et al. 2017 [2]',   'Z-Alizadeh Sani (other)',  'GA + neural network',            93.85),
], columns=['Study', 'Dataset', 'Method', 'Reported accuracy (%)'])
ours = pd.DataFrame([
    ('This notebook: fused soft voting',    'Cleveland', '10-fold CV (Section 16)',          round(cv_acc_ensemble.mean()*100, 1)),
    ('This notebook: clinical soft voting', 'Cleveland', '10-fold CV (Section 16)',          round(cv_cl_vote.mean()*100, 1)),
    ('This notebook: fused soft voting',    'Cleveland', 'single 61-patient hold-out',       round(acc_best*100, 1)),
], columns=['Study', 'Dataset', 'Method', 'Reported accuracy (%)'])
sota = pd.concat([lit, ours], ignore_index=True)
display(sota)
sota.to_csv('outputs/sota_comparison.csv', index=False)
print("Protocols, datasets and leakage controls differ between studies; these numbers are not directly comparable.")
print("For this work, use the repeated-CV estimates with confidence intervals in Section 17B.")
'''

CELL51 = '''# ══════════════════════════════════════════════════════════════════════════
# 25. FINAL SUMMARY                                                    [v5 fix]
# The v4 version of this cell drew a "conclusion dashboard" with claims that are
# not supported (e.g. "AUC surpasses best published (Haq 0.950)", "first work
# fusing Cleveland + PTB-XL", accuracy comparisons across different datasets and
# protocols). It is replaced by a factual summary of the controlled results.
# ══════════════════════════════════════════════════════════════════════════
import os
verdict_path = next((p for p in ['outputs/ablation/tables/E11_verdict.csv', '../results/ablation/tables/E11_verdict.csv',
                                 os.environ.get('MCF_ABL_DIR', '') + '/tables/E11_verdict.csv'] if p and os.path.exists(p)), None)
print('MultiCardioFusion — summary of evidence')
print('=' * 70)
print(f'Single 61-patient split (seed 42), soft voting: fused acc={acc_best:.3f}, AUC={auc_best:.3f}; '
      f'clinical acc={cl_xgb_acc:.3f}, AUC={cl_xgb_auc:.3f}')
print(f'10-fold CV accuracy, soft voting: fused {cv_acc_ensemble.mean():.3f} vs clinical {cv_cl_vote.mean():.3f}')
if verdict_path:
    print('\\nClaim-by-claim verdict under the controlled protocol (Section 17B-11):')
    display(pd.read_csv(verdict_path, index_col=0))
print('\\nConclusion: the ECG-proxy features do not improve on the clinical variables; '
      'the model is a post-test classifier (it relies on ca and thal) and is not validated for clinical use.')
'''

MD_REPLACE = {
    45: """## 21. DISCUSSION  *(revised in v5)*

### What the results show
Under a controlled, leakage-free protocol (Section 17B) the ECG-proxy features do **not** improve on the 13 clinical
variables: the clinical-only system performs as well as the fused system, and the proxy block performs no better than
added noise. This is expected, because every proxy feature is a deterministic function of variables the models
already receive. The v4 text in this section, which argued that fusion outperforms unimodal models, is withdrawn.

### Clinical relevance
The dominant predictors are chest-pain type (`cp`), vessels coloured by fluoroscopy (`ca`) and the thallium scan
(`thal`). `ca` and `thal` are advanced test results from the same diagnostic work-up as the angiographic label, so
the model is a *post-test* classifier, not a screening tool. External validation on three other hospitals gives lower
AUC (0.71–0.89). The model is not validated for clinical use.

### Explainability
SHAP values are post-hoc and associational. The top of the ranking is stable across folds, but credit assigned to
engineered features is redistributed from the variables they are computed from (Section 17B-8).
""",
    38: None,  # handled below (single sentence)
}


def main():
    nb = json.load(open(SRC))
    src = lambda i: ''.join(nb['cells'][i]['source'])
    setsrc = lambda i, s: nb['cells'][i].__setitem__('source', s.splitlines(True))
    for fix in FIXES:
        idx, old, new, why = fix[:4]
        expected = fix[4] if len(fix) > 4 else 1
        s = src(idx)
        n = s.count(old)
        assert n == expected, f'cell {idx}: expected {expected} match(es), found {n} for: {old[:70]!r} ({why})'
        setsrc(idx, s.replace(old, new))
    setsrc(10, CELL10)
    s = src(13); assert s.count(CELL13_OLD) == 1; setsrc(13, s.replace(CELL13_OLD, CELL13_NEW))
    setsrc(44, CELL44)
    setsrc(51, CELL51)
    setsrc(45, MD_REPLACE[45])
    s = src(38)
    old = "is consistent with established cardiology knowledge, validating clinical trustworthiness."
    if old in s:
        setsrc(38, s.replace(old, "is consistent with established cardiology knowledge. Consistency is necessary but not "
                                  "sufficient evidence of validity; SHAP is post-hoc and associational (see Section 17B-8)."))
    s = src(9)
    setsrc(9, "> **[v5]** The data are now downloaded from PhysioNet (PTB-XL v1.0.3, open access); no Kaggle key is needed.\n\n" + s)
    for c in nb['cells']:                       # outputs are regenerated by executing the notebook
        if c['cell_type'] == 'code':
            c['outputs'] = []; c['execution_count'] = None
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(nb, open(OUT, 'w'), indent=1, ensure_ascii=False)
    print(f'applied {len(FIXES) + 6} fixes -> {OUT}')


if __name__ == '__main__':
    main()
