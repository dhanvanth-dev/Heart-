# ══════════════════════════════════════════════════════════════════════════
# 17B-3. CLASS-IMBALANCE STRATEGIES (reviewer rows 12 & 15)
# SMOTE on mixed continuous/binary/ordinal data can create unrealistic
# patients. Compare: no correction, class weighting, random oversampling,
# SMOTE (after features, as in the paper), SMOTE on raw rows before feature
# derivation, SMOTE-NC (categorical-aware), Borderline-SMOTE and ADASYN.
# All resampling happens inside training folds only.
# ══════════════════════════════════════════════════════════════════════════
print("── 17B-3 Resampling strategies ──────────────────────────────────────")
ABL3 = {
    REF: ABL1[REF],
    'No correction':                         make_pipe(FEAT_FULL(), sampler=None),
    'Class weighting (no resampling)':       make_pipe(FEAT_FULL(), sampler=None, balanced=True),
    'Random oversampling':                   make_pipe(FEAT_FULL(), sampler='ros'),
    'SMOTE on raw rows → derive proxy':      make_pipe(FEAT_FULL(), sampler='smote', sampler_stage='pre'),
    'SMOTE-NC on raw rows → derive proxy':   make_pipe(FEAT_FULL(), sampler='smotenc', sampler_stage='pre'),
    'Borderline-SMOTE':                      make_pipe(FEAT_FULL(), sampler='borderline'),
    'ADASYN':                                make_pipe(FEAT_FULL(), sampler='adasyn'),
}
for name, pipe in ABL3.items():
    evaluate(name, pipe)
abl3 = summarize(list(ABL3), REF, title='E3_resampling')
display(fmt_table(abl3))

# ── How realistic are SMOTE patients? Share of synthetic rows with impossible
#    (non-integer) categorical codes, and derived-feature inconsistency ──────
tr, te = SPLITS[0]
imp = SimpleImputer(strategy='median').fit(X_cl_raw[tr])
Xtr_imp = imp.transform(X_cl_raw[tr])
pf = ProxyFeatures().fit(None)
Z = pf.transform(Xtr_imp); sc = StandardScaler().fit(Z)
Zs, ys = SMOTE(random_state=SEED).fit_resample(sc.transform(Z), y_cl[tr])
synth = sc.inverse_transform(Zs[len(tr):])
names = pf.feature_names_
cat_idx = [names.index(c) for c in CAT_COLS]
non_integer = np.mean(np.any(np.abs(synth[:, cat_idx] - np.round(synth[:, cat_idx])) > 1e-6, axis=1))
op, op_sq = synth[:, names.index('oldpeak')], synth[:, names.index('oldpeak_sq')]
inconsistent = np.mean(np.abs(op ** 2 - op_sq) > 0.05 * np.maximum(1, op_sq))
dup_gap = np.mean(np.abs(synth[:, names.index('oldpeak')] - synth[:, names.index('proxy:oldpeak')]) > 1e-6)
realism = pd.DataFrame({
    'Synthetic rows (fold 1)': [len(synth)],
    'Rows with non-integer categorical codes': [f'{non_integer:.1%}'],
    'Rows where oldpeak_sq ≠ oldpeak² (>5%)': [f'{inconsistent:.1%}'],
    'Rows where duplicate oldpeak copies disagree': [f'{dup_gap:.1%}'],
})
display(realism); realism.to_csv(f'{TAB_DIR}/E3_smote_realism.csv', index=False)

# ── Class counts per partition (reviewer row 15) ─────────────────────────
X_tr_h, X_te_h, y_tr_h, y_te_h = train_test_split(X_cl_raw, y_cl, test_size=0.2,
                                                  stratify=y_cl, random_state=SEED)
_, y_after = SMOTE(random_state=SEED).fit_resample(
    np.nan_to_num(SimpleImputer(strategy='median').fit_transform(X_tr_h)), y_tr_h)
counts = pd.DataFrame({
    'Dataset': ['D1/D3 Cleveland (80/20 hold-out)', 'D1/D3 Cleveland (one 10-fold CV fold)'],
    'Train (healthy/disease)': [f'{np.sum(y_tr_h == 0)}/{np.sum(y_tr_h == 1)}',
                                f'{np.sum(y_cl[tr] == 0)}/{np.sum(y_cl[tr] == 1)}'],
    'Train after SMOTE': [f'{np.sum(y_after == 0)}/{np.sum(y_after == 1)}',
                          f'{np.sum(ys == 0)}/{np.sum(ys == 1)}'],
    'Test (healthy/disease)': [f'{np.sum(y_te_h == 0)}/{np.sum(y_te_h == 1)}',
                               f'{np.sum(y_cl[te] == 0)}/{np.sum(y_cl[te] == 1)}'],
})
display(counts); counts.to_csv(f'{TAB_DIR}/E3_class_counts.csv', index=False)
