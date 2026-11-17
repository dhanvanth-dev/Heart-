# ══════════════════════════════════════════════════════════════════════════
# 17B-7. EXTERNAL VALIDATION ON THE OTHER UCI SITES (distribution shift)
# Train on all 303 Cleveland patients; test unchanged on Hungary (Budapest),
# Switzerland (Zurich/Basel) and VA Long Beach. These sites have different
# disease prevalence and many missing tests (ca, thal, slope), so they are a
# realistic stress test. Missing values are imputed with Cleveland medians.
# ══════════════════════════════════════════════════════════════════════════
print("── 17B-7 External validation ────────────────────────────────────────")
SITES = {s: load_uci_site(s) for s in ('hungarian', 'switzerland', 'va')}
SITE_LABEL = {'hungarian': 'Hungary', 'switzerland': 'Switzerland', 'va': 'VA Long Beach'}

miss = pd.DataFrame({SITE_LABEL.get(s, s): {
    'n': len(y), 'disease prevalence': f'{y.mean():.1%}',
    **{f'missing {c}': f'{np.isnan(X[c]).mean():.0%}' for c in ('ca', 'thal', 'slope', 'chol', 'fbs', 'oldpeak')}}
    for s, (X, y) in [('Cleveland', (Xdf_cl, y_cl))] + list(SITES.items())}).T
display(miss); miss.to_csv(f'{TAB_DIR}/E7_site_characteristics.csv')

REDUCED = ('ca', 'thal', 'slope')         # mostly unavailable outside Cleveland
def drop_cols(cols):
    return lambda X: np.where(np.isin(np.arange(X.shape[1]), [CLIN_COLS.index(c) for c in cols]), np.nan, X)

EXT_MODELS = {
    'Fused (full system)': (ABL1[REF], None),
    'Clinical only': (ABL1['− ECG-proxy block (clinical only)'], None),
    'Fused, without ca/thal/slope': (ABL1[REF], drop_cols(REDUCED)),
    'Clinical, without ca/thal/slope': (ABL1['− ECG-proxy block (clinical only)'], drop_cols(REDUCED)),
}

def boot_ci(y, p, B=2000, seed=SEED):
    rng = np.random.RandomState(seed); vals = []
    for _ in range(B):
        b = rng.randint(0, len(y), len(y))
        if len(np.unique(y[b])) == 2: vals.append(roc_auc_score(y[b], p[b]))
    return np.percentile(vals, [2.5, 97.5])

def boot_delta(y, p1, p2, B=2000, seed=SEED):
    rng = np.random.RandomState(seed); vals = []
    for _ in range(B):
        b = rng.randint(0, len(y), len(y))
        if len(np.unique(y[b])) == 2: vals.append(roc_auc_score(y[b], p1[b]) - roc_auc_score(y[b], p2[b]))
    vals = np.array(vals); lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi, 2 * min(np.mean(vals <= 0), np.mean(vals >= 0))

ext_rows, ext_probs = [], {}
for mname, (pipe, tf) in EXT_MODELS.items():
    Xtr = X_cl_raw if tf is None else tf(X_cl_raw)
    model = clone(pipe).fit(Xtr, y_cl)
    for site, (Xs, ys) in SITES.items():
        Xte = Xs.values if tf is None else tf(Xs.values)
        p = model.predict_proba(Xte)[:, 1]; ext_probs[(mname, site)] = p
        mm = fold_metrics(ys, p); lo, hi = boot_ci(ys, p)
        ext_rows.append({'Model': mname, 'Site': SITE_LABEL[site], 'n': len(ys), 'AUC': mm['AUC'],
                         'AUC 95% CI': f'[{lo:.3f}, {hi:.3f}]', 'Accuracy': mm['Accuracy'],
                         'Sensitivity': mm['Sensitivity'], 'Specificity': mm['Specificity'],
                         'Brier': mm['Brier']})
ext = pd.DataFrame(ext_rows).set_index(['Site', 'Model']).sort_index()
display(ext.round(3)); ext.to_csv(f'{TAB_DIR}/E7_external_validation.csv')

delta_rows = []
for site, (Xs, ys) in SITES.items():
    for a, b in (('Fused (full system)', 'Clinical only'),
                 ('Fused, without ca/thal/slope', 'Clinical, without ca/thal/slope')):
        lo, hi, p = boot_delta(ys, ext_probs[(a, site)], ext_probs[(b, site)])
        d = roc_auc_score(ys, ext_probs[(a, site)]) - roc_auc_score(ys, ext_probs[(b, site)])
        delta_rows.append({'Site': SITE_LABEL[site], 'Comparison': f'{a} − {b}', 'ΔAUC': d,
                           '95% CI': f'[{lo:+.3f}, {hi:+.3f}]', 'bootstrap p': p})
ext_delta = pd.DataFrame(delta_rows).set_index(['Site', 'Comparison'])
display(ext_delta.round(3)); ext_delta.to_csv(f'{TAB_DIR}/E7_external_delta.csv')

# Edge case: cholesterol recorded as 0 (Switzerland/VA) — missing vs literal value
chol_rows = []
for site in ('switzerland', 'va'):
    Xs0, ys0 = load_uci_site(site, chol_zero_as_missing=False)
    for mname in ('Fused (full system)', 'Clinical only'):
        m = clone(EXT_MODELS[mname][0]).fit(X_cl_raw, y_cl)
        chol_rows.append({'Site': SITE_LABEL[site], 'Model': mname,
                          'AUC, chol=0 treated as missing': roc_auc_score(SITES[site][1], ext_probs[(mname, site)]),
                          'AUC, chol=0 taken literally': roc_auc_score(ys0, m.predict_proba(Xs0.values)[:, 1]),
                          'rows with chol=0': int((Xs0.chol == 0).sum())})
chol_df = pd.DataFrame(chol_rows).set_index(['Site', 'Model'])
display(chol_df.round(3)); chol_df.to_csv(f'{TAB_DIR}/E7_chol_zero_edge_case.csv')

fig, ax = plt.subplots(figsize=(9, 4))
w = 0.2
for i, mname in enumerate(EXT_MODELS):
    vals = [ext.loc[(SITE_LABEL[s], mname), 'AUC'] for s in SITES]
    ax.bar(np.arange(3) + (i - 1.5) * w, vals, w, label=mname)
ax.axhline(RESULTS[REF]['folds'].AUC.mean(), color='k', ls='--', lw=1, label='Cleveland internal CV (fused)')
ax.set_xticks(range(3)); ax.set_xticklabels([SITE_LABEL[s] for s in SITES]); ax.set_ylim(0.4, 1.0)
ax.set_ylabel('External AUC'); ax.legend(fontsize=8, ncol=2)
ax.set_title('Figure A6: External validation (trained on Cleveland only)', fontweight='bold')
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/figA6_external.png', dpi=200, bbox_inches='tight'); plt.show()
