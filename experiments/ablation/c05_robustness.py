# ══════════════════════════════════════════════════════════════════════════
# 17B-5. ROBUSTNESS & EDGE CASES — does the ECG-proxy block help when the
# data are harder?  Each scenario is applied identically (same random seed per
# fold) to the full system and to the clinical-only system, so the comparison
# is paired.  3 × 10-fold CV to bound the run time.
# ══════════════════════════════════════════════════════════════════════════
print("── 17B-5 Robustness and edge cases ──────────────────────────────────")
ROB_SPLITS = SPLITS[:3 * N_FOLDS]
FUSED_PIPE, CLIN_PIPE = ABL1[REF], ABL1['− ECG-proxy block (clinical only)']
col = {c: i for i, c in enumerate(CLIN_COLS)}

def flip_labels(rate):
    def tf(X, y, seed):
        rng = np.random.RandomState(seed); y = y.copy()
        idx = rng.choice(len(y), int(round(rate * len(y))), replace=False); y[idx] = 1 - y[idx]
        return X, y
    return tf

def mcar(rate):
    def tf(X, seed):
        rng = np.random.RandomState(seed); X = X.copy(); X[rng.rand(*X.shape) < rate] = np.nan
        return X
    return tf

def mcar_train(rate):
    f = mcar(rate)
    return lambda X, y, seed: (f(X, seed), y)

def gauss_noise(sigma):
    sd = np.nanstd(X_cl_raw[:, [col[c] for c in CONT_COLS]], axis=0)
    def tf(X, seed):
        rng = np.random.RandomState(seed); X = X.copy()
        for j, c in enumerate(CONT_COLS):
            X[:, col[c]] += rng.randn(len(X)) * sigma * sd[j]
        return X
    return tf

def subsample(frac):
    def tf(X, y, seed):
        idx, _ = train_test_split(np.arange(len(y)), train_size=frac, stratify=y, random_state=seed)
        return X[idx], y[idx]
    return tf

def unavailable(cols):          # feature never measured (train AND test)
    def tf(X, seed=None):
        X = X.copy(); X[:, [col[c] for c in cols]] = np.nan; return X
    return tf

def unavailable_train(cols):
    f = unavailable(cols)
    return lambda X, y, seed: (f(X), y)

NO_ANGIO, NO_EXERCISE = ['ca', 'thal'], ['thalach', 'exang', 'oldpeak', 'slope']
SCENARIOS = {'Baseline (no perturbation)': (None, None)}
for r in (0.05, 0.10, 0.20):
    SCENARIOS[f'Training label noise {int(r * 100)}%'] = (flip_labels(r), None)
for r in (0.10, 0.20, 0.30):
    SCENARIOS[f'Test-time missing values (MCAR) {int(r * 100)}%'] = (None, mcar(r))
SCENARIOS['Missing values in train AND test 20%'] = (mcar_train(0.2), mcar(0.2))
for s in (0.25, 0.5, 1.0):
    SCENARIOS[f'Measurement noise σ={s}×SD (test)'] = (None, gauss_noise(s))
for fr in (0.25, 0.50, 0.75):
    SCENARIOS[f'Training set reduced to {int(fr * 100)}%'] = (subsample(fr), None)
SCENARIOS['No angiography/thallium (ca, thal unavailable)'] = (unavailable_train(NO_ANGIO), unavailable(NO_ANGIO))
SCENARIOS['No exercise test (thalach, exang, oldpeak, slope)'] = (unavailable_train(NO_EXERCISE), unavailable(NO_EXERCISE))
SCENARIOS['Resting data only (both unavailable)'] = (unavailable_train(NO_ANGIO + NO_EXERCISE),
                                                     unavailable(NO_ANGIO + NO_EXERCISE))
rob_rows = []
for sc_name, (trf, tef) in SCENARIOS.items():
    for tag, pipe in (('fused', FUSED_PIPE), ('clinical', CLIN_PIPE)):
        evaluate(f'[rob] {sc_name} | {tag}', pipe, splits=ROB_SPLITS, train_tf=trf, test_tf=tef,
                 verbose=False)
    fu = RESULTS[f'[rob] {sc_name} | fused']['folds']; cl = RESULTS[f'[rob] {sc_name} | clinical']['folds']
    d, hw, p = corrected_t(fu.AUC.values - cl.AUC.values)
    da, hwa, pa = corrected_t(fu.Accuracy.values - cl.Accuracy.values)
    rob_rows.append({'Scenario': sc_name, 'AUC fused': fu.AUC.mean(), 'AUC clinical': cl.AUC.mean(),
                     'ΔAUC': d, 'ΔAUC_lo': d - hw, 'ΔAUC_hi': d + hw, 'p_AUC': p,
                     'Sens fused': fu.Sensitivity.mean(), 'Sens clinical': cl.Sensitivity.mean(),
                     'ΔAcc': da, 'p_Acc': pa})
rob = pd.DataFrame(rob_rows).set_index('Scenario')
rob['p_AUC_holm'] = holm(rob['p_AUC'].values)
rob.to_csv(f'{TAB_DIR}/E5_robustness.csv')
show = rob[['AUC fused', 'AUC clinical']].round(3)
show['ΔAUC [95% CI]'] = [f'{v:+.3f} [{lo:+.3f}, {hi:+.3f}]' for v, lo, hi in zip(rob['ΔAUC'], rob['ΔAUC_lo'], rob['ΔAUC_hi'])]
show['p (Holm)'] = rob['p_AUC_holm'].round(3)
show['Sens fused / clinical'] = [f'{a:.3f} / {b:.3f}' for a, b in zip(rob['Sens fused'], rob['Sens clinical'])]
display(show)
forest_plot(rob,'Figure A4: Fused − clinical ΔAUC under perturbations',
            'figA4_robustness.png', ref_label='clinical only (same perturbation)')

# ── Subgroup analysis on out-of-fold predictions (averaged over repeats) ─────
p_fu = np.nanmean(RESULTS[REF]['oof'], axis=0)
p_cl = np.nanmean(RESULTS['− ECG-proxy block (clinical only)']['oof'], axis=0)
GROUPS = {'All patients': np.ones(len(y_cl), bool),
          'Female': Xdf_cl.sex.values == 0, 'Male': Xdf_cl.sex.values == 1,
          'Age < 55': Xdf_cl.age.values < 55, 'Age ≥ 55': Xdf_cl.age.values >= 55,
          'Asymptomatic chest pain (cp=4)': Xdf_cl.cp.values == 4,
          'Symptomatic chest pain (cp=1–3)': Xdf_cl.cp.values != 4,
          'Normal resting ECG': Xdf_cl.restecg.values == 0,
          'Abnormal resting ECG': Xdf_cl.restecg.values > 0}

def boot_auc(mask, B=2000, seed=SEED):
    rng = np.random.RandomState(seed); idx = np.where(mask)[0]; yy = y_cl[idx]
    a_f, a_c, d = [], [], []
    for _ in range(B):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y_cl[b])) < 2: continue
        af, ac = roc_auc_score(y_cl[b], p_fu[b]), roc_auc_score(y_cl[b], p_cl[b])
        a_f.append(af); a_c.append(ac); d.append(af - ac)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {'n': len(idx), 'disease': int(yy.sum()), 'AUC fused': roc_auc_score(yy, p_fu[idx]),
            'AUC clinical': roc_auc_score(yy, p_cl[idx]),
            'ΔAUC': roc_auc_score(yy, p_fu[idx]) - roc_auc_score(yy, p_cl[idx]),
            'ΔAUC 95% CI': f'[{lo:+.3f}, {hi:+.3f}]'}

sub = pd.DataFrame({g: boot_auc(m) for g, m in GROUPS.items()}).T
display(sub); sub.to_csv(f'{TAB_DIR}/E5_subgroups.csv')
