# ══════════════════════════════════════════════════════════════════════════
# 17B-4. EVALUATION PROTOCOL & LEAKAGE (reviewer rows 8, 13, 16)
#  (a) preprocessing fitted on all 303 rows before CV (as in Sections 4–9)
#  (b) SMOTE applied to the whole dataset before CV (classic leakage)
#  (c) nested hyper-parameter tuning vs the fixed Section-15 values
#  (d) hold-out split sensitivity: 200 random 80/20 splits instead of seed 42
# ══════════════════════════════════════════════════════════════════════════
print("── 17B-4 Protocol and leakage ───────────────────────────────────────")
from sklearn.model_selection import RandomizedSearchCV

def ci_row(name, label=None):
    f = RESULTS[name]['folds']; row = {'Variant': label or name}
    for m in ('AUC', 'Accuracy', 'Sensitivity', 'Specificity'):
        mu, hw, _ = corrected_t(f[m]); row[m] = f'{mu:.3f} [{mu - hw:.3f}, {mu + hw:.3f}]'
    return row

# (a) global preprocessing (imputation + proxy + scaling fitted on all rows)
X_glob = StandardScaler().fit_transform(
    ProxyFeatures().fit(None).transform(SimpleImputer(strategy='median').fit_transform(X_cl_raw)))
leaky_a = ImbPipeline([('sampler', SMOTE(random_state=SEED)), ('clf', make_clf('voting'))])
evaluate('Leaky (a): preprocessing fitted on all rows', leaky_a, X=X_glob)

# (b) SMOTE on the full dataset, then CV (synthetic neighbours of test rows in training)
X_sm_all, y_sm_all = SMOTE(random_state=SEED).fit_resample(X_glob, y_cl)
splits_sm = list(RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS,
                                         random_state=SEED).split(X_sm_all, y_sm_all))
evaluate('Leaky (b): SMOTE before CV', ImbPipeline([('clf', make_clf('voting'))]),
         X=X_sm_all, y=y_sm_all, splits=splits_sm)

# (c) nested tuning of XGBoost (inner 3-fold randomized search, 20 candidates)
xgb_space = {'clf__n_estimators': [200, 300, 500], 'clf__max_depth': [2, 3, 4],
             'clf__learning_rate': [0.01, 0.05, 0.1], 'clf__subsample': [0.7, 0.8, 1.0],
             'clf__colsample_bytree': [0.7, 0.8, 1.0], 'clf__min_child_weight': [1, 3, 5],
             'clf__gamma': [0, 0.1, 0.2]}
nested = RandomizedSearchCV(make_pipe(FEAT_FULL(), clf='xgb'), xgb_space, n_iter=20, cv=3,
                            scoring='roc_auc', random_state=SEED, n_jobs=1)
evaluate('XGBoost, nested tuning (inner 3-fold)', nested)
evaluate('XGBoost, library defaults', make_pipe(FEAT_FULL(), clf=XGBClassifier(
    eval_metric='logloss', random_state=SEED, verbosity=0, n_jobs=1)))
evaluate('− soft voting → XGBoost only', ABL1['− soft voting → XGBoost only'], verbose=False)

prot = pd.DataFrame([ci_row(REF, 'Leak-free CV (reference)'),
                     ci_row('Leaky (a): preprocessing fitted on all rows'),
                     ci_row('Leaky (b): SMOTE before CV'),
                     ci_row('− soft voting → XGBoost only', 'XGBoost, fixed Section-15 params'),
                     ci_row('XGBoost, nested tuning (inner 3-fold)'),
                     ci_row('XGBoost, library defaults')]).set_index('Variant')
display(prot); prot.to_csv(f'{TAB_DIR}/E4_protocol_leakage.csv')

# (d) hold-out sensitivity: the paper's 0.8852 comes from ONE 80/20 split
N_SPLITS_HO = int(os.environ.get('MCF_HOLDOUT_SPLITS', 200))
ho_key = f'{ABL_DIR}/cache/holdout_{N_SPLITS_HO}.pkl'

def _holdout(seed):
    tr, te = train_test_split(np.arange(len(y_cl)), test_size=0.2, stratify=y_cl, random_state=seed)
    out = {'seed': seed}
    for tag, pipe in (('fused', ABL1[REF]), ('clinical', ABL1['− ECG-proxy block (clinical only)'])):
        p = clone(pipe).fit(X_cl_raw[tr], y_cl[tr]).predict_proba(X_cl_raw[te])[:, 1]
        mm = fold_metrics(y_cl[te], p)
        out[f'{tag}_AUC'], out[f'{tag}_Acc'] = mm['AUC'], mm['Accuracy']
    return out

if os.path.exists(ho_key):
    ho = pickle.load(open(ho_key, 'rb'))
else:
    ho = pd.DataFrame(Parallel(n_jobs=N_JOBS)(delayed(_holdout)(s) for s in range(N_SPLITS_HO)))
    pickle.dump(ho, open(ho_key, 'wb'))
ho['dAcc'] = ho.fused_Acc - ho.clinical_Acc; ho['dAUC'] = ho.fused_AUC - ho.clinical_AUC
s42 = ho.loc[ho.seed == 42].iloc[0]
q = lambda s: f'{s.mean():.3f} (2.5–97.5%: {s.quantile(.025):.3f}–{s.quantile(.975):.3f})'
ho_summary = pd.DataFrame({
    'Metric': ['Fused accuracy', 'Clinical accuracy', 'Fused − clinical accuracy',
               'Fused AUC', 'Clinical AUC', 'Fused − clinical AUC'],
    f'Over {N_SPLITS_HO} random 80/20 splits': [q(ho.fused_Acc), q(ho.clinical_Acc), q(ho.dAcc),
                                                  q(ho.fused_AUC), q(ho.clinical_AUC), q(ho.dAUC)],
    'Seed 42 (paper split)': [f'{s42.fused_Acc:.3f}', f'{s42.clinical_Acc:.3f}', f'{s42.dAcc:+.3f}',
                              f'{s42.fused_AUC:.3f}', f'{s42.clinical_AUC:.3f}', f'{s42.dAUC:+.3f}'],
    'Percentile of seed 42': [f'{stats.percentileofscore(ho[c], s42[c]):.0f}'
                              for c in ('fused_Acc', 'clinical_Acc', 'dAcc', 'fused_AUC', 'clinical_AUC', 'dAUC')],
}).set_index('Metric')
display(ho_summary); ho_summary.to_csv(f'{TAB_DIR}/E4_holdout_sensitivity.csv')
print(f"Fused beats clinical (accuracy) on {np.mean(ho.dAcc > 0):.0%} of splits, ties on "
      f"{np.mean(ho.dAcc == 0):.0%}, loses on {np.mean(ho.dAcc < 0):.0%}.")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (a, b, lab) in zip(axes, [('fused_Acc', 'clinical_Acc', 'Accuracy'), ('fused_AUC', 'clinical_AUC', 'AUC')]):
    ax.hist(ho[a], bins=25, alpha=.6, label='Fused (full system)', color='#2980b9')
    ax.hist(ho[b], bins=25, alpha=.6, label='Clinical only', color='#e67e22')
    ax.axvline(s42[a], color='#2980b9', ls='--', lw=2); ax.axvline(s42[b], color='#e67e22', ls='--', lw=2)
    ax.set_xlabel(f'Hold-out {lab} (61 patients)'); ax.legend(fontsize=8)
    ax.set_title(f'{lab}: {N_SPLITS_HO} random splits (dashed = seed 42)', fontweight='bold', fontsize=10)
plt.suptitle('Figure A3: Hold-out estimates depend strongly on the split', fontweight='bold')
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/figA3_holdout_sensitivity.png', dpi=200, bbox_inches='tight'); plt.show()
