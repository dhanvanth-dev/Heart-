# ══════════════════════════════════════════════════════════════════════════
# 17B-8. SHAP STABILITY & AGREEMENT (reviewer rows 6 & 18)
# SHAP is post-hoc and associational; with N=303 and correlated / duplicated
# proxy features its rankings may be unstable. We measure:
#   • fold-to-fold Spearman correlation of global mean|SHAP| rankings
#   • seed-to-seed stability (10 XGBoost seeds, same data)
#   • agreement with permutation importance (AUC drop) on the same test folds
#   • how attribution is split between duplicated columns
# SHAP is computed for the XGBoost member (TreeSHAP), as in Section 18.
# ══════════════════════════════════════════════════════════════════════════
print("── 17B-8 SHAP stability ─────────────────────────────────────────────")
import shap
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline as SkPipeline

FEATS36 = feature_names(FEAT_FULL())
shap_key = f'{ABL_DIR}/cache/shap_stability_{len(SPLITS)}.pkl'

def _shap_fold(k, tr, te):
    m = clone(make_pipe(FEAT_FULL(), clf='xgb')).fit(X_cl_raw[tr], y_cl[tr])
    prep = SkPipeline(m.steps[:3]); Zte = prep.transform(X_cl_raw[te]); clf = m.named_steps['clf']
    sv = shap.TreeExplainer(clf).shap_values(Zte)
    pi = permutation_importance(clf, Zte, y_cl[te], scoring='roc_auc', n_repeats=10, random_state=SEED + k)
    return k, te, sv, Zte, np.abs(sv).mean(0), pi.importances_mean

if os.path.exists(shap_key):
    shap_out = pickle.load(open(shap_key, 'rb'))
else:
    shap_out = Parallel(n_jobs=N_JOBS)(delayed(_shap_fold)(k, tr, te) for k, (tr, te) in enumerate(SPLITS))
    pickle.dump(shap_out, open(shap_key, 'wb'))

G = np.array([o[4] for o in shap_out])          # folds × 36 mean|SHAP|
PI = np.array([o[5] for o in shap_out])          # folds × 36 permutation importance
rank = pd.DataFrame(G, columns=FEATS36).rank(axis=1, ascending=False)

iu = np.triu_indices(len(G), 1)
rho_ff = stats.spearmanr(G.T).correlation[iu]
top5 = [set(np.argsort(-g)[:5]) for g in G]
jac = [len(top5[i] & top5[j]) / len(top5[i] | top5[j]) for i, j in zip(*iu)]
rho_pi = [stats.spearmanr(g, p).correlation for g, p in zip(G, PI)]

# seed stability: full data, 10 seeds, stochastic subsampling enabled
imp_all = SimpleImputer(strategy='median').fit(X_cl_raw)
Z_all = StandardScaler().fit_transform(ProxyFeatures().fit(None).transform(imp_all.transform(X_cl_raw)))
seed_imp = []
for s in range(10):
    Zs, ys = SMOTE(random_state=s).fit_resample(Z_all, y_cl)
    xm = XGBClassifier(**{**TUNED['xgb'], 'subsample': 0.8}, eval_metric='logloss',
                       random_state=s, verbosity=0, n_jobs=1).fit(Zs, ys)
    seed_imp.append(np.abs(shap.TreeExplainer(xm).shap_values(Z_all)).mean(0))
seed_imp = np.array(seed_imp)
rho_seed = stats.spearmanr(seed_imp.T).correlation[np.triu_indices(10, 1)]

ci = lambda a: f'{np.mean(a):.3f} (2.5–97.5%: {np.percentile(a, 2.5):.3f}–{np.percentile(a, 97.5):.3f})'
stab = pd.DataFrame({'Stability measure': [
    'Spearman ρ of SHAP rankings, fold vs fold (all 36 features)',
    'Jaccard overlap of top-5 SHAP features, fold vs fold',
    'Spearman ρ of SHAP rankings, seed vs seed (full data)',
    'Spearman ρ, SHAP vs permutation importance (same fold)'],
    'Value': [ci(rho_ff), ci(jac), ci(rho_seed), ci(rho_pi)]}).set_index('Stability measure')
display(stab); stab.to_csv(f'{TAB_DIR}/E8_shap_stability.csv')

order = np.argsort(-G.mean(0))
top_tab = pd.DataFrame({
    'feature': [FEATS36[i] for i in order[:15]],
    'block': ['clinical' if i < 13 else 'ECG-proxy' for i in order[:15]],
    'mean |SHAP|': G.mean(0)[order[:15]],
    'median rank (IQR)': [f'{rank.iloc[:, i].median():.0f} ({rank.iloc[:, i].quantile(.25):.0f}–{rank.iloc[:, i].quantile(.75):.0f})'
                          for i in order[:15]],
    '% folds in top-5': [f'{np.mean([i in t for t in top5]):.0%}' for i in order[:15]],
    'perm. importance (ΔAUC)': PI.mean(0)[order[:15]],
}).set_index('feature')
display(top_tab.round(4)); top_tab.to_csv(f'{TAB_DIR}/E8_shap_top_features.csv')

share_proxy = G[:, 13:].sum(1) / G.sum(1)
dup = pd.DataFrame({c: {'clinical copy': G[:, FEATS36.index(c)].mean(),
                        'proxy copy': G[:, FEATS36.index(f'proxy:{c}')].mean()}
                    for c in PROXY_GROUPS['raw']}).T
dup['proxy share of total'] = dup['proxy copy'] / dup.sum(1)
print(f"Share of total |SHAP| carried by the ECG-proxy block: {ci(share_proxy)}")
display(dup.round(4)); dup.to_csv(f'{TAB_DIR}/E8_duplicate_attribution_split.csv')

# Figure A7: rank stability of the top-15 features
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.boxplot([rank.iloc[:, i] for i in order[:15]], vert=False)
ax.set_yticks(range(1, 16)); ax.set_yticklabels([FEATS36[i] for i in order[:15]])
ax.invert_yaxis(); ax.set_xlabel(f'SHAP rank across {len(G)} CV folds (1 = most important)')
ax.set_title('Figure A7: Stability of the SHAP feature ranking', fontweight='bold')
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/figA7_shap_rank_stability.png', dpi=200, bbox_inches='tight'); plt.show()

# Figure A8: dependence plots on pooled out-of-fold SHAP values (repeat 1)
rep0 = [o for o in shap_out if o[0] < N_FOLDS]
SV0 = np.zeros((len(y_cl), len(FEATS36))); Z0 = np.zeros_like(SV0)
for _, te, sv, Zte, *_ in rep0:
    SV0[te], Z0[te] = sv, Zte
raw_imp = imp_all.transform(X_cl_raw)
fig, axes = plt.subplots(1, 5, figsize=(20, 3.8))
for ax, c in zip(axes, ['oldpeak', 'ca', 'thal', 'cp', 'thalach']):
    j = FEATS36.index(c)
    ax.scatter(raw_imp[:, CLIN_COLS.index(c)] + np.random.RandomState(0).uniform(-.08, .08, len(y_cl)),
               SV0[:, j], c=y_cl, cmap='coolwarm', s=14, alpha=.75, edgecolors='none')
    ax.axhline(0, color='k', lw=.7); ax.set_xlabel(c); ax.set_ylabel('SHAP value')
plt.suptitle('Figure A8: SHAP dependence (out-of-fold; red = disease)', fontweight='bold')
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/figA8_shap_dependence.png', dpi=200, bbox_inches='tight'); plt.show()

# Figure A9: local explanations for one TP, FN, FP and TN patient
pred = RESULTS[REF]['oof'][0]          # soft-voting out-of-fold probability, repeat 1
cases = {'True positive': (y_cl == 1) & (pred >= .5), 'False negative': (y_cl == 1) & (pred < .5),
         'False positive': (y_cl == 0) & (pred >= .5), 'True negative': (y_cl == 0) & (pred < .5)}
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, (lab, msk) in zip(axes, cases.items()):
    idx = np.where(msk)[0]
    if not len(idx):
        ax.set_axis_off(); continue
    i = idx[np.argmax(np.abs(pred[idx] - .5))]
    top = np.argsort(-np.abs(SV0[i]))[:8][::-1]
    ax.barh([FEATS36[t] for t in top], SV0[i, top], color=['#c0392b' if v > 0 else '#2980b9' for v in SV0[i, top]])
    ax.axvline(0, color='k', lw=.7)
    ax.set_title(f'{lab} (patient {i}, p={pred[i]:.2f})', fontweight='bold', fontsize=10)
plt.suptitle('Figure A9: Local SHAP explanations (XGBoost member, out-of-fold)', fontweight='bold')
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/figA9_shap_local.png', dpi=200, bbox_inches='tight'); plt.show()
