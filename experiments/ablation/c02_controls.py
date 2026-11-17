# ══════════════════════════════════════════════════════════════════════════
# 17B-2. CONTROL EXPERIMENTS — is the ECG-proxy block special?
# If the proxy features carry genuine extra signal, the full system should beat
#   (a) clinical + 18 engineered NON-ECG features (same transform types),
#   (b) clinical + 18 pure-noise columns,
#   (c) clinical + the proxy block row-shuffled (realistic marginals, no link
#       to the patient).
# A label-permutation test checks that the full system is above chance at all.
# ══════════════════════════════════════════════════════════════════════════
print("── 17B-2 Control experiments ────────────────────────────────────────")
ABL2 = {
    REF: ABL1[REF],
    'Clinical only': ABL1['− ECG-proxy block (clinical only)'],
    'Clinical + 18 engineered non-ECG features': make_pipe(FEAT_CLINICAL(extra='nonecg')),
    'Clinical + 18 Gaussian-noise features':     make_pipe(FEAT_CLINICAL(extra='noise')),
    'Clinical + row-shuffled ECG-proxy block':   make_pipe(FEAT_FULL(drop_dup_raw=True, extra='shuffled_proxy')),
    'Clinical + ECG-proxy + non-ECG features':   make_pipe(FEAT_FULL(extra='nonecg')),
}
for name, pipe in ABL2.items():
    evaluate(name, pipe)
abl2 = summarize(list(ABL2), REF, title='E2_controls')
display(fmt_table(abl2))

# ── Label-permutation test (Ojala & Garriga 2010) ─────────────────────────
N_PERM = int(os.environ.get('MCF_NPERM', 100))
perm_key = f'{ABL_DIR}/cache/perm_{N_PERM}.pkl'
skf5 = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_cl_raw, y_cl))

def _perm_auc(pipe, yy):
    p = np.zeros(len(yy))
    for tr, te in skf5:
        p[te] = clone(pipe).fit(X_cl_raw[tr], yy[tr]).predict_proba(X_cl_raw[te])[:, 1]
    return roc_auc_score(yy, p)

if os.path.exists(perm_key):
    obs_auc, null_auc = pickle.load(open(perm_key, 'rb'))
else:
    rng = np.random.RandomState(SEED)
    obs_auc = _perm_auc(ABL1[REF], y_cl)
    null_auc = np.array(Parallel(n_jobs=N_JOBS)(
        delayed(_perm_auc)(ABL1[REF], rng.permutation(y_cl)) for _ in range(N_PERM)))
    pickle.dump((obs_auc, null_auc), open(perm_key, 'wb'))
p_perm = (1 + np.sum(null_auc >= obs_auc)) / (1 + len(null_auc))
print(f"\nPermutation test: observed AUC={obs_auc:.3f}; null AUC={null_auc.mean():.3f}±{null_auc.std():.3f}; "
      f"p={p_perm:.4f} ({len(null_auc)} permutations)")
pd.DataFrame({'observed_auc': [obs_auc], 'null_mean': [null_auc.mean()], 'null_sd': [null_auc.std()],
              'null_max': [null_auc.max()], 'p_value': [p_perm], 'n_perm': [len(null_auc)]}
             ).to_csv(f'{TAB_DIR}/E2_permutation_test.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={'width_ratios': [1.4, 1]})
yy = np.arange(len(abl2))[::-1]
axes[0].errorbar(abl2['AUC'], yy, xerr=[abl2['AUC'] - abl2['AUC_lo'], abl2['AUC_hi'] - abl2['AUC']],
                 fmt='o', color='#2c3e50', capsize=3)
axes[0].set_yticks(yy); axes[0].set_yticklabels(abl2.index, fontsize=9)
axes[0].set_xlabel('AUC (95% corrected CI)'); axes[0].set_title('Feature-block controls', fontweight='bold')
axes[1].hist(null_auc, bins=20, color='#bdc3c7', edgecolor='k')
axes[1].axvline(obs_auc, color='#c0392b', lw=2, label=f'observed {obs_auc:.3f}')
axes[1].set_xlabel('AUC under permuted labels'); axes[1].legend()
axes[1].set_title(f'Label-permutation test (p={p_perm:.3f})', fontweight='bold')
plt.suptitle('Figure A2: Control experiments', fontweight='bold')
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/figA2_controls.png', dpi=200, bbox_inches='tight'); plt.show()
