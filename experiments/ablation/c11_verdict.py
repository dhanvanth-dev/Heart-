# ══════════════════════════════════════════════════════════════════════════
# 17B-11. VERDICT — the paper's claims re-tested under the controlled protocol
# A claim is "supported" only if the 95% CI of the paired difference excludes
# zero in the claimed direction.
# ══════════════════════════════════════════════════════════════════════════
print("── 17B-11 Claim-by-claim verdict ────────────────────────────────────")

def paired(a, b, m='AUC'):
    fa, fb = RESULTS[a]['folds'], RESULTS[b]['folds']; idx = fa.index.intersection(fb.index)
    d, hw, p = corrected_t(fa.loc[idx, m].values - fb.loc[idx, m].values)
    return d, d - hw, d + hw, p

def verdict(lo, hi, direction=+1):
    if direction > 0:
        return 'Supported' if lo > 0 else ('Contradicted' if hi < 0 else 'Not supported (CI includes 0)')
    return 'Supported' if hi < 0 else ('Contradicted' if lo > 0 else 'Not supported (CI includes 0)')

CLIN = '− ECG-proxy block (clinical only)'
rows = []
def add(claim, paper, a, b, m='AUC', direction=+1, scale=1.0, unit=''):
    d, lo, hi, p = paired(a, b, m)
    rows.append({'Claim in v4 paper': claim, 'Paper value': paper,
                 'Controlled estimate [95% CI]': f'{d * scale:+.3f}{unit} [{lo * scale:+.3f}, {hi * scale:+.3f}]',
                 'p': f'{p:.3f}', 'Verdict': verdict(lo, hi, direction)})

add('ECG-proxy fusion improves accuracy over clinical-only', '+7.04 pp', REF, CLIN, 'Accuracy', +1, 100, ' pp')
add('ECG-proxy fusion improves AUC over clinical-only', '+0.87 pp (0.9535 vs 0.9459)', REF, CLIN, 'AUC')
add('Removing the ECG-proxy layer lowers AUC', '≈ −0.06', CLIN, REF, 'AUC', -1)
add('Removing SMOTE causes the largest drop', '≈ −3.3% accuracy', '− SMOTE (no resampling)', REF, 'Accuracy', -1, 100, ' pp')
add('Removing SMOTE lowers AUC', '"noticeable drop"', '− SMOTE (no resampling)', REF, 'AUC', -1)
add('Ensemble beats a single (XGBoost) classifier', 'AUC −0.014 without ensemble', '− soft voting → XGBoost only', REF, 'AUC', -1)
add('Feature engineering contributes', 'AUC −0.005 without it', 'Clinical + 18 Gaussian-noise features', REF, 'AUC', -1)
add('ECG-proxy block beats an equal-size non-ECG block', '(implied)', REF, 'Clinical + 18 engineered non-ECG features', 'AUC')
if 'DL: dual branch + concatenation' in RESULTS:
    add('Cross-attention improves the dual-branch network', '(implied by MCF-BiModNet)',
        'DL: dual branch + cross-attention (L=1, paper)', 'DL: dual branch + concatenation', 'AUC')
verd = pd.DataFrame(rows).set_index('Claim in v4 paper')

extra = []
if 'rho_ff' in globals():
    extra.append({'Claim in v4 paper': 'SHAP ranking is a stable basis for clinical interpretation',
                  'Paper value': 'thal, ca, cp, oldpeak dominate',
                  'Controlled estimate [95% CI]': f'fold-to-fold Spearman ρ = {np.mean(rho_ff):.2f}; '
                                                  f'top-5 Jaccard = {np.mean(jac):.2f}',
                  'p': '—', 'Verdict': 'Partly supported (top features stable, tail unstable)'})
if 'ext' in globals():
    a = ext.xs('Fused (full system)', level='Model')['AUC']
    extra.append({'Claim in v4 paper': 'Performance generalises beyond Cleveland', 'Paper value': '(not tested)',
                  'Controlled estimate [95% CI]': ', '.join(f'{s} AUC={v:.3f}' for s, v in a.items()),
                  'p': '—', 'Verdict': 'Lower AUC at all external sites; fused ≈ clinical'})
if 'ho' in globals():
    extra.append({'Claim in v4 paper': 'Hold-out accuracy 88.52% is representative', 'Paper value': '0.8852 (seed 42)',
                  'Controlled estimate [95% CI]': f'{ho.fused_Acc.mean():.3f} over {len(ho)} splits '
                                                  f'(seed 42 = {stats.percentileofscore(ho.fused_Acc, s42.fused_Acc):.0f}th pct)',
                  'p': '—', 'Verdict': 'Optimistic single split'})
verd = pd.concat([verd, pd.DataFrame(extra).set_index('Claim in v4 paper')]) if extra else verd
display(verd); verd.to_csv(f'{TAB_DIR}/E11_verdict.csv')

# Master table of every variant evaluated in this section
master = []
for n, r in RESULTS.items():
    f = r['folds']; row = {'Variant': n, 'folds': len(f)}
    for m in ('AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1', 'Brier'):
        mu, hw, _ = corrected_t(f[m]); row[m] = mu; row[f'{m}_ci'] = hw
    master.append(row)
pd.DataFrame(master).set_index('Variant').to_csv(f'{TAB_DIR}/ALL_variants.csv')
print(f"\n{len(master)} variants evaluated; tables in {TAB_DIR}, figures in {FIG_DIR}")
