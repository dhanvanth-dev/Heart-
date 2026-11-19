"""Pilot analysis: does cross-modal electromechanical timing add to ECG and PCG features?

Run from the repository root after extract_features.py:  python experiments/fusion/analyse.py
"""
import os
import warnings

import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
OUT = 'results/fusion'
F = pd.read_csv(f'{OUT}/features_training_a.csv', index_col=0)
ECG = [c for c in F if c.startswith('ecg_')]
PCG = [c for c in F if c.startswith('pcg_')]
X_ = ['x_emat', 'x_rs1', 'x_qs2', 'x_emw', 'x_pct_emat', 'x_emat_sd', 'x_qs2_sd', 'x_rs1_sd']
SETS = {'ECG only': ECG, 'PCG only': PCG, 'Cross-modal timing only': X_,
        'ECG + PCG (concatenation)': ECG + PCG, 'ECG + PCG + cross-modal timing': ECG + PCG + X_}
lines = []
log = lambda s='': (print(s), lines.append(s))

# ── 1. Physiological validation of the extracted intervals (normal recordings) ─
n = F[F.diagnosis == 'Normal'].dropna(subset=['ecg_hr', 'x_qs2', 'pcg_s1s2_interval', 'x_emat'])
log(f'## 1. Physiological validation (normal recordings, n={len(n)}, groups={n.group_c.nunique()})')
log('Expected (Weissler 1968): all systolic time intervals shorten as heart rate rises; steepest for QS2 and '
    'ejection time, smallest for the pre-ejection period. S1 follows the R wave; S2 lies near the end of T.')
rows = []
for c, lab in [('x_qs2', 'QS2 (Q -> S2 onset)'), ('pcg_s1s2_interval', 'S1 -> S2 interval (ejection proxy)'),
               ('x_emat', 'EMAT (Q -> S1 centre)')]:
    sl = stats.linregress(n.ecg_hr, n[c]); rho = stats.spearmanr(n.ecg_hr, n[c])
    rows.append({'Interval': lab, 'median (ms)': n[c].median(), 'slope (ms per bpm)': sl.slope,
                 'slope 95% CI': f'[{sl.slope - 1.96 * sl.stderr:.2f}, {sl.slope + 1.96 * sl.stderr:.2f}]',
                 'Spearman rho vs HR': rho.correlation, 'p': rho.pvalue})
phys = pd.DataFrame(rows).set_index('Interval'); log(phys.round(3).to_string())
log(f"R -> S1 onset median {F.x_rs1.median():.1f} ms (IQR {F.x_rs1.quantile(.25):.1f} to {F.x_rs1.quantile(.75):.1f}); "
    f"EMW median {F.x_emw.median():.1f} ms")
phys.to_csv(f'{OUT}/physiology_validation.csv')


# ── helpers ───────────────────────────────────────────────────────────────
def lr():
    return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(),
                         LogisticRegression(max_iter=5000, C=0.5))


def xgb():
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=2, eval_metric='logloss', random_state=0, verbosity=0, n_jobs=4)


def cv_auc(cols, y, groups, model, n_rep=20, grouped=True):
    """Repeated (grouped) 5-fold CV. Returns list of per-fold AUCs, ordered identically for every call."""
    aucs = []
    for rep in range(n_rep):
        cv = (StratifiedGroupKFold(5, shuffle=True, random_state=rep) if grouped
              else StratifiedKFold(5, shuffle=True, random_state=rep))
        for tr, te in (cv.split(F, y, groups) if grouped else cv.split(F, y)):
            if len(np.unique(y[te])) < 2:
                aucs.append(np.nan); continue
            m = model().fit(F.iloc[tr][cols], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(F.iloc[te][cols])[:, 1]))
    return np.array(aucs)


def corrected(d, ratio=1 / 4):
    d = d[np.isfinite(d)]; k = len(d)
    se = np.sqrt(np.var(d, ddof=1) * (1 / k + ratio)); t = stats.t.ppf(.975, k - 1)
    return d.mean(), d.mean() - t * se, d.mean() + t * se, 2 * stats.t.sf(abs(d.mean() / se), k - 1)


groups = F.group_c.values
results = []
for target in ('label_pathological', 'label_original'):
    y = F[target].values
    log(f'\n## Target: {target}  (positives {y.sum()}/{len(y)} recordings)')
    # ── 2. Leakage: grouped vs ungrouped CV ─────────────────────────────
    full = SETS['ECG + PCG + cross-modal timing']
    for mname, model in (('LogReg', lr), ('XGBoost', xgb)):
        a_g = cv_auc(full, y, groups, model); a_u = cv_auc(full, y, groups, model, grouped=False)
        log(f'{mname}, full feature set: record-level CV AUC {np.nanmean(a_u):.3f} vs patient-grouped CV '
            f'{np.nanmean(a_g):.3f}  (inflation {np.nanmean(a_u) - np.nanmean(a_g):+.3f})')
        results.append({'target': target, 'model': mname, 'set': 'LEAKAGE ungrouped', 'AUC': np.nanmean(a_u)})
    # ── 3. Modality ablation (grouped CV, identical folds) ──────────────
    for mname, model in (('LogReg', lr), ('XGBoost', xgb)):
        aucs = {s: cv_auc(cols, y, groups, model) for s, cols in SETS.items()}
        base = aucs['ECG + PCG (concatenation)']
        for s, a in aucs.items():
            m, lo, hi, _ = corrected(a - 0)
            row = {'target': target, 'model': mname, 'set': s, 'AUC': m, 'AUC CI': f'[{lo:.3f}, {hi:.3f}]'}
            if s == 'ECG + PCG + cross-modal timing':
                d, dlo, dhi, p = corrected(a - base)
                row.update({'dAUC vs concat': f'{d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]', 'p': p})
            if s == 'Cross-modal timing only':
                d, dlo, dhi, p = corrected(a - aucs['ECG only'])
                row.update({'dAUC vs ECG only': f'{d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]', 'p_vs_ecg': p})
            results.append(row)
R = pd.DataFrame(results)
log('\n## 3. Modality ablation (patient-grouped 5-fold CV x 20 repeats; CIs corrected for repeated CV)')
log(R[R.set != 'LEAKAGE ungrouped'].drop(columns='target').round(4).to_string(index=False)
    if False else '')
for target in ('label_pathological', 'label_original'):
    log(f'\n-- {target}')
    log(R[(R.target == target) & (R.set != 'LEAKAGE ungrouped')].drop(columns='target').round(4).to_string(index=False))
R.to_csv(f'{OUT}/modality_ablation.csv', index=False)

# ── 4. SHAP on the full XGBoost model (pathological target) ─────────────
y = F['label_pathological'].values
full = SETS['ECG + PCG + cross-modal timing']
m = xgb().fit(F[full], y)
sv = shap.TreeExplainer(m).shap_values(F[full])
imp = pd.Series(np.abs(sv).mean(0), index=full).sort_values(ascending=False)
fam = {c: 'ECG' if c in ECG else 'PCG' if c in PCG else 'cross-modal' for c in full}
share = imp.groupby(lambda c: fam[c]).sum() / imp.sum()
direction = {c: stats.spearmanr(F[c], sv[:, i], nan_policy='omit').correlation for i, c in enumerate(full)}
top = pd.DataFrame({'mean|SHAP|': imp, 'family': [fam[c] for c in imp.index],
                    'direction (rho value vs SHAP)': [direction[c] for c in imp.index]}).head(12)
log('\n## 4. SHAP (XGBoost, pathological target, fitted on all recordings — descriptive)')
log(top.round(3).to_string()); log('share of |SHAP| by family: ' + ', '.join(f'{k} {v:.0%}' for k, v in share.items()))
top.to_csv(f'{OUT}/shap_top.csv')

# ── 5. Diagnosis-specific physiology (descriptive) ────────────────────────
log('\n## 5. Rate-corrected intervals by diagnosis (median; groups in brackets)')
F['qs2_index'] = F.x_qs2 + 2.1 * F.ecg_hr          # HR-adjusted, slope from the normal-recording regression is reported above
desc = F.groupby('diagnosis').agg(n=('x_qs2', 'size'), groups=('group_c', 'nunique'), EMAT=('x_emat', 'median'),
                                  pct_EMAT=('x_pct_emat', 'median'), QS2=('x_qs2', 'median'),
                                  S1S2=('pcg_s1s2_interval', 'median'), EMW=('x_emw', 'median'), HR=('ecg_hr', 'median'))
log(desc.round(1).to_string())
open(f'{OUT}/pilot_report.txt', 'w').write('\n'.join(lines))
