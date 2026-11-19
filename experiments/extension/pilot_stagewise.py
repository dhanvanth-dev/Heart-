"""Pilot for the proposed extension: stage-wise incremental value of the clinical work-up
(symptoms -> risk factors -> resting ECG -> exercise ECG / labs -> imaging) for angiographic CAD,
on patient-level paired data from five cohorts.

A. Z-Alizadeh Sani (n=303): nested logistic models, repeated 10-fold CV, paired dAUC with
   Nadeau-Bengio corrected CIs, and net benefit at the ESC 2019 testing threshold (15%).
B. UCI Cleveland, Hungary, Switzerland, VA: leave-one-hospital-out (internal-external) validation
   of the same stages, comparing median imputation with reduced-feature ("pattern") submodels
   for tests that are missing at the external hospital; random-effects pooling of site AUCs.

Run from the repository root:  python experiments/extension/pilot_stagewise.py
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
SEED = 42
OUT = 'results/extension'
os.makedirs(OUT, exist_ok=True)
PT = 0.15          # ESC 2019: non-invasive testing most beneficial when pre-test probability > 15%


def lr():
    return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(),
                         LogisticRegression(max_iter=5000, C=1.0))


def net_benefit(y, p, pt=PT):
    """Vickers & Elkin (2006): NB = TP/n - FP/n * pt/(1-pt)."""
    pos = p >= pt
    n = len(y)
    return (np.sum(pos & (y == 1)) - np.sum(pos & (y == 0)) * pt / (1 - pt)) / n


def corrected_ci(d, ratio=1 / 9):
    d = np.asarray(d); k = len(d)
    se = np.sqrt(np.var(d, ddof=1) * (1 / k + ratio))
    t = stats.t.ppf(0.975, k - 1)
    p = 2 * stats.t.sf(abs(d.mean() / se), k - 1) if se > 0 else 1.0
    return d.mean(), d.mean() - t * se, d.mean() + t * se, p


# ── A. Z-Alizadeh Sani ────────────────────────────────────────────────────
z = pd.read_excel('data/zalizadeh/z_alizadeh_sani.xlsx')
z.columns = [c.strip() for c in z.columns]
yz = (z['Cath'].str.strip() == 'Cad').astype(int).values
yn = lambda s: s.astype(str).str.strip().str.upper().isin(['Y', '1', 'YES']).astype(int)
Z = pd.DataFrame({'age': z['Age'], 'male': (z['Sex'].str.strip() == 'Male').astype(int),
                  'typical_cp': z['Typical Chest Pain'].astype(int), 'atypical_cp': yn(z['Atypical']),
                  'nonanginal_cp': yn(z['Nonanginal']), 'dyspnea': yn(z['Dyspnea'])})
RISK = {'dm': z['DM'], 'htn': z['HTN'], 'smoker': z['Current Smoker'], 'ex_smoker': z['EX-Smoker'],
        'fh': z['FH'], 'dlp': yn(z['DLP']), 'bmi': z['BMI']}
ECG = {'q_wave': z['Q Wave'], 'st_elev': z['St Elevation'], 'st_dep': z['St Depression'],
       't_inv': z['Tinversion'], 'lvh': yn(z['LVH']), 'poor_r': yn(z['Poor R Progression']),
       'bbb': (z['BBB'].astype(str).str.strip() != 'N').astype(int)}
LAB = {c.lower().replace(' ', '_'): z[c] for c in ['BP', 'PR', 'FBS', 'CR', 'TG', 'LDL', 'HDL', 'BUN', 'ESR',
                                                    'HB', 'K', 'Na', 'WBC', 'Lymph', 'Neut', 'PLT']}
ECHO = {'ef': z['EF-TTE'], 'rwma': z['Region RWMA'],
        'vhd': z['VHD'].astype(str).str.strip().map({'N': 0, 'mild': 1, 'Moderate': 2, 'Severe': 3}).fillna(0)}
for blk in (RISK, ECG, LAB, ECHO):
    for k, v in blk.items():
        Z[k] = pd.to_numeric(v, errors='coerce')
S1 = ['age', 'male', 'typical_cp', 'atypical_cp', 'nonanginal_cp', 'dyspnea']
STAGES_Z = {'S1 symptoms (age, sex, chest pain)': S1,
            'S2 + risk factors': S1 + list(RISK),
            'S3 + resting ECG': S1 + list(RISK) + list(ECG),
            'S4 + labs & vitals': S1 + list(RISK) + list(ECG) + list(LAB),
            'S5 + echocardiography': S1 + list(RISK) + list(ECG) + list(LAB) + list(ECHO),
            'Control: S2 + ECG only (no labs)': S1 + list(RISK) + list(ECG)}
splits = list(RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=SEED).split(Z, yz))
fold_auc = {k: [] for k in STAGES_Z}
oof = {k: np.zeros((5, len(yz))) for k in STAGES_Z}
for i, (tr, te) in enumerate(splits):
    for k, cols in STAGES_Z.items():
        p = lr().fit(Z.iloc[tr][cols], yz[tr]).predict_proba(Z.iloc[te][cols])[:, 1]
        fold_auc[k].append(roc_auc_score(yz[te], p)); oof[k][i // 10, te] = p
rows, names = [], list(STAGES_Z)[:5]
for j, k in enumerate(names):
    m, lo, hi, _ = corrected_ci(fold_auc[k])
    row = {'Stage': k, 'n features': len(STAGES_Z[k]), 'AUC': m, 'AUC 95% CI': f'[{lo:.3f}, {hi:.3f}]',
           'Net benefit @15%': np.mean([net_benefit(yz, o) for o in oof[k]])}
    if j:
        d, dlo, dhi, p = corrected_ci(np.array(fold_auc[k]) - np.array(fold_auc[names[j - 1]]))
        row.update({'dAUC vs previous stage': f'{d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]', 'p': round(p, 4)})
    rows.append(row)
treat_all = np.mean(yz) - (1 - np.mean(yz)) * PT / (1 - PT)
za = pd.DataFrame(rows).set_index('Stage')
print(f"\n=== A. Z-Alizadeh Sani (n={len(yz)}, CAD {yz.mean():.1%}) — logistic regression, 5x10 CV ===")
print(za.round(3).to_string())
print(f"Net benefit of 'test everyone' at 15%: {treat_all:.3f}")
za.to_csv(f'{OUT}/pilot_A_zalizadeh_stagewise.csv')

# ── B. UCI four hospitals: leave-one-hospital-out ─────────────────────────
COLS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
site_df = []
for s in ('cleveland', 'hungarian', 'switzerland', 'va'):
    d = pd.read_csv(f'data/uci/processed.{s}.data', header=None, names=COLS + ['num'], na_values='?').astype(float)
    d.loc[d.chol == 0, 'chol'] = np.nan
    d['site'] = s; site_df.append(d)
U = pd.concat(site_df, ignore_index=True)
yu = (U.pop('num') > 0).astype(int).values
U['typical_cp'] = (U.cp == 1).astype(int); U['atypical_cp'] = (U.cp == 2).astype(int)
U['nonanginal_cp'] = (U.cp == 3).astype(int)
U1 = ['age', 'sex', 'typical_cp', 'atypical_cp', 'nonanginal_cp']
STAGES_U = {'S1 symptoms': U1, 'S2 + risk factors': U1 + ['trestbps', 'chol', 'fbs'],
            'S3 + resting ECG': U1 + ['trestbps', 'chol', 'fbs', 'restecg'],
            'S4 + exercise ECG': U1 + ['trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope'],
            'S5 + fluoroscopy/thallium': U1 + ['trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']}


def pattern_predict(Xtr, ytr, Xte, cols):
    """Reduced-feature pattern submodels: for each missingness pattern in the test set, fit a model on
    the training patients that have all of that pattern's observed predictors."""
    p = np.zeros(len(Xte))
    obs = Xte[cols].notna().values
    for pat in {tuple(r) for r in obs}:
        use = [c for c, o in zip(cols, pat) if o]
        idx = np.where((obs == np.array(pat)).all(1))[0]
        trm = Xtr[use].notna().all(1).values if use else np.ones(len(Xtr), bool)
        if not use or len(np.unique(ytr[trm])) < 2:
            p[idx] = ytr.mean(); continue
        p[idx] = lr().fit(Xtr.loc[trm, use], ytr[trm]).predict_proba(Xte.iloc[idx][use])[:, 1]
    return p


def boot_se(y, p, B=500, seed=SEED):
    r = np.random.RandomState(seed); v = []
    for _ in range(B):
        b = r.randint(0, len(y), len(y))
        if len(np.unique(y[b])) == 2: v.append(roc_auc_score(y[b], p[b]))
    return np.std(v, ddof=1)


def dersimonian_laird(est, se):
    w = 1 / se ** 2; fe = np.sum(w * est) / w.sum(); q = np.sum(w * (est - fe) ** 2); dfree = len(est) - 1
    tau2 = max(0, (q - dfree) / (w.sum() - np.sum(w ** 2) / w.sum()))
    wr = 1 / (se ** 2 + tau2); m = np.sum(wr * est) / wr.sum(); s = np.sqrt(1 / wr.sum())
    i2 = max(0, (q - dfree) / q) if q > 0 else 0
    return m, m - 1.96 * s, m + 1.96 * s, i2


rowsB = []
for stage, cols in STAGES_U.items():
    for method in ('median imputation', 'pattern submodels'):
        aucs, ses = [], []
        for s in ('cleveland', 'hungarian', 'switzerland', 'va'):
            tr, te = (U.site != s).values, (U.site == s).values
            Xtr, Xte = U.loc[tr, cols].reset_index(drop=True), U.loc[te, cols].reset_index(drop=True)
            if method == 'median imputation':
                p = lr().fit(Xtr, yu[tr]).predict_proba(Xte)[:, 1]
            else:
                p = pattern_predict(Xtr, yu[tr], Xte, cols)
            a = roc_auc_score(yu[te], p); aucs.append(a); ses.append(boot_se(yu[te], p))
            rowsB.append({'Stage': stage, 'Method': method, 'Held-out site': s, 'AUC': a,
                          'Sensitivity@15%': np.mean(p[yu[te] == 1] >= PT),
                          'Specificity@15%': np.mean(p[yu[te] == 0] < PT),
                          'Net benefit@15%': net_benefit(yu[te], p)})
        lg = np.log(np.array(aucs) / (1 - np.array(aucs))); lse = np.array(ses) / (np.array(aucs) * (1 - np.array(aucs)))
        m, lo, hi, i2 = dersimonian_laird(lg, lse)
        inv = lambda x: 1 / (1 + np.exp(-x))
        rowsB.append({'Stage': stage, 'Method': method, 'Held-out site': 'POOLED (random effects)',
                      'AUC': inv(m), 'AUC 95% CI': f'[{inv(lo):.3f}, {inv(hi):.3f}]', 'I2': i2})
ub = pd.DataFrame(rowsB)
print(f"\n=== B. Leave-one-hospital-out, 4 UCI sites (n={len(yu)}) — logistic regression ===")
print(ub.pivot_table(index=['Stage', 'Method'], columns='Held-out site', values='AUC').round(3).to_string())
print(ub[ub['Held-out site'].str.startswith('POOLED')][['Stage', 'Method', 'AUC', 'AUC 95% CI', 'I2']].round(3).to_string(index=False))
ub.to_csv(f'{OUT}/pilot_B_uci_leave_one_hospital_out.csv', index=False)

# ── B2. Paired stage increments per held-out hospital (median imputation) ──
def paired_boot(y, p1, p0, B=2000, seed=SEED):
    r = np.random.RandomState(seed); v = []
    for _ in range(B):
        b = r.randint(0, len(y), len(y))
        if len(np.unique(y[b])) == 2: v.append(roc_auc_score(y[b], p1[b]) - roc_auc_score(y[b], p0[b]))
    return np.array(v)


preds = {}
for stage, cols in STAGES_U.items():
    for s in ('cleveland', 'hungarian', 'switzerland', 'va'):
        tr, te = (U.site != s).values, (U.site == s).values
        preds[(stage, s)] = lr().fit(U.loc[tr, cols], yu[tr]).predict_proba(U.loc[te, cols])[:, 1]
inc_rows = []
names_u = list(STAGES_U)
for a, b in zip(names_u[:-1], names_u[1:]):
    ds, ses = [], []
    for s in ('cleveland', 'hungarian', 'switzerland', 'va'):
        y = yu[(U.site == s).values]
        d = roc_auc_score(y, preds[(b, s)]) - roc_auc_score(y, preds[(a, s)]); bs = paired_boot(y, preds[(b, s)], preds[(a, s)])
        lo, hi = np.percentile(bs, [2.5, 97.5]); ds.append(d); ses.append(bs.std(ddof=1))
        inc_rows.append({'Increment': f'{a} -> {b}', 'Held-out site': s, 'dAUC': d, '95% CI': f'[{lo:+.3f}, {hi:+.3f}]'})
    m, lo, hi, i2 = dersimonian_laird(np.array(ds), np.array(ses))
    inc_rows.append({'Increment': f'{a} -> {b}', 'Held-out site': 'POOLED (random effects)', 'dAUC': m,
                     '95% CI': f'[{lo:+.3f}, {hi:+.3f}]', 'I2': i2})
inc = pd.DataFrame(inc_rows)
print("\n=== B2. Paired stage increments, leave-one-hospital-out (median imputation) ===")
print(inc.round(3).to_string(index=False))
inc.to_csv(f'{OUT}/pilot_B2_uci_stage_increments.csv', index=False)
