# ══════════════════════════════════════════════════════════════════════════
# 17B. EXTENDED ABLATION STUDY — shared setup
# ──────────────────────────────────────────────────────────────────────────
# Every experiment in this section uses the SAME protocol so that variants are
# directly comparable (reviewer action-plan row 16):
#   • raw Cleveland data (missing values kept) → all preprocessing happens
#     INSIDE each training fold: median imputation → ECG-proxy features →
#     StandardScaler → resampling (training rows only) → classifier
#   • repeated stratified 10-fold CV (N_REPEATS × 10 folds, identical splits
#     for every variant)
#   • identical, fixed hyper-parameters for every variant (the values found by
#     RandomizedSearchCV in Section 15), i.e. the same tuning budget
#   • metrics per fold: AUC, accuracy, sensitivity, specificity, F1, Brier
#   • uncertainty: 95% CI from the Nadeau–Bengio corrected resampled t-test,
#     which accounts for the overlap between CV training sets; differences
#     between variants are paired by fold; p-values are Holm-corrected
# ══════════════════════════════════════════════════════════════════════════
import os, time, json, pickle, hashlib, warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              VotingClassifier, StackingClassifier)
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import (SMOTE, SMOTENC, ADASYN, BorderlineSMOTE,
                                    RandomOverSampler)
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'          # also silences joblib worker processes
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

SEED      = 42
N_FOLDS   = 10
N_REPEATS = int(os.environ.get('MCF_REPEATS', 5))      # 5 × 10 = 50 folds
N_JOBS    = int(os.environ.get('MCF_JOBS', -1))
ABL_DIR   = os.environ.get('MCF_ABL_DIR', 'outputs/ablation')
FIG_DIR, TAB_DIR = f'{ABL_DIR}/figures', f'{ABL_DIR}/tables'
for d in (FIG_DIR, TAB_DIR, f'{ABL_DIR}/cache'):
    os.makedirs(d, exist_ok=True)

# ── Data: raw UCI files (missing values kept as NaN) ──────────────────────
CLIN_COLS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
             'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
CAT_COLS  = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
CONT_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
UCI_URL   = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/'

def load_uci_site(site, chol_zero_as_missing=True):
    """Load one site of the UCI heart-disease data. Target: num > 0 → 1."""
    for root in ('data/uci', '../data/uci'):
        path = f'{root}/processed.{site}.data'
        if os.path.exists(path):
            break
    else:
        path = UCI_URL + f'processed.{site}.data'
    df = pd.read_csv(path, header=None, names=CLIN_COLS + ['num'], na_values='?')
    y = (df.pop('num') > 0).astype(int).values
    df = df.astype(float)
    if chol_zero_as_missing:          # Switzerland/VA code missing cholesterol as 0
        df.loc[df['chol'] == 0, 'chol'] = np.nan
    return df, y

Xdf_cl, y_cl = load_uci_site('cleveland')
X_cl_raw = Xdf_cl.values
print(f"Cleveland: {X_cl_raw.shape}, healthy={np.sum(y_cl == 0)} "
      f"({np.mean(y_cl == 0):.1%}), disease={np.sum(y_cl == 1)} ({np.mean(y_cl == 1):.1%}), "
      f"missing cells={int(np.isnan(X_cl_raw).sum())}")

# ── ECG-proxy features (identical formulas to Section 9) ──────────────────
PROXY_GROUPS = {
    'raw':         ['restecg', 'oldpeak', 'slope', 'thalach', 'exang'],
    'polynomial':  ['oldpeak_sq', 'thalach_sq'],
    'interaction': ['oldpeak_x_slope', 'thalach_x_exang', 'oldpeak_x_thalach',
                    'restecg_x_oldpeak', 'slope_x_thalach'],
    'log':         ['log_oldpeak', 'log_thalach'],
    'ratio':       ['oldpeak_per_thalach', 'thalach_per_slope'],
    'threshold':   ['high_oldpeak', 'low_thalach', 'abnormal_restecg', 'downslope'],
    'hr_reserve':  ['hr_reserve', 'hr_reserve_pct'],
    'composite':   ['ischaemia_score'],
}
ALL_GROUPS = tuple(PROXY_GROUPS)

def ecg_proxy_frame(d, fix_downslope=False):
    """23 ECG-proxy columns. fix_downslope=True uses the correct UCI code (3)."""
    e = pd.DataFrame(index=d.index)
    for c in PROXY_GROUPS['raw']:
        e[c] = d[c]
    op_s = np.clip(d['oldpeak'], 0, None); th_s = np.clip(d['thalach'], 1, None)
    e['oldpeak_sq'] = d['oldpeak'] ** 2;            e['thalach_sq'] = d['thalach'] ** 2
    e['oldpeak_x_slope'] = d['oldpeak'] * d['slope']
    e['thalach_x_exang'] = d['thalach'] * d['exang']
    e['oldpeak_x_thalach'] = d['oldpeak'] * d['thalach']
    e['restecg_x_oldpeak'] = d['restecg'] * d['oldpeak']
    e['slope_x_thalach'] = d['slope'] * d['thalach']
    e['log_oldpeak'] = np.log1p(op_s);              e['log_thalach'] = np.log(th_s)
    e['oldpeak_per_thalach'] = op_s / th_s
    e['thalach_per_slope'] = th_s / np.clip(d['slope'], 1, None)
    e['high_oldpeak'] = (d['oldpeak'] > 1.0).astype(int)
    e['low_thalach'] = (d['thalach'] < 140).astype(int)
    e['abnormal_restecg'] = (d['restecg'] > 0).astype(int)
    e['downslope'] = (d['slope'] == (3 if fix_downslope else 2)).astype(int)
    pred_max = np.clip(220 - np.clip(d['age'], 1, None), 1, None)
    e['hr_reserve'] = th_s - pred_max;              e['hr_reserve_pct'] = e['hr_reserve'] / pred_max
    e['ischaemia_score'] = op_s * (d['slope'] + 1) + d['exang'] * 2
    return e

NONECG_NAMES = ['age_sq', 'chol_sq', 'trestbps_sq', 'age_x_chol', 'age_x_trestbps',
                'chol_x_trestbps', 'sex_x_age', 'fbs_x_chol', 'log_chol', 'log_trestbps',
                'chol_per_age', 'trestbps_per_age', 'high_chol', 'high_bp', 'old_age',
                'cp_x_age', 'ca_x_age', 'thal_x_sex']

def nonecg_control_frame(d):
    """18 engineered NON-ECG features (same transform types as the proxy block).
    Control for 'does adding ANY engineered features help?'"""
    e = pd.DataFrame(index=d.index)
    e['age_sq'] = d['age'] ** 2; e['chol_sq'] = d['chol'] ** 2; e['trestbps_sq'] = d['trestbps'] ** 2
    e['age_x_chol'] = d['age'] * d['chol']; e['age_x_trestbps'] = d['age'] * d['trestbps']
    e['chol_x_trestbps'] = d['chol'] * d['trestbps']; e['sex_x_age'] = d['sex'] * d['age']
    e['fbs_x_chol'] = d['fbs'] * d['chol']
    e['log_chol'] = np.log(np.clip(d['chol'], 1, None)); e['log_trestbps'] = np.log(np.clip(d['trestbps'], 1, None))
    e['chol_per_age'] = d['chol'] / np.clip(d['age'], 1, None)
    e['trestbps_per_age'] = d['trestbps'] / np.clip(d['age'], 1, None)
    e['high_chol'] = (d['chol'] > 240).astype(int); e['high_bp'] = (d['trestbps'] > 140).astype(int)
    e['old_age'] = (d['age'] >= 55).astype(int); e['cp_x_age'] = d['cp'] * d['age']
    e['ca_x_age'] = d['ca'] * d['age']; e['thal_x_sex'] = d['thal'] * d['sex']
    return e

ONEHOT_LEVELS = {'cp': [1, 2, 3, 4], 'restecg': [0, 1, 2], 'slope': [1, 2, 3], 'thal': [3, 6, 7]}

class ProxyFeatures(BaseEstimator, TransformerMixin):
    """Fold-safe feature builder working on the (imputed) 13 clinical columns.

    groups        : ECG-proxy groups to append (empty tuple → clinical only)
    keep_clinical : keep the 13 clinical columns
    drop_dup_raw  : drop proxy 'raw' columns that duplicate clinical ones
    extra         : None | 'noise' | 'nonecg' | 'shuffled_proxy'  (control blocks)
    onehot        : one-hot encode cp/restecg/slope/thal instead of ordinal codes
    """
    def __init__(self, groups=ALL_GROUPS, keep_clinical=True, fix_downslope=False,
                 drop_dup_raw=False, extra=None, onehot=False, seed=SEED):
        self.groups, self.keep_clinical, self.fix_downslope = groups, keep_clinical, fix_downslope
        self.drop_dup_raw, self.extra, self.onehot, self.seed = drop_dup_raw, extra, onehot, seed

    def fit(self, X, y=None):
        return self

    def _cols(self):
        cols = [c for g in self.groups for c in PROXY_GROUPS[g]]
        if self.drop_dup_raw:
            cols = [c for c in cols if c not in PROXY_GROUPS['raw']]
        return cols

    def transform(self, X):
        d = pd.DataFrame(np.asarray(X, dtype=float), columns=CLIN_COLS)
        parts, names = [], []
        if self.keep_clinical:
            if self.onehot:
                ordinal = [c for c in CLIN_COLS if c not in ONEHOT_LEVELS]
                parts.append(d[ordinal].values); names += ordinal
                for c, lv in ONEHOT_LEVELS.items():
                    for v in lv:
                        parts.append((np.round(d[c].values) == v).astype(float)[:, None])
                        names.append(f'{c}={v}')
            else:
                parts.append(d.values); names += CLIN_COLS
        cols = self._cols()
        if cols:
            P = ecg_proxy_frame(d, self.fix_downslope)[cols].values
            if self.extra == 'shuffled_proxy':     # break the row ↔ patient link
                rng = np.random.RandomState(self.seed + len(d))
                P = P[rng.permutation(len(d))]
            parts.append(P); names += [f'proxy:{c}' if c in CLIN_COLS else c for c in cols]
        if self.extra == 'nonecg':
            parts.append(nonecg_control_frame(d).values); names += NONECG_NAMES
        elif self.extra == 'noise':
            rng = np.random.RandomState(self.seed + len(d))
            parts.append(rng.randn(len(d), 18)); names += [f'noise_{i}' for i in range(18)]
        self.feature_names_ = names
        return np.nan_to_num(np.hstack(parts), nan=0.0, posinf=0.0, neginf=0.0)

def feature_names(feat):
    f = clone(feat)
    f.fit(None).transform(SimpleImputer(strategy='median').fit_transform(X_cl_raw))
    return f.feature_names_

# ── Models: fixed hyper-parameters from Section 15 (same budget everywhere) ─
TUNED = {
    'xgb': dict(n_estimators=200, max_depth=3, learning_rate=0.01, subsample=1.0,
                colsample_bytree=0.7, min_child_weight=1, gamma=0),
    'svm': dict(kernel='rbf', C=100, gamma=0.001),
    'gbm': dict(n_estimators=200, max_depth=3, learning_rate=0.01, subsample=1.0, max_features=None),
    'rf':  dict(n_estimators=500, max_depth=20, min_samples_split=5, max_features='log2'),
}
POS_WEIGHT = float(np.sum(y_cl == 0) / np.sum(y_cl == 1))

def make_base(name, balanced=False):
    cw = 'balanced' if balanced else None
    if name == 'xgb':
        return XGBClassifier(**TUNED['xgb'], eval_metric='logloss', random_state=SEED,
                             verbosity=0, n_jobs=1,
                             scale_pos_weight=POS_WEIGHT if balanced else 1.0)
    if name == 'svm':
        return SVC(**TUNED['svm'], probability=True, random_state=SEED, class_weight=cw)
    if name == 'gbm':
        return GradientBoostingClassifier(**TUNED['gbm'], random_state=SEED)
    if name == 'rf':
        return RandomForestClassifier(**TUNED['rf'], random_state=SEED, n_jobs=1, class_weight=cw)
    if name == 'lr':
        return LogisticRegression(max_iter=2000, random_state=SEED, class_weight=cw)
    raise ValueError(name)

class HardVoting(BaseEstimator, ClassifierMixin):
    """Hard (majority) voting; the score used for AUC is the fraction of votes."""
    def __init__(self, members=('xgb', 'svm', 'gbm', 'rf')):
        self.members = members
    def fit(self, X, y):
        self.models_ = [make_base(m).fit(X, y) for m in self.members]
        self.classes_ = np.array([0, 1]); return self
    def predict_proba(self, X):
        v = np.mean([m.predict(X) for m in self.models_], axis=0)
        return np.column_stack([1 - v, v])
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

def make_clf(kind='voting', balanced=False):
    members = [(m, make_base(m, balanced)) for m in ('xgb', 'svm', 'gbm', 'rf')]
    if kind == 'voting':
        return VotingClassifier(members, voting='soft')
    if kind == 'hard_voting':
        return HardVoting()
    if kind == 'stacking':
        return StackingClassifier(members, final_estimator=LogisticRegression(max_iter=2000),
                                  cv=5, stack_method='predict_proba')
    return make_base(kind, balanced)

SAMPLERS = {
    None:          lambda: None,
    'smote':       lambda: SMOTE(random_state=SEED),
    'ros':         lambda: RandomOverSampler(random_state=SEED),
    'borderline':  lambda: BorderlineSMOTE(random_state=SEED),
    'adasyn':      lambda: ADASYN(random_state=SEED),
    'smotenc':     lambda: SMOTENC(categorical_features=[CLIN_COLS.index(c) for c in CAT_COLS],
                                   random_state=SEED),
}

def make_pipe(feat=None, sampler='smote', clf='voting', sampler_stage='post', balanced=False,
              impute=True, scale=True):
    """sampler_stage='post' (as in the paper): resample AFTER features+scaling.
       sampler_stage='pre' : resample the raw imputed clinical rows, THEN derive
                             the proxy features (synthetic patients stay internally
                             consistent, e.g. oldpeak_sq == oldpeak**2)."""
    feat = ProxyFeatures() if feat is None else feat
    smp = SAMPLERS[sampler]()
    steps = []
    if impute:
        steps.append(('impute', SimpleImputer(strategy='median', keep_empty_features=True)))
    if smp is not None and sampler_stage == 'pre':
        steps.append(('sampler', smp))
    steps.append(('features', feat))
    if scale:
        steps.append(('scale', StandardScaler()))
    if smp is not None and sampler_stage == 'post':
        steps.append(('sampler', smp))
    steps.append(('clf', make_clf(clf, balanced) if isinstance(clf, str) else clf))
    return ImbPipeline(steps)

FEAT_FULL     = lambda **kw: ProxyFeatures(**kw)
FEAT_CLINICAL = lambda **kw: ProxyFeatures(groups=(), **kw)

# ── Evaluation engine ──────────────────────────────────────────────────────
SPLITS = list(RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS,
                                      random_state=SEED).split(X_cl_raw, y_cl))
TEST_TRAIN_RATIO = 1 / (N_FOLDS - 1)
RESULTS = {}            # name → dict(folds=DataFrame, oof=array[repeats, n])

def fold_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {'AUC': roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
            'Accuracy': (tp + tn) / len(y),
            'Sensitivity': tp / (tp + fn) if (tp + fn) else np.nan,
            'Specificity': tn / (tn + fp) if (tn + fp) else np.nan,
            'F1': f1_score(y, yhat, zero_division=0),
            'Brier': brier_score_loss(y, p)}

def _fit_predict(pipe, X, y, tr, te, k, train_tf, test_tf):
    Xtr, ytr, Xte = X[tr].copy(), y[tr].copy(), X[te].copy()
    if train_tf is not None:
        Xtr, ytr = train_tf(Xtr, ytr, SEED + k)
    if test_tf is not None:
        Xte = test_tf(Xte, SEED + 10_000 + k)
    m = clone(pipe).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]

def _cache_key(name, n_splits):
    return f'{ABL_DIR}/cache/' + hashlib.md5(f'{name}|{n_splits}|{N_REPEATS}'.encode()).hexdigest() + '.pkl'

def evaluate(name, pipe, X=None, y=None, splits=None, train_tf=None, test_tf=None,
             n_jobs=None, use_cache=True, verbose=True):
    """Run `pipe` on every split; store per-fold metrics and out-of-fold probabilities."""
    X = X_cl_raw if X is None else X
    y = y_cl if y is None else y
    splits = SPLITS if splits is None else splits
    key = _cache_key(name, len(splits))
    if use_cache and os.path.exists(key):
        RESULTS[name] = pickle.load(open(key, 'rb'))
    else:
        t0 = time.time()
        probs = Parallel(n_jobs=N_JOBS if n_jobs is None else n_jobs)(
            delayed(_fit_predict)(pipe, X, y, tr, te, k, train_tf, test_tf)
            for k, (tr, te) in enumerate(splits))
        rows, oof = [], np.full((int(np.ceil(len(splits) / N_FOLDS)), len(y)), np.nan)
        for k, ((tr, te), p) in enumerate(zip(splits, probs)):
            r = fold_metrics(y[te], p); r['fold'] = k; rows.append(r)
            oof[k // N_FOLDS, te] = p
        RESULTS[name] = {'folds': pd.DataFrame(rows).set_index('fold'), 'oof': oof,
                         'seconds': time.time() - t0}
        pickle.dump(RESULTS[name], open(key, 'wb'))
    if verbose:
        f = RESULTS[name]['folds']
        print(f"  {name:<46} AUC={f.AUC.mean():.4f}  Acc={f.Accuracy.mean():.4f}  "
              f"Sens={f.Sensitivity.mean():.4f}  Spec={f.Specificity.mean():.4f}")
    return RESULTS[name]

# ── Statistics ─────────────────────────────────────────────────────────────
def corrected_t(diffs, ratio=TEST_TRAIN_RATIO):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated CV.
    Returns mean, 95% CI half-width, two-sided p-value."""
    d = np.asarray(diffs, float); d = d[~np.isnan(d)]; k = len(d)
    var = np.var(d, ddof=1) * (1 / k + ratio)
    se = np.sqrt(var) if var > 0 else 0.0
    t_crit = stats.t.ppf(0.975, k - 1)
    if se == 0:
        return d.mean(), 0.0, 1.0 if d.mean() == 0 else 0.0
    t = d.mean() / se
    return d.mean(), t_crit * se, 2 * stats.t.sf(abs(t), k - 1)

def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx]); adj[idx] = min(1.0, run)
    return adj

def calibration_slope(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(lp[:, None], y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])

def summarize(names, ref, metrics=('AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1', 'Brier'),
              title=None, labels=None):
    """Table: mean [95% CI] per metric, ΔAUC vs `ref` with corrected CI and Holm p."""
    rows = []
    ref_f = RESULTS[ref]['folds']
    for n in names:
        f = RESULTS[n]['folds']; idx = f.index.intersection(ref_f.index)
        row = {'Variant': labels.get(n, n) if labels else n}
        for m in metrics:
            mu, hw, _ = corrected_t(f.loc[idx, m] - 0.0)
            row[m] = mu; row[f'{m}_lo'] = mu - hw; row[f'{m}_hi'] = mu + hw
        slopes = [calibration_slope(y_cl[~np.isnan(o)], o[~np.isnan(o)])[0]
                  for o in RESULTS[n]['oof'] if np.isfinite(o).sum() > 50]
        row['CalSlope'] = float(np.mean(slopes)) if slopes else np.nan
        for m in ('AUC', 'Accuracy', 'Sensitivity'):
            d, hw, p = corrected_t(f.loc[idx, m].values - ref_f.loc[idx, m].values)
            tag = 'Δ' + m
            row[tag] = d; row[f'{tag}_lo'] = d - hw; row[f'{tag}_hi'] = d + hw
            row[f'p_{m}'] = p if n != ref else np.nan
        rows.append(row)
    df = pd.DataFrame(rows).set_index('Variant')
    mask = df['p_AUC'].notna()
    if mask.any():
        df.loc[mask, 'p_AUC_holm'] = holm(df.loc[mask, 'p_AUC'])
        df.loc[mask, 'p_Accuracy_holm'] = holm(df.loc[mask, 'p_Accuracy'])
    if title:
        df.to_csv(f'{TAB_DIR}/{title}.csv')
    return df

def fmt_table(df, cols=('AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1', 'Brier')):
    """Compact human-readable view: value [lo, hi]."""
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = [f'{v:.3f} [{lo:.3f}, {hi:.3f}]' for v, lo, hi in
                  zip(df[c], df[f'{c}_lo'], df[f'{c}_hi'])]
    out['ΔAUC [95% CI]'] = [f'{v:+.3f} [{lo:+.3f}, {hi:+.3f}]' for v, lo, hi in
                            zip(df['ΔAUC'], df['ΔAUC_lo'], df['ΔAUC_hi'])]
    if 'p_AUC_holm' in df:
        out['p (Holm)'] = [('—' if np.isnan(p) else f'{p:.3f}') for p in df['p_AUC_holm']]
    if 'CalSlope' in df:
        out['Cal. slope'] = df['CalSlope'].round(2)
    return out

def forest_plot(df, title, fname, metric='ΔAUC', ref_label='reference'):
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(df) + 1.4))
    yy = np.arange(len(df))[::-1]
    lo, hi, v = df[f'{metric}_lo'], df[f'{metric}_hi'], df[metric]
    sig = df.get('p_AUC_holm', pd.Series(np.nan, index=df.index)).fillna(1) < 0.05
    colors = ['#c0392b' if (s and val < 0) else '#1e8449' if (s and val > 0) else '#555'
              for s, val in zip(sig, v)]
    ax.hlines(yy, lo, hi, color=colors, lw=2)
    ax.scatter(v, yy, color=colors, zorder=3, s=28)
    ax.axvline(0, color='k', lw=0.8, ls='--')
    ax.set_yticks(yy); ax.set_yticklabels(df.index, fontsize=9)
    ax.set_xlabel(f'{metric} vs {ref_label} (95% corrected CI)')
    ax.set_title(title, fontweight='bold', fontsize=11)
    plt.tight_layout(); plt.savefig(f'{FIG_DIR}/{fname}', dpi=200, bbox_inches='tight')
    plt.show()

print(f"Protocol: {N_REPEATS}×{N_FOLDS}-fold repeated stratified CV = {len(SPLITS)} folds; "
      f"outputs → {ABL_DIR}")
