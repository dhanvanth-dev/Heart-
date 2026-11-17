# ══════════════════════════════════════════════════════════════════════════
# 17B-10. PTB-XL ECG BRANCH (D2) ABLATION
# D2 is the parallel ECG-only benchmark (PTB-XL labels, not Cleveland labels).
# Ablate: lead choice, tsfresh parameter set, 12-lead vs single lead,
# VarianceThreshold fitted inside vs outside the folds, classifier, SMOTE,
# training-set size (303 → 1212 records), and signal-quality subgroups.
# Records are read from data/ptbxl (or PTB_BASE_DIR in this notebook); missing
# ones are downloaded from PhysioNet (PTB-XL v1.0.3, open access).
# ══════════════════════════════════════════════════════════════════════════
print("── 17B-10 PTB-XL ECG-branch ablation ────────────────────────────────")
import ast, urllib.request
import wfdb
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute as ts_impute
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
from sklearn.feature_selection import VarianceThreshold

PTB_DIR = next((d for d in [globals().get('PTB_BASE_DIR'), 'data/ptbxl', '../data/ptbxl']
                if d and os.path.exists(os.path.join(d, 'ptbxl_database.csv'))), 'data/ptbxl')
PTB_URL = 'https://physionet.org/files/ptb-xl/1.0.3/'
os.makedirs(PTB_DIR, exist_ok=True)
for f in ('ptbxl_database.csv', 'scp_statements.csv'):
    if not os.path.exists(f'{PTB_DIR}/{f}'):
        urllib.request.urlretrieve(PTB_URL + f, f'{PTB_DIR}/{f}')

ptb = pd.read_csv(f'{PTB_DIR}/ptbxl_database.csv', index_col='ecg_id')
scp = pd.read_csv(f'{PTB_DIR}/scp_statements.csv', index_col=0)
scp = scp[scp.diagnostic == 1.0]

def _is_normal(raw):                       # same rule as Section 6
    d = {k: v for k, v in ast.literal_eval(raw).items() if k in scp.index and v > 0}
    return bool(d) and list(d) == ['NORM']
ptb['ecg_target'] = ptb.scp_codes.apply(lambda r: 0 if _is_normal(r) else 1)

def ptb_sample(n, seed=SEED):              # proportional stratified sample (Section 6)
    d0, d1 = ptb[ptb.ecg_target == 0], ptb[ptb.ecg_target == 1]
    n0 = int(n * len(d0) / len(ptb))
    return pd.concat([d0.sample(n=n0, random_state=seed),
                      d1.sample(n=n - n0, random_state=seed)]).sample(frac=1, random_state=seed)

N_BIG = int(os.environ.get('MCF_PTB_N', 1212))
s_big = ptb_sample(N_BIG)                  # the 303-record sample is a subset of this one
s_303 = ptb_sample(303)
assert set(s_303.index) <= set(s_big.index)

def read_signals(df):
    sig = []
    for fn in df.filename_lr:
        base = os.path.join(PTB_DIR, fn)
        for ext in ('hea', 'dat'):
            if not os.path.exists(f'{base}.{ext}'):
                os.makedirs(os.path.dirname(base), exist_ok=True)
                urllib.request.urlretrieve(f'{PTB_URL}{fn}.{ext}', f'{base}.{ext}')
        sig.append(wfdb.rdsamp(base)[0])
    return np.stack(sig)                   # (n, 1000, 12)

LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def tsfresh_lead(signals, ids, lead, params, tag):
    key = f'{ABL_DIR}/cache/tsfresh_{tag}_{lead}_{len(ids)}.pkl'
    if os.path.exists(key):
        return pickle.load(open(key, 'rb'))
    j = LEADS.index(lead)
    long = pd.DataFrame({'id': np.repeat(np.arange(len(ids)), signals.shape[1]),
                         'time': np.tile(np.arange(signals.shape[1]), len(ids)),
                         'value': signals[:, :, j].ravel()})
    F = extract_features(long, column_id='id', column_sort='time', column_value='value',
                         default_fc_parameters=params, impute_function=ts_impute,
                         disable_progressbar=True, n_jobs=os.cpu_count() or 1)
    F.index = ids; F.columns = [f'{lead}__{c}' for c in F.columns]
    pickle.dump(F, open(key, 'wb')); return F

t0 = time.time()
SIG_BIG = read_signals(s_big)
print(f"Loaded {len(SIG_BIG)} PTB-XL records in {time.time() - t0:.0f}s; "
      f"normal={np.sum(s_big.ecg_target == 0)}, abnormal={np.sum(s_big.ecg_target == 1)}")
pos303 = np.array([s_big.index.get_loc(i) for i in s_303.index])

F_eff = {'II': tsfresh_lead(SIG_BIG, s_big.index, 'II', EfficientFCParameters(), 'eff')}
for ld in ('I', 'V1', 'V5'):
    F_eff[ld] = tsfresh_lead(SIG_BIG[pos303], s_303.index, ld, EfficientFCParameters(), 'eff')
F_min = pd.concat([tsfresh_lead(SIG_BIG, s_big.index, ld, MinimalFCParameters(), 'min') for ld in LEADS], axis=1)
y303 = s_303.ecg_target.values
X_eff = {ld: F.loc[s_303.index].values for ld, F in F_eff.items()}
X_min_II = F_min.loc[s_303.index, [c for c in F_min.columns if c.startswith('II__')]].values
X_min_12 = F_min.loc[s_303.index].values
print(f"tsfresh features: Efficient Lead II={X_eff['II'].shape[1]}, Minimal 12-lead={X_min_12.shape[1]}")

ECG_SPLITS = list(RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS,
                                          random_state=SEED).split(X_eff['II'], y303))

def ecg_pipe(clf='voting', sampler='smote', vt=True):
    steps = [('vt', VarianceThreshold(0.01))] if vt else []
    steps += [('scale', StandardScaler())]
    if sampler:
        steps.append(('sampler', SAMPLERS[sampler]()))
    steps.append(('clf', make_clf(clf)))
    return ImbPipeline(steps)

X_glob_vt = StandardScaler().fit_transform(VarianceThreshold(0.01).fit_transform(X_eff['II']))
ECG_VARIANTS = {
    '[ECG] Lead II, Efficient tsfresh, soft voting (reference)': (ecg_pipe(), X_eff['II']),
    '[ECG] VarianceThreshold + scaling fitted on all records':   (ecg_pipe(vt=False), X_glob_vt),
    '[ECG] Random Forest only (Section 13 best)':                (ecg_pipe('rf'), X_eff['II']),
    '[ECG] No SMOTE':                                            (ecg_pipe(sampler=None), X_eff['II']),
    '[ECG] Minimal tsfresh (10 features), Lead II':              (ecg_pipe(), X_min_II),
    '[ECG] Lead I, Efficient':                                   (ecg_pipe(), X_eff['I']),
    '[ECG] Lead V1, Efficient':                                  (ecg_pipe(), X_eff['V1']),
    '[ECG] Lead V5, Efficient':                                  (ecg_pipe(), X_eff['V5']),
    '[ECG] 12-lead Minimal tsfresh (120 features)':              (ecg_pipe(), X_min_12),
    '[ECG] 12-lead Minimal + Lead II Efficient':                 (ecg_pipe(), np.hstack([X_min_12, X_eff['II']])),
}
for name, (pipe, X) in ECG_VARIANTS.items():
    evaluate(name, pipe, X=X, y=y303, splits=ECG_SPLITS)

_y_backup = y_cl
y_cl = y303                                      # summarize() uses y_cl for calibration slopes
ECG_REF = '[ECG] Lead II, Efficient tsfresh, soft voting (reference)'
abl10 = summarize(list(ECG_VARIANTS), ECG_REF, title='E10_ptbxl_branch')
y_cl = _y_backup
display(fmt_table(abl10))

# ── Training-set size: 303 → 606 → 1212 records (unpaired, own CV splits) ─
size_rows = []
X_big_II = F_eff['II'].values; y_big = s_big.ecg_target.values
for n in (303, 606, N_BIG):
    sub = np.arange(n) if n != 303 else pos303
    Xs, ys = X_big_II[sub], y_big[sub]
    spl = list(RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=3, random_state=SEED).split(Xs, ys))
    r = evaluate(f'[ECG-size] Lead II Efficient, n={n}', ecg_pipe(), X=Xs, y=ys, splits=spl)['folds']
    mu, hw, _ = corrected_t(r.AUC); ma, hwa, _ = corrected_t(r.Accuracy)
    size_rows.append({'Training records': n, 'Unique patients': s_big.iloc[sub].patient_id.nunique(),
                      'AUC [95% CI]': f'{mu:.3f} [{mu - hw:.3f}, {mu + hw:.3f}]',
                      'Accuracy [95% CI]': f'{ma:.3f} [{ma - hwa:.3f}, {ma + hwa:.3f}]'})
size_df = pd.DataFrame(size_rows).set_index('Training records')
display(size_df); size_df.to_csv(f'{TAB_DIR}/E10_ptbxl_sample_size.csv')

# ── Signal-quality subgroups (PTB-XL annotations) on out-of-fold predictions ─
noise_cols = ['baseline_drift', 'static_noise', 'burst_noise', 'electrodes_problems']
noisy = s_303[noise_cols].notna().any(axis=1).values
p_ecg = np.nanmean(RESULTS[ECG_REF]['oof'], axis=0)
qual = pd.DataFrame({
    'n': [len(y303), int((~noisy).sum()), int(noisy.sum())],
    'abnormal': [int(y303.sum()), int(y303[~noisy].sum()), int(y303[noisy].sum())],
    'AUC': [roc_auc_score(y303, p_ecg), roc_auc_score(y303[~noisy], p_ecg[~noisy]),
            roc_auc_score(y303[noisy], p_ecg[noisy])]},
    index=['All 303 records', 'No noise annotation', 'Any noise/electrode annotation'])
display(qual.round(3)); qual.to_csv(f'{TAB_DIR}/E10_ptbxl_signal_quality.csv')

forest_plot(abl10.drop(ECG_REF), 'Figure A10: PTB-XL ECG branch — ΔAUC vs Lead II reference',
            'figA10_ptbxl_branch.png', ref_label='Lead II Efficient')
