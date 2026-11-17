"""Generate LaTeX tables and number macros for the paper from results/ablation/tables.

Every number quoted in paper/main.tex comes from here, so the text cannot drift
from the experiments. Run from the repository root:  python paper/make_tables.py
"""
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, 'results', 'ablation', 'tables')
OUT = os.path.join(ROOT, 'paper', 'generated')
os.makedirs(OUT, exist_ok=True)

UNI = {'−': '$-$', 'Δ': '$\\Delta$', '≥': '$\\geq$', '≤': '$\\leq$', '≡': '$\\equiv$', 'σ': '$\\sigma$',
       '×': '$\\times$', '→': '$\\rightarrow$', '–': '--', '≈': '$\\approx$', 'ρ': '$\\rho$', '★': '',
       '≠': '$\\neq$', '²': '$^2$', '|': '$|$'}


def tex(s):
    s = str(s)
    for a, b in (('&', '\\&'), ('%', '\\%'), ('_', '\\_'), ('#', '\\#')):
        s = s.replace(a, b)
    for a, b in UNI.items():
        s = s.replace(a, b)
    return s


def read(name, **kw):
    return pd.read_csv(os.path.join(TAB, name), **kw)


def ci(v, lo, hi, d=3, sign=False):
    f = f'{{:+.{d}f}}' if sign else f'{{:.{d}f}}'
    return f'{f.format(v)} [{f.format(lo)}, {f.format(hi)}]'


def pfmt(p):
    if pd.isna(p):
        return '--'
    return '$<$0.001' if p < 0.001 else f'{p:.3f}'


def write_table(fname, header, rows, colspec, caption, label, star=False, size='\\scriptsize', note=None):
    env = 'table*' if star else 'table'
    lines = [f'\\begin{{{env}}}[!t]', '\\centering', size, f'\\caption{{{caption}}}', f'\\label{{{label}}}',
             '\\setlength{\\tabcolsep}{3pt}', '\\begin{adjustbox}{max width=\\linewidth}', f'\\begin{{tabular}}{{{colspec}}}', '\\toprule',
             ' & '.join(header) + ' \\\\', '\\midrule']
    for r in rows:
        lines.append('\\midrule' if r == 'MIDRULE' else ' & '.join(r) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{adjustbox}']
    if note:
        lines.append(f'\\\\[2pt]\\parbox{{\\linewidth}}{{\\tiny {note}}}')
    lines.append(f'\\end{{{env}}}')
    open(os.path.join(OUT, fname), 'w').write('\n'.join(lines) + '\n')


MACROS = {}

# ── Cohort demographics (computed directly from the raw UCI file) ────────
COLS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
cl = pd.read_csv(os.path.join(ROOT, 'data', 'uci', 'processed.cleveland.data'), header=None, names=COLS, na_values='?')
cl['y'] = (cl.num > 0).astype(int)
from scipy import stats as _st  # noqa: E402


def _cont(c):
    a, b = cl.loc[cl.y == 0, c].dropna(), cl.loc[cl.y == 1, c].dropna()
    return [f'{a.mean():.1f} $\\pm$ {a.std():.1f}', f'{b.mean():.1f} $\\pm$ {b.std():.1f}', pfmt(_st.mannwhitneyu(a, b).pvalue)]


def _bin(mask_fn, label):
    a, b = cl[cl.y == 0], cl[cl.y == 1]
    na, nb = int(mask_fn(a).sum()), int(mask_fn(b).sum())
    p = _st.chi2_contingency([[na, len(a) - na], [nb, len(b) - nb]])[1]
    return [label, f'{na} ({na / len(a):.0%})'.replace('%', '\\%'), f'{nb} ({nb / len(b):.0%})'.replace('%', '\\%'), pfmt(p)]


demo_rows = [['Patients, n', str(int((cl.y == 0).sum())), str(int((cl.y == 1).sum())), '--'],
             ['Age, years'] + _cont('age'),
             _bin(lambda d: d.sex == 0, 'Female, n (\\%)'),
             _bin(lambda d: d.cp == 4, 'Asymptomatic chest pain'),
             ['Resting BP, mm Hg'] + _cont('trestbps'),
             ['Cholesterol, mg/dL'] + _cont('chol'),
             ['Max.\\ heart rate'] + _cont('thalach'),
             _bin(lambda d: d.exang == 1, 'Exercise angina'),
             ['ST depression (oldpeak)'] + _cont('oldpeak'),
             _bin(lambda d: d.fbs == 1, 'Fasting glucose $>$120')]
write_table('tab_demo.tex', ['Characteristic', 'No CAD', 'CAD', '$p$'], demo_rows, 'lccc',
            'Cleveland cohort by class (mean $\\pm$ SD or n (\\%); Mann--Whitney U or $\\chi^2$ test).', 'tab:demo')


def macro(name, value):
    assert re.fullmatch(r'[A-Za-z]+', name), name
    MACROS[name] = value


# ── E1 component ablation ────────────────────────────────────────────────
e1 = read('E1_component_ablation.csv', index_col=0)
REF = e1.index[0]
rows = []
for v, r in e1.iterrows():
    rows.append([tex(v), ci(r.AUC, r.AUC_lo, r.AUC_hi), f'{r.Accuracy:.3f}', f'{r.Sensitivity:.3f}',
                 f'{r.Specificity:.3f}', f'{r.F1:.3f}', f'{r.Brier:.3f}',
                 '--' if v == REF else ci(r['ΔAUC'], r['ΔAUC_lo'], r['ΔAUC_hi'], sign=True),
                 '--' if v == REF else pfmt(r.p_AUC_holm)])
    if v in (REF, '− clinical block (ECG-proxy only)', 'downslope coding fixed (slope==3)',
             '− SMOTE (no resampling)', 'soft voting → stacking (LR meta-learner)'):
        rows.append('MIDRULE')
write_table('tab_e1.tex', ['Variant', 'AUC [95\\% CI]', 'Acc.', 'Sens.', 'Spec.', 'F1', 'Brier',
                           '$\\Delta$AUC [95\\% CI]', '$p_{\\mathrm{Holm}}$'],
            rows, 'lcccccccc',
            'Component ablation under the controlled protocol (5$\\times$10-fold repeated CV, 50 paired folds, '
            'fixed hyper-parameters, all preprocessing inside folds). $\\Delta$AUC is paired against the full system; '
            'CIs use the Nadeau--Bengio correction.', 'tab:e1', star=True)
f = e1.loc[REF]; c = e1.loc['− ECG-proxy block (clinical only)']
macro('fullAUC', f'{f.AUC:.3f}'); macro('fullAUCci', f'[{f.AUC_lo:.3f}, {f.AUC_hi:.3f}]')
macro('fullAcc', f'{f.Accuracy:.3f}'); macro('fullSens', f'{f.Sensitivity:.3f}'); macro('fullSpec', f'{f.Specificity:.3f}')
macro('clinAUC', f'{c.AUC:.3f}'); macro('clinAUCci', f'[{c.AUC_lo:.3f}, {c.AUC_hi:.3f}]'); macro('clinAcc', f'{c.Accuracy:.3f}')
macro('dClinAUC', ci(c['ΔAUC'], c['ΔAUC_lo'], c['ΔAUC_hi'], sign=True))
macro('dClinAcc', ci(c['ΔAccuracy'] * 100, c['ΔAccuracy_lo'] * 100, c['ΔAccuracy_hi'] * 100, d=1, sign=True))
for key, name in (('− clinical block (ECG-proxy only)', 'ProxyOnly'), ('− SMOTE (no resampling)', 'NoSmote'),
                  ('− soft voting → XGBoost only', 'XgbOnly'), ('− soft voting → SVM only', 'SvmOnly'),
                  ('soft → hard (majority) voting', 'Hard'), ('− soft voting → RF only', 'RfOnly'),
                  ('− soft voting → GBM only', 'GbmOnly'), ('soft voting → stacking (LR meta-learner)', 'Stack'),
                  ('Clinical only, no SMOTE, logistic regression', 'ClinLR'),
                  ('− soft voting → logistic regression', 'LR')):
    r = e1.loc[key]
    macro(f'd{name}AUC', ci(r['ΔAUC'], r['ΔAUC_lo'], r['ΔAUC_hi'], sign=True))
    macro(f'{name[0].lower() + name[1:]}AUCval', f'{r.AUC:.3f}')
    macro(f'p{name}', pfmt(r.p_AUC_holm))
r = e1.loc['− SMOTE (no resampling)']
macro('dNoSmoteAcc', ci(r['ΔAccuracy'] * 100, r['ΔAccuracy_lo'] * 100, r['ΔAccuracy_hi'] * 100, d=1, sign=True))
macro('noSmoteSpec', f'{r.Specificity:.3f}'); macro('noSmoteSens', f'{r.Sensitivity:.3f}')
eng = e1.loc[[i for i in e1.index if i.startswith('− ') and any(
    k in i for k in ('polynomial', 'interaction', 'log', 'ratio', 'threshold', 'HR-reserve', 'ischaemia', 'duplicated'))]]
macro('engMaxAbs', f'{eng["ΔAUC"].abs().max():.3f}')
macro('calFull', f'{f.CalSlope:.2f}')

# ── E2 controls ─────────────────────────────────────────────────────────
e2 = read('E2_controls.csv', index_col=0)
rows = [[tex(v), ci(r.AUC, r.AUC_lo, r.AUC_hi), f'{r.Accuracy:.3f}',
         '--' if v == REF else ci(r['ΔAUC'], r['ΔAUC_lo'], r['ΔAUC_hi'], sign=True),
         '--' if v == REF else pfmt(r.p_AUC_holm)] for v, r in e2.iterrows()]
write_table('tab_e2.tex', ['Feature set (soft voting, SMOTE)', 'AUC [95\\% CI]', 'Acc.', '$\\Delta$AUC vs full', '$p_{\\mathrm{Holm}}$'],
            rows, 'lcccc', 'Control experiments: is the ECG-proxy block better than other added features?', 'tab:e2', star=True)
for key, name in (('Clinical + 18 Gaussian-noise features', 'Noise'),
                  ('Clinical + 18 engineered non-ECG features', 'NonEcg'),
                  ('Clinical + row-shuffled ECG-proxy block', 'Shuffled')):
    macro(f'{name.lower()}AUC', f'{e2.loc[key].AUC:.3f}')
perm = read('E2_permutation_test.csv').iloc[0]
macro('permObs', f'{perm.observed_auc:.3f}'); macro('permNull', f'{perm.null_mean:.3f}')
macro('permNullSd', f'{perm.null_sd:.3f}'); macro('permP', f'{perm.p_value:.3f}'); macro('permN', f'{int(perm.n_perm)}')

# ── E3 resampling ───────────────────────────────────────────────────────
e3 = read('E3_resampling.csv', index_col=0)
rows = [[tex(v.replace('Full system (clinical + ECG-proxy, SMOTE, soft voting)', 'SMOTE after features (paper)')),
         f'{r.AUC:.3f}', f'{r.Sensitivity:.3f}', f'{r.Specificity:.3f}', f'{r.Brier:.3f}', f'{r.CalSlope:.2f}',
         '--' if v == REF else ci(r['ΔAUC'], r['ΔAUC_lo'], r['ΔAUC_hi'], sign=True)] for v, r in e3.iterrows()]
write_table('tab_e3.tex', ['Imbalance strategy', 'AUC', 'Sens.', 'Spec.', 'Brier', 'Cal.\\ slope', '$\\Delta$AUC [95\\% CI]'],
            rows, 'lcccccc', 'Class-imbalance strategies (full feature set, soft voting; resampling inside training folds only). '
            'Calibration slope: ideal 1; $>$1 means under-confident probabilities.', 'tab:e3', star=True)
real = read('E3_smote_realism.csv').iloc[0]
macro('smoteNonInt', tex(real['Rows with non-integer categorical codes']))
macro('smoteInconsistent', tex(real['Rows where oldpeak_sq ≠ oldpeak² (>5%)']))
macro('smoteDupGap', tex(real['Rows where duplicate oldpeak copies disagree']))
cc = read('E3_class_counts.csv')
write_table('tab_counts.tex', ['Partition', 'Train (H/D)', 'After SMOTE', 'Test (H/D)'],
            [[tex(r.Dataset), r['Train (healthy/disease)'], r['Train after SMOTE'], r['Test (healthy/disease)']]
             for _, r in cc.iterrows()], 'lccc',
            'Class counts per partition (H = healthy, D = disease). SMOTE is fitted on training rows only.', 'tab:counts')

# ── E4 protocol ─────────────────────────────────────────────────────────
e4 = read('E4_protocol_leakage.csv', index_col=0)
write_table('tab_e4.tex', ['Protocol', 'AUC [95\\% CI]', 'Accuracy [95\\% CI]', 'Sensitivity [95\\% CI]'],
            [[tex(v), r.AUC, r.Accuracy, r.Sensitivity] for v, r in e4.iterrows()], 'lccc',
            'Leakage and tuning protocol (5$\\times$10-fold CV). Leaky rows are shown only to quantify the bias.', 'tab:e4', star=True)
lk = e4.loc['Leaky (b): SMOTE before CV', 'AUC'].split()[0]
macro('leakSmoteAUC', lk)
macro('leakPreAUC', e4.loc['Leaky (a): preprocessing fitted on all rows', 'AUC'].split()[0])
macro('nestedAUC', e4.loc['XGBoost, nested tuning (inner 3-fold)', 'AUC'].split()[0])
macro('fixedXgbAUC', e4.loc['XGBoost, fixed Section-15 params', 'AUC'].split()[0])
macro('defaultXgbAUC', e4.loc['XGBoost, library defaults', 'AUC'].split()[0])
ho = read('E4_holdout_sensitivity.csv', index_col=0)
write_table('tab_holdout.tex', ['Metric', 'Mean (2.5--97.5\\%) over random 80/20 splits', 'Seed 42', 'Pct.'],
            [[tex(m), tex(r.iloc[0]), tex(r.iloc[1]), tex(r.iloc[2])] for m, r in ho.iterrows()], 'lccc',
            'Sensitivity of the 61-patient hold-out estimate to the random split (leak-free pipeline). '
            '``Pct.'' is the percentile of the seed-42 split used in the v4 paper.', 'tab:holdout', star=True)
macro('hoN', re.search(r'Over (\d+)', ho.columns[0]).group(1))
for m, name in (('Fused accuracy', 'FuAcc'), ('Clinical accuracy', 'ClAcc'), ('Fused − clinical accuracy', 'DAcc'),
                ('Fused AUC', 'FuAUC'), ('Fused − clinical AUC', 'DAUC')):
    macro(f'ho{name}', tex(ho.loc[m].iloc[0])); macro(f'ho{name}Seed', tex(ho.loc[m].iloc[1]))
    macro(f'ho{name}Pct', tex(ho.loc[m].iloc[2]))

# ── E5 robustness & subgroups ──────────────────────────────────────────
e5 = read('E5_robustness.csv', index_col=0)
rows = [[tex(v), f'{r["AUC fused"]:.3f}', f'{r["AUC clinical"]:.3f}',
         ci(r['ΔAUC'], r['ΔAUC_lo'], r['ΔAUC_hi'], sign=True), pfmt(r.p_AUC_holm),
         f'{r["Sens fused"]:.3f} / {r["Sens clinical"]:.3f}'] for v, r in e5.iterrows()]
write_table('tab_e5.tex', ['Scenario (3$\\times$10-fold CV)', 'AUC fused', 'AUC clin.', '$\\Delta$AUC [95\\% CI]',
                           '$p_{\\mathrm{Holm}}$', 'Sens. fused / clin.'], rows, 'lccccc',
            'Robustness and edge cases. Each perturbation is applied identically to the fused and the clinical-only system.',
            'tab:e5', star=True)
macro('robMinD', f'{e5["ΔAUC"].min():+.3f}'); macro('robMaxD', f'{e5["ΔAUC"].max():+.3f}')
macro('robNsig', str(int((e5.p_AUC_holm < 0.05).sum()))); macro('robN', str(len(e5)))
for key, name in (('No angiography/thallium (ca, thal unavailable)', 'NoAngio'),
                  ('No exercise test (thalach, exang, oldpeak, slope)', 'NoEx'),
                  ('Resting data only (both unavailable)', 'Rest'), ('Training label noise 20%', 'Noise'),
                  ('Training set reduced to 25%', 'Small')):
    r = e5.loc[key]
    macro(f'rob{name}Fu', f'{r["AUC fused"]:.3f}'); macro(f'rob{name}Cl', f'{r["AUC clinical"]:.3f}')
    macro(f'rob{name}D', ci(r['ΔAUC'], r['ΔAUC_lo'], r['ΔAUC_hi'], sign=True))
sg = read('E5_subgroups.csv', index_col=0)
write_table('tab_subgroups.tex', ['Subgroup', 'n', 'Dis.', 'AUC fused', 'AUC clin.', '$\\Delta$AUC [95\\% CI]'],
            [[tex(g), str(int(r.n)), str(int(r.disease)), f'{r["AUC fused"]:.3f}', f'{r["AUC clinical"]:.3f}',
              f'{r["ΔAUC"]:+.3f} {tex(r["ΔAUC 95% CI"])}'] for g, r in sg.iterrows()], 'lccccc',
            'Subgroup performance (out-of-fold predictions averaged over 5 repeats; bootstrap CIs).', 'tab:subgroups')

# ── E6 encoding & calibration ──────────────────────────────────────────
e6 = read('E6_encoding_calibration.csv', index_col=0)
write_table('tab_e6.tex', ['Variant', 'AUC', 'Brier', 'Cal.\\ slope', '$\\Delta$AUC [95\\% CI]'],
            [[tex(v), f'{r.AUC:.3f}', f'{r.Brier:.3f}', f'{r.CalSlope:.2f}',
              '--' if v == REF else ci(r['ΔAUC'], r['ΔAUC_lo'], r['ΔAUC_hi'], sign=True)] for v, r in e6.iterrows()],
            'lcccc', 'Ordinal versus one-hot encoding and probability calibration.', 'tab:e6', star=True)
for key, name in (('Isotonic-calibrated soft voting', 'Iso'), ('Platt-calibrated soft voting', 'Platt'),
                  ('One-hot cp/restecg/slope/thal (full system)', 'Onehot')):
    macro(f'cal{name}Slope', f'{e6.loc[key].CalSlope:.2f}'); macro(f'cal{name}Brier', f'{e6.loc[key].Brier:.3f}')
    macro(f'cal{name}AUC', f'{e6.loc[key].AUC:.3f}')
macro('brierFull', f'{e6.loc[REF].Brier:.3f}')

# ── E7 external ─────────────────────────────────────────────────────────
sites = read('E7_site_characteristics.csv', index_col=0)
write_table('tab_sites.tex', ['Site', 'n', 'Prev.', 'ca miss.', 'thal miss.', 'slope miss.', 'chol miss.'],
            [[tex(s), str(r.n), tex(r['disease prevalence']), tex(r['missing ca']), tex(r['missing thal']),
              tex(r['missing slope']), tex(r['missing chol'])] for s, r in sites.iterrows()], 'lcccccc',
            'Characteristics of the four UCI heart-disease sites (chol = 0 treated as missing).', 'tab:sites')
ext = read('E7_external_validation.csv', index_col=[0, 1])
rows = []
for (site, model), r in ext.iterrows():
    rows.append([tex(site), tex(model), f'{r.AUC:.3f} {r["AUC 95% CI"]}', f'{r.Sensitivity:.3f}',
                 f'{r.Specificity:.3f}', f'{r.Brier:.3f}'])
write_table('tab_e7.tex', ['Site', 'Model (trained on Cleveland)', 'AUC [95\\% CI]', 'Sens.', 'Spec.', 'Brier'],
            rows, 'llcccc', 'External validation: trained on all 303 Cleveland patients, tested unchanged on three other hospitals.',
            'tab:e7', star=True)
ed = read('E7_external_delta.csv', index_col=[0, 1])
macro('extDeltaList', '; '.join(f'{tex(s)}: {r["ΔAUC"]:+.3f} {r["95% CI"]}' for (s, cmp_), r in ed.iterrows()
                                if cmp_.startswith('Fused (full')))
for s, name in (('Hungary', 'Hun'), ('Switzerland', 'Swi'), ('VA Long Beach', 'Va')):
    macro(f'ext{name}Fu', f'{ext.loc[(s, "Fused (full system)")].AUC:.3f}')
    macro(f'ext{name}Cl', f'{ext.loc[(s, "Clinical only")].AUC:.3f}')

# ── E8 SHAP ─────────────────────────────────────────────────────────────
st = read('E8_shap_stability.csv', index_col=0)
write_table('tab_e8.tex', ['Stability measure', 'Mean (2.5--97.5\\%)'],
            [[tex(m), tex(r.Value)] for m, r in st.iterrows()], 'lc',
            'Stability of TreeSHAP attributions (XGBoost member) across 50 CV folds and 10 seeds.', 'tab:e8')
vals = [float(x.split()[0]) for x in st.Value]
macro('shapRhoFold', f'{vals[0]:.2f}'); macro('shapJac', f'{vals[1]:.2f}'); macro('shapRhoSeed', f'{vals[2]:.2f}')
macro('shapRhoPI', f'{vals[3]:.2f}')
top = read('E8_shap_top_features.csv', index_col=0)
write_table('tab_shaptop.tex', ['Feature', 'Block', 'Mean $|$SHAP$|$', 'Median rank (IQR)', 'Top-5', 'Perm.\\ imp.'],
            [[tex(fn), tex(r.block), f'{r["mean |SHAP|"]:.3f}', tex(r['median rank (IQR)']), tex(r['% folds in top-5']),
              f'{r["perm. importance (ΔAUC)"]:.4f}'] for fn, r in top.head(10).iterrows()], 'llcccc',
            'Top-10 features by mean $|$SHAP$|$ over 50 folds, with rank stability and permutation importance (AUC drop).',
            'tab:shaptop')
dup = read('E8_duplicate_attribution_split.csv', index_col=0)
macro('dupShareMean', f'{dup["proxy share of total"].mean():.0%}'.replace('%', '\\%'))

# ── E9 cross-attention ─────────────────────────────────────────────────
e9 = read('E9_cross_attention.csv', index_col=0)
ref9 = 'DL: dual branch + cross-attention (L=1, paper)'
write_table('tab_e9.tex', ['Model (3$\\times$10-fold CV)', 'AUC [95\\% CI]', 'Acc.', 'Sens.', '$\\Delta$AUC vs L=1 attention'],
            [[tex(v), ci(r.AUC, r.AUC_lo, r.AUC_hi), f'{r.Accuracy:.3f}', f'{r.Sensitivity:.3f}',
              '--' if v == ref9 else ci(r['ΔAUC'], r['ΔAUC_lo'], r['ΔAUC_hi'], sign=True)] for v, r in e9.iterrows()],
            'lcccc', 'Dual-branch networks with and without cross-attention (identical preprocessing, SMOTE and training budget).',
            'tab:e9', star=True)
macro('xattnAUC', f'{e9.loc[ref9].AUC:.3f}')
macro('concatAUC', f'{e9.loc["DL: dual branch + concatenation"].AUC:.3f}')
macro('linearAUC', f'{e9.loc["DL: dual branch + linear layer (≡ L=1 attention)"].AUC:.3f}')
macro('tokensAUC', f'{e9.loc["DL: dual branch + token-level cross-attention"].AUC:.3f}')
macro('mlpClinAUC', f'{e9.loc["DL: clinical-only MLP (Section 14)"].AUC:.3f}')
r = e9.loc['DL: dual branch + concatenation']; macro('dConcat', ci(r['ΔAUC'], r['ΔAUC_lo'], r['ΔAUC_hi'], sign=True))
r = e9.loc['DL: dual branch + token-level cross-attention']; macro('dTokens', ci(r['ΔAUC'], r['ΔAUC_lo'], r['ΔAUC_hi'], sign=True))

# ── E10 PTB-XL ──────────────────────────────────────────────────────────
if os.path.exists(os.path.join(TAB, 'E10_ptbxl_branch.csv')):
    e10 = read('E10_ptbxl_branch.csv', index_col=0)
    ref10 = e10.index[0]
    write_table('tab_e10.tex', ['ECG-branch variant (PTB-XL labels)', 'AUC [95\\% CI]', 'Acc.', '$\\Delta$AUC [95\\% CI]', '$p_{\\mathrm{Holm}}$'],
                [[tex(v.replace('[ECG] ', '')), ci(r.AUC, r.AUC_lo, r.AUC_hi), f'{r.Accuracy:.3f}',
                  '--' if v == ref10 else ci(r['ΔAUC'], r['ΔAUC_lo'], r['ΔAUC_hi'], sign=True),
                  '--' if v == ref10 else pfmt(r.p_AUC_holm)] for v, r in e10.iterrows()], 'lcccc',
                'PTB-XL ECG-branch (D2) ablation, 303 records, 5$\\times$10-fold CV.', 'tab:e10', star=True)
    macro('ecgRefAUC', f'{e10.loc[ref10].AUC:.3f}')
    for key, name in (('[ECG] 12-lead Minimal tsfresh (120 features)', 'TwelveMin'),
                      ('[ECG] Minimal tsfresh (10 features), Lead II', 'MinII'),
                      ('[ECG] Lead V1, Efficient', 'VOne'), ('[ECG] Lead I, Efficient', 'LeadI'),
                      ('[ECG] Lead V5, Efficient', 'VFive'),
                      ('[ECG] 12-lead Minimal + Lead II Efficient', 'TwelvePlus')):
        macro(f'ecg{name}AUC', f'{e10.loc[key].AUC:.3f}')
    sz = read('E10_ptbxl_sample_size.csv', index_col=0)
    write_table('tab_e10size.tex', ['Records', 'Patients', 'AUC [95\\% CI]', 'Accuracy [95\\% CI]'],
                [[str(n), str(r['Unique patients']), r['AUC [95% CI]'], r['Accuracy [95% CI]']] for n, r in sz.iterrows()],
                'cccc', 'PTB-XL Lead-II branch: effect of the number of training records (3$\\times$10-fold CV).', 'tab:e10size')
    macro('ecgSizeList', '; '.join(f'n={n}: {r["AUC [95% CI]"].split()[0]}' for n, r in sz.iterrows()))
    q = read('E10_ptbxl_signal_quality.csv', index_col=0)
    macro('ecgCleanAUC', f'{q.iloc[1].AUC:.3f}'); macro('ecgNoisyAUC', f'{q.iloc[2].AUC:.3f}')
    macro('ecgNoisyN', str(int(q.iloc[2].n)))

# ── E11 verdict ─────────────────────────────────────────────────────────
v = read('E11_verdict.csv', index_col=0)
write_table('tab_verdict.tex', ['Claim in the v4 manuscript', 'v4 value', 'Controlled estimate [95\\% CI]', 'Verdict'],
            [[tex(cl), tex(r['Paper value']), tex(r['Controlled estimate [95% CI]']), tex(r.Verdict)]
             for cl, r in v.iterrows()], 'p{4.6cm}p{2.6cm}p{5.6cm}p{3.6cm}',
            'Claims of the v4 manuscript re-tested under the controlled protocol. A claim is supported only if the 95\\% CI '
            'of the paired difference excludes zero in the claimed direction.', 'tab:verdict', star=True)
macro('nVariants', str(len(read('ALL_variants.csv'))))

with open(os.path.join(OUT, 'numbers.tex'), 'w') as fh:
    for k, val in sorted(MACROS.items()):
        fh.write(f'\\newcommand{{\\{k}}}{{{val}}}\n')
print(f'{len(MACROS)} macros, tables in {OUT}')
