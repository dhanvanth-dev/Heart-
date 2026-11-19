"""Per-recording features from simultaneous ECG + PCG (PhysioNet/CinC 2016, training-a).

Three feature families, kept strictly separate so modality ablations are clean:
  ECG-only   : rhythm and ECG intervals (RR, HRV, QRS, QT/QTc, amplitudes)
  PCG-only   : heart-sound state durations and acoustic energy/spectrum per state
  CROSS-MODAL: systolic time intervals that need BOTH clocks, i.e. an ECG fiducial and a
               heart-sound event in the same beat:
                 EMAT  = Q -> S1 (centre)          electromechanical activation time
                 RS1   = R -> S1 (onset)           robust variant (no Q delineation)
                 QS2   = Q -> S2 (onset)           total electromechanical systole
                 EMW   = QS2 - QT                  electromechanical window
                 %EMAT = EMAT / RR                 heart-rate-normalised EMAT
               plus their beat-to-beat variability (SD).

S1/S2 timing comes from the PhysioNet hand-corrected state annotations (S1, systole, S2,
diastole); noisy segments marked '(N'...'N)' are excluded. ECG fiducials come from NeuroKit2
(Pan-Tompkins-style R peaks, DWT delineation for Q peaks and T offsets) at 500 Hz.

Run from the repository root: python experiments/fusion/extract_features.py
"""
import os
import warnings

import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal as ss
import wfdb

warnings.filterwarnings('ignore')
ROOT = 'data/physionet2016'
FS = 2000
FS_ECG = 500
OUT = 'results/fusion'
os.makedirs(OUT, exist_ok=True)


def states(rec):
    """List of (start_sample, state) with noisy segments removed."""
    m = scipy.io.loadmat(f'{ROOT}/states/{rec}_StateAns.mat')['state_ans']
    out, noisy = [], False
    for start, name in m:
        s = int(np.squeeze(start))
        n = str(np.ravel(np.ravel(name)[0])[0]).strip()      # names are stored as nested 1x1 cell arrays
        if n.startswith('(N'):
            noisy = True; continue
        if n.endswith('N)'):
            noisy = False; continue
        if not noisy:
            out.append((s, n))
    return out


def cycles(st):
    """Complete cardiac cycles: S1 start, systole start, S2 start, diastole start, next S1 start."""
    cyc = []
    for i in range(len(st) - 4):
        names = [st[i + k][1] for k in range(5)]
        if names == ['S1', 'systole', 'S2', 'diastole', 'S1']:
            cyc.append([st[i + k][0] for k in range(5)])
    return np.array(cyc)


def band_energy(x, fs, lo, hi):
    f, p = ss.welch(x, fs=fs, nperseg=min(256, len(x)))
    sel = (f >= lo) & (f < hi)
    return float(np.trapezoid(p[sel], f[sel])) if sel.any() else np.nan


def pcg_features(pcg, cyc):
    pcg = ss.sosfiltfilt(ss.butter(4, [25, 400], btype='band', fs=FS, output='sos'), pcg)
    d = {k: [] for k in ['s1', 'sys', 's2', 'dia', 'cyc']}
    e = {k: [] for k in ['s1', 'sys', 's2', 'dia']}
    hf = {k: [] for k in ['sys', 'dia']}
    cent = []
    for c in cyc:
        seg = {'s1': (c[0], c[1]), 'sys': (c[1], c[2]), 's2': (c[2], c[3]), 'dia': (c[3], c[4])}
        for k, (a, b) in seg.items():
            d[k].append((b - a) / FS * 1000)
            e[k].append(np.mean(pcg[a:b] ** 2) if b > a else np.nan)
        d['cyc'].append((c[4] - c[0]) / FS * 1000)
        for k in ('sys', 'dia'):
            a, b = seg[k]
            if b - a > 64:
                hf[k].append(band_energy(pcg[a:b], FS, 150, 400) / (band_energy(pcg[a:b], FS, 25, 400) + 1e-12))
        a, b = seg['sys']
        if b - a > 64:
            f, p = ss.welch(pcg[a:b], fs=FS, nperseg=min(256, b - a)); cent.append(np.sum(f * p) / (np.sum(p) + 1e-12))
    m = lambda v: float(np.nanmedian(v)) if len(v) else np.nan
    s1e, s2e = m(e['s1']), m(e['s2'])
    return {'pcg_s1_dur': m(d['s1']), 'pcg_s2_dur': m(d['s2']), 'pcg_sys_dur': m(d['sys']), 'pcg_dia_dur': m(d['dia']),
            'pcg_s1s2_interval': m(np.array(d['s1']) + np.array(d['sys'])),          # S1 onset -> S2 onset
            'pcg_sys_frac': m(np.array(d['sys']) / np.array(d['cyc'])),
            'pcg_s2_s1_energy': s2e / (s1e + 1e-12),
            'pcg_sys_s1_energy': m(e['sys']) / (s1e + 1e-12),                          # systolic murmur energy
            'pcg_dia_s1_energy': m(e['dia']) / (s1e + 1e-12),                          # diastolic murmur energy
            'pcg_sys_hf_ratio': m(hf['sys']), 'pcg_dia_hf_ratio': m(hf['dia']),
            'pcg_sys_centroid': m(cent), 'pcg_cycle_sd': float(np.std(d['cyc'])) if len(d['cyc']) > 1 else np.nan}


def ecg_fiducials(ecg):
    x = ss.resample_poly(ecg, 1, FS // FS_ECG)
    x = nk.ecg_clean(x, sampling_rate=FS_ECG)
    _, info = nk.ecg_peaks(x, sampling_rate=FS_ECG)
    r = np.asarray(info['ECG_R_Peaks'])
    q, t_off, s_pk, t_pk = (np.full(len(r), np.nan) for _ in range(4))
    try:
        _, w = nk.ecg_delineate(x, r, sampling_rate=FS_ECG, method='dwt')
        q = np.asarray(w.get('ECG_Q_Peaks', q), float); t_off = np.asarray(w.get('ECG_T_Offsets', t_off), float)
        s_pk = np.asarray(w.get('ECG_S_Peaks', s_pk), float); t_pk = np.asarray(w.get('ECG_T_Peaks', t_pk), float)
    except Exception:
        pass
    scale = FS // FS_ECG
    return x, r * scale, q * scale, s_pk * scale, t_off * scale, t_pk * scale


def record_features(rec):
    sig, _ = wfdb.rdsamp(f'{ROOT}/training-a/{rec}')
    pcg = sig[:, 0]
    cyc = cycles(states(rec))
    feats = {'record': rec, 'n_cycles': len(cyc), 'has_ecg': int(sig.shape[1] > 1)}
    feats.update(pcg_features(pcg, cyc) if len(cyc) >= 3 else {})
    if sig.shape[1] < 2:                 # 4 recordings have PCG only: ECG and cross-modal features stay missing
        return feats
    x, r, q, s_pk, t_off, t_pk = ecg_fiducials(sig[:, 1])
    if len(r) >= 3:
        rr = np.diff(r) / FS * 1000
        qrs = (s_pk - q) / FS * 1000; qt = (t_off - q) / FS * 1000
        rr_for_qt = np.r_[rr, np.nan][:len(qt)]
        feats.update({'ecg_hr': 60000 / np.median(rr), 'ecg_sdnn': np.std(rr), 'ecg_rmssd': np.sqrt(np.mean(np.diff(rr) ** 2)),
                      'ecg_qrs_dur': np.nanmedian(qrs), 'ecg_qt': np.nanmedian(qt),
                      'ecg_qtc': np.nanmedian(qt / np.sqrt(rr_for_qt / 1000)),
                      'ecg_r_amp': np.median(x[(r // (FS // FS_ECG)).astype(int)]),
                      'ecg_t_amp': np.nanmedian([x[int(i)] for i in t_pk / (FS // FS_ECG) if np.isfinite(i)]) if np.isfinite(t_pk).any() else np.nan})
    # cross-modal: pair each heart-sound cycle with the R peak that precedes its S1 by <= 250 ms
    emat, rs1, qs2, emw, pemat = [], [], [], [], []
    for c in cyc:
        s1_on, s1_c, s2_on = c[0], (c[0] + c[1]) / 2, c[2]
        j = np.where((r <= s1_on + 0.02 * FS) & (r >= s1_on - 0.25 * FS))[0]
        if not len(j):
            continue
        j = j[-1]
        rs1.append((s1_on - r[j]) / FS * 1000)
        if j + 1 < len(r):
            rr_j = (r[j + 1] - r[j]) / FS * 1000
        else:
            rr_j = np.nan
        if np.isfinite(q[j]):
            emat.append((s1_c - q[j]) / FS * 1000); qs2.append((s2_on - q[j]) / FS * 1000)
            pemat.append(emat[-1] / rr_j * 100 if np.isfinite(rr_j) else np.nan)
            if np.isfinite(t_off[j]):
                emw.append(qs2[-1] - (t_off[j] - q[j]) / FS * 1000)
    md = lambda v: float(np.nanmedian(v)) if len(v) else np.nan
    sd = lambda v: float(np.nanstd(v)) if len(v) > 2 else np.nan
    feats.update({'x_emat': md(emat), 'x_rs1': md(rs1), 'x_qs2': md(qs2), 'x_emw': md(emw), 'x_pct_emat': md(pemat),
                  'x_emat_sd': sd(emat), 'x_qs2_sd': sd(qs2), 'x_rs1_sd': sd(rs1), 'x_n_paired_beats': len(rs1)})
    return feats


if __name__ == '__main__':
    from joblib import Parallel, delayed
    recs = [f'a{i:04d}' for i in range(1, 410)]
    rows = Parallel(n_jobs=-1)(delayed(record_features)(r) for r in recs)
    F = pd.DataFrame(rows).set_index('record')
    ap = pd.read_csv(f'{ROOT}/Online_Appendix_training_set.csv')
    ap = ap[ap['Database'] == 'training-a'].set_index('Challenge record name')
    F['diagnosis'] = ap['Diagnosis'].str.strip()
    F['label_original'] = (ap['Class (-1=normal 1=abnormal)'] == 1).astype(int)
    F['label_pathological'] = F['diagnosis'].isin(['MVP', 'AD', 'MPC']).astype(int)
    F['group_c'] = ap['Original record name'].str.extract(r'^(C\d+)S')[0]
    F['site'] = ap['Transducer site on body'].str.strip()
    F.to_csv(f'{OUT}/features_training_a.csv')
    print(F.shape); print(F.filter(like='x_').describe().T.round(1))
