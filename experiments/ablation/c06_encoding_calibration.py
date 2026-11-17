# ══════════════════════════════════════════════════════════════════════════
# 17B-6. ORDINAL ENCODING & PROBABILITY CALIBRATION (reviewer rows 8d, 13)
#  • cp, restecg, slope, thal are nominal/ordinal codes but are standardised as
#    if continuous; compare with one-hot encoding.
#  • SMOTE trains on a 50/50 class mix while the real prevalence is 45.9%, which
#    can distort probabilities. Compare Brier score and calibration slope
#    (ideal = 1) with and without SMOTE and with Platt / isotonic calibration.
# ══════════════════════════════════════════════════════════════════════════
print("── 17B-6 Encoding and calibration ───────────────────────────────────")
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

ABL6 = {
    REF: ABL1[REF],
    'One-hot cp/restecg/slope/thal (full system)':  make_pipe(FEAT_FULL(onehot=True)),
    'One-hot cp/restecg/slope/thal (clinical only)': make_pipe(FEAT_CLINICAL(onehot=True)),
    'Clinical only': ABL1['− ECG-proxy block (clinical only)'],
    'Platt-calibrated soft voting':    make_pipe(FEAT_FULL(), clf=CalibratedClassifierCV(
                                                 make_clf('voting'), method='sigmoid', cv=5)),
    'Isotonic-calibrated soft voting': make_pipe(FEAT_FULL(), clf=CalibratedClassifierCV(
                                                 make_clf('voting'), method='isotonic', cv=5)),
    '− SMOTE (no resampling)': ABL1['− SMOTE (no resampling)'],
}
for name, pipe in ABL6.items():
    evaluate(name, pipe)
abl6 = summarize(list(ABL6), REF, title='E6_encoding_calibration')
display(fmt_table(abl6))

fig, ax = plt.subplots(figsize=(6.2, 5.5))
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='perfect calibration')
for name, c in [(REF, '#2980b9'), ('− SMOTE (no resampling)', '#27ae60'),
                ('Isotonic-calibrated soft voting', '#8e44ad'), ('Clinical only', '#e67e22')]:
    o = RESULTS[name]['oof'][0]
    fr, mp = calibration_curve(y_cl, o, n_bins=8, strategy='quantile')
    ax.plot(mp, fr, 'o-', color=c, label=f"{name[:38]} (Brier {brier_score_loss(y_cl, o):.3f})")
ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed disease fraction')
ax.set_title('Figure A5: Reliability (out-of-fold, repeat 1)', fontweight='bold'); ax.legend(fontsize=7.5)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/figA5_calibration.png', dpi=200, bbox_inches='tight'); plt.show()
