# ══════════════════════════════════════════════════════════════════════════
# 17B-1. COMPONENT ABLATION (reviewer row 16)
# Remove one component at a time from the full system under the identical
# protocol. Full system = 13 clinical + 23 ECG-proxy features, SMOTE (training
# folds only), soft-voting over tuned XGB + SVM + GBM + RF.
# ══════════════════════════════════════════════════════════════════════════
print("── 17B-1 Component ablation ─────────────────────────────────────────")
ABL1 = {
    'Full system (clinical + ECG-proxy, SMOTE, soft voting)': make_pipe(FEAT_FULL()),
    # data / feature components
    '− ECG-proxy block (clinical only)':                     make_pipe(FEAT_CLINICAL()),
    '− clinical block (ECG-proxy only)':                     make_pipe(FEAT_FULL(keep_clinical=False)),
    '− duplicated raw proxy columns':                         make_pipe(FEAT_FULL(drop_dup_raw=True)),
    '− polynomial terms':        make_pipe(FEAT_FULL(groups=tuple(g for g in ALL_GROUPS if g != 'polynomial'))),
    '− interaction terms':       make_pipe(FEAT_FULL(groups=tuple(g for g in ALL_GROUPS if g != 'interaction'))),
    '− log transforms':          make_pipe(FEAT_FULL(groups=tuple(g for g in ALL_GROUPS if g != 'log'))),
    '− ratio features':          make_pipe(FEAT_FULL(groups=tuple(g for g in ALL_GROUPS if g != 'ratio'))),
    '− threshold flags':         make_pipe(FEAT_FULL(groups=tuple(g for g in ALL_GROUPS if g != 'threshold'))),
    '− HR-reserve features':     make_pipe(FEAT_FULL(groups=tuple(g for g in ALL_GROUPS if g != 'hr_reserve'))),
    '− ischaemia composite':     make_pipe(FEAT_FULL(groups=tuple(g for g in ALL_GROUPS if g != 'composite'))),
    'downslope coding fixed (slope==3)':                       make_pipe(FEAT_FULL(fix_downslope=True)),
    # imbalance handling
    '− SMOTE (no resampling)':                                 make_pipe(FEAT_FULL(), sampler=None),
    # ensemble / classifier
    '− soft voting → XGBoost only':                            make_pipe(FEAT_FULL(), clf='xgb'),
    '− soft voting → SVM only':                                make_pipe(FEAT_FULL(), clf='svm'),
    '− soft voting → GBM only':                                make_pipe(FEAT_FULL(), clf='gbm'),
    '− soft voting → RF only':                                 make_pipe(FEAT_FULL(), clf='rf'),
    '− soft voting → logistic regression':                     make_pipe(FEAT_FULL(), clf='lr'),
    'soft → hard (majority) voting':                           make_pipe(FEAT_FULL(), clf='hard_voting'),
    'soft voting → stacking (LR meta-learner)':                make_pipe(FEAT_FULL(), clf='stacking'),
    # minimal systems
    'Clinical only, no SMOTE, logistic regression':            make_pipe(FEAT_CLINICAL(), sampler=None, clf='lr'),
}
for name, pipe in ABL1.items():
    evaluate(name, pipe)

REF = 'Full system (clinical + ECG-proxy, SMOTE, soft voting)'
abl1 = summarize(list(ABL1), REF, title='E1_component_ablation')
display(fmt_table(abl1))
forest_plot(abl1.drop(REF), 'Figure A1: Component ablation — ΔAUC vs full system',
            'figA1_component_ablation.png', ref_label='full system')
