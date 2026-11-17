# Reproducibility check

**Question:** do the numbers in the paper come from the final code, or from a stale result cache?

**Method:** all 94 configurations (cells c00–c11) were re-run from an **empty result cache** into
`results/ablation_rerun/`, using the same final code (`python experiments/run_ablation.py` with
`MCF_ABL_DIR=results/ablation_rerun`). Only the tsfresh feature matrices were reused; their extraction is
deterministic. Every table was then compared cell by cell with `results/ablation/tables/`.

**Result:** all **24 tables are identical**. The maximum absolute numeric difference is **0.0**, and no text cell
differs. This includes the TensorFlow networks (17B-9) and the PTB-XL branch (17B-10).

| Table | max abs numeric diff | differing text cells |
|---|---|---|
| E1–E11, ALL_variants (24 files) | 0.0 | 0 |

The paper's numbers are generated from these tables by `paper/make_tables.py`, so they are reproducible from the
released code.

**Notebook:** `notebook/MultiCardioFusion_IEEE_2_extended_ablation.ipynb` was rebuilt by `tools/build_notebook.py`.
That script:
1. applies the 40 code and text corrections in `tools/fix_original_notebook.py`;
2. inserts Section 17B;
3. executes the whole notebook from top to bottom with nbclient.

All 40 code cells ran without error. PTB-XL is downloaded from PhysioNet (v1.0.3). The v4 run used the Kaggle copy
(21,837 records), so the PTB-XL sample and the ECG-only numbers in the notebook differ slightly from v4.

Environment used: Python 3.13, scikit-learn 1.9.1, imbalanced-learn 0.14.2, XGBoost 3.4.1, SHAP 0.54.1,
TensorFlow 2.21, tsfresh, pandas 3.0.5.
