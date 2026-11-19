# Pilot: physiologically grounded ECG + heart-sound fusion with XAI

**Question.** Can two simultaneously recorded modalities (ECG = electrical, phonocardiogram = mechanical) be fused
through quantities that *only exist when both are present*, the electromechanical timing intervals, and can XAI
explain the predictions in physiological units?

**Data.** PhysioNet/CinC 2016 `training-a` (MIT heart-sound database): 409 recordings with **simultaneous ECG + PCG at
2 kHz** (4 have PCG only).

- Diagnoses: normal 117, mitral valve prolapse 134, innocent/benign murmur 118, aortic disease 17, miscellaneous 23.
- Hand-corrected S1/systole/S2/diastole annotations from PhysioNet.
- Patient grouping is recovered from the original record names (60 conservative groups).

Code: `experiments/fusion/extract_features.py`, `experiments/fusion/analyse.py`. Outputs: `results/fusion/`.

## Why this counts as fusion, and why it is grounded

| Cross-modal feature | Definition | Physiological meaning | Evidence base |
|---|---|---|---|
| EMAT | ECG Q → S1 centre | Electrical activation to mitral closure; LV contractility | Validated against catheterisation; EMAT ≥ 15% of RR flags LV dysfunction ([PMC9267668](https://pmc.ncbi.nlm.nih.gov/articles/PMC9267668)) |
| QS2 | ECG Q → S2 onset | Total electromechanical systole | Systolic time intervals, Weissler et al., *Circulation* 1968 |
| R→S1 | ECG R → S1 onset | Electromechanical delay | S1 follows R ([PhysioNet HSS](https://archive.physionet.org/physiotools/hss/)) |
| EMW | QS2 − QT | Electromechanical window | Mechanical systole outlasts repolarisation in normal hearts |

None of these can be computed from the ECG alone or from the PCG alone.

## Results (patient-grouped CV unless stated)

### 1. Measurement validity

In 115 normal recordings, against Weissler's physiology:

- QS2 shortens by **−1.47 ms/bpm** [−2.00, −0.95].
- The S1→S2 ejection proxy shortens by **−1.77 ms/bpm** [−2.10, −1.44].
- EMAT is flat (+0.12, n.s.).

This is the expected ordering: steep for electromechanical systole and ejection time, smallest for the
pre-ejection phase. R→S1 has a median of 7.5 ms and EMW is positive (17 ms), both as expected.

### 2. Patient leakage inflates performance by 0.10–0.20 AUC

| Model, target | Record-level CV | Patient-grouped CV | Inflation |
|---|---|---|---|
| XGBoost, pathology | 0.837 | 0.698 | +0.139 |
| XGBoost, original label | 0.794 | 0.592 | +0.202 |
| LogReg, pathology | 0.748 | 0.651 | +0.097 |

Leakage in heart-sound benchmarks has been raised before, but we found no quantification for this simultaneous
ECG+PCG subset.

### 3. Modality ablation (XGBoost, pathological vs normal/benign)

| Features | AUC [95% CI] |
|---|---|
| ECG only | 0.609 [0.518, 0.701] |
| PCG only | 0.692 [0.620, 0.765] |
| Cross-modal timing only (8 features) | 0.603 [0.504, 0.703] |
| ECG + PCG concatenation | 0.685 [0.598, 0.773] |
| ECG + PCG + cross-modal timing | 0.698 [0.582, 0.814]; Δ vs concatenation +0.013 [−0.050, +0.076] |

With logistic regression the cross-modal block alone (0.611) clearly beats ECG alone (0.522). The increment over
concatenation is **positive but not significant** with about 60 patients.

### 4. XAI

SHAP shares are PCG 42%, ECG 34%, cross-modal 24%. The directions are physiologically coherent:

- Systolic high-frequency energy (a murmur) increases predicted pathology.
- Longer R→S1 and higher %EMAT increase it.
- Aortic disease shows the longest QS2 (424 ms) and S1→S2 (360 ms) despite a higher heart rate. This is consistent with
  prolonged LV ejection in aortic valve disease.

### 5. Label issue found

The original "abnormal" class includes 118 *innocent* murmurs. We therefore used a clinically defined target
(pathological = MVP, aortic disease, miscellaneous) alongside the original one.

## Verdict

**The idea is sound and original enough to pursue.** Supported so far:

- The fusion features are valid measurements.
- They carry signal on their own.
- They give explanations in clinical units.

What the pilot cannot yet show is a **significant gain over simple concatenation**, because the only open simultaneous
ECG+PCG disease dataset has about 60–120 patients. The leakage result is solid and publishable on its own.

## Next steps

1. Deep-learning black-box fusion baseline (1D-CNN on raw ECG + PCG, cross-attention), under the same patient-grouped
   CV. This tests whether 8 interpretable cross-modal features match the black box once leakage is removed.
2. Beat-level modelling: about 27 paired beats per recording gives more statistical units, with the patient still the
   unit of inference.
3. More paired data. EPHNOGRAM (simultaneous ECG+PCG, 24 healthy adults, rest/exercise) can validate the interval
   physiology across heart rates. EchoNext (restricted) or MIMIC-IV (credentialed) for a larger follow-up.
4. Before publication, verify the Weissler regression equations against the 1968 paper; only qualitative agreement is
   claimed here.
