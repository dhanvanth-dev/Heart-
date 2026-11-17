# MultiCardioFusion

Explainable ECG-proxy-enhanced clinical prediction of heart disease (UCI Cleveland),
with PTB-XL ECG benchmarking, and an extended, statistically controlled ablation study.

- Progress, decisions and findings: [`WORKLOG.md`](WORKLOG.md)
- Original files as received: [`original/`](original/)
- Extended ablation code: [`experiments/`](experiments/), results: [`results/ablation/`](results/ablation/)

## Reproduce the extended ablation

```bash
pip install -r requirements.txt
python experiments/run_ablation.py          # writes results/ablation/*
```

The data in `data/` comes from the UCI Machine Learning Repository (Heart Disease, id 45)
and PhysioNet (PTB-XL v1.0.3).
