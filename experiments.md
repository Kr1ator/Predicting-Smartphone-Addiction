# Experiment Log

## Current Best

- Model: CatBoost
- 5-Fold CV: 0.964007 ± 0.000473
- Public LB: 0.96538
- Iterations: 8223
- Learning rate: 0.05
- Depth: 6
- Feature engineering: None
- ID: Removed

## Experiments

| ID | Model | Validation | CV / Local Score | Public LB | Result | Notes |
|---|---|---|---:|---:|---|---|
| v1 | CatBoost | Holdout | 0.950917 | 0.95227 | Baseline | 500 iterations, no feature engineering |
| cv01 | CatBoost | 5-Fold | 0.951449 ± 0.000527 | - | Baseline CV | 500 iterations |
| exp01 | CatBoost | 5-Fold | 0.962744 ± 0.000459 | - | Improved | Max 3000 iterations; nearly every fold hit the limit |
| exp01-probe | CatBoost | Single Fold | 0.963351 | - | Probe | Best iteration = 8044 |
| v2 | CatBoost | 5-Fold | **0.964007 ± 0.000473** | **0.96538** | **Keep** | Mean best iteration = 8223; final model trained for 8223 iterations |