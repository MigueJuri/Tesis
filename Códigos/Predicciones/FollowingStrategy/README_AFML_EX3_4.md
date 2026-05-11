# AFML Exercise 3.4 — Trend Following + RF Meta-Labeling

This script implements Exercise 3.4 from *Advances in Financial Machine Learning*:

1. Build a trend-following side model from EWMA crossover (`side ∈ {-1, +1}`).
2. Derive meta-labels with:
   - `ptSl = [1, 2]`
   - `t1` vertical barrier with `numDays = 1`
   - `trgt` from daily volatility (Snippet 3.1 style).
3. Train a Random Forest to decide **trade or not trade** (`meta_label ∈ {0, 1}`).

## File

- `/home/runner/work/Tesis/Tesis/Códigos/Predicciones/FollowingStrategy/afml_exercise_3_4_rf_trend_following.py`

## Run

```bash
python /home/runner/work/Tesis/Tesis/Códigos/Predicciones/FollowingStrategy/afml_exercise_3_4_rf_trend_following.py
```

Optional parameters:

```bash
python /home/runner/work/Tesis/Tesis/Códigos/Predicciones/FollowingStrategy/afml_exercise_3_4_rf_trend_following.py \
  --data-path /home/runner/work/Tesis/Tesis/Códigos/Data/sp500_data_only_1993-01-29_to_2026-01-02.csv \
  --pt 1 --sl 2 --vertical-days 1
```

## Outputs

By default, outputs are written to:

- `/home/runner/work/Tesis/Tesis/Códigos/Predicciones/FollowingStrategy/outputs_ex3_4/afml_ex3_4_labeled_events.csv`
- `/home/runner/work/Tesis/Tesis/Códigos/Predicciones/FollowingStrategy/outputs_ex3_4/afml_ex3_4_test_predictions.csv`

