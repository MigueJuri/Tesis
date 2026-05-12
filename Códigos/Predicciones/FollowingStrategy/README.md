# Multi-Asset RF Trade Filter

This implementation refactors the original notebook flow into reusable Python modules for training a pooled Random Forest model on crossing events across multiple assets.

## What It Supports

- Input format: stacked CSV with at least these columns:
  - `asset`
  - `timestamp` (or `Date`)
  - `Close`
- Per-asset processing for:
  - EWMA crossing events
  - Triple-barrier labels
  - Feature engineering
  - Overlap-based sample weights
- Pooled RF training with `asset` encoded as `asset_code`
- Asset-stratified train/test split
- Train-only scaling
- Per-asset and aggregated portfolio backtest summary

## Files

- `data_loader.py`: reads stacked multi-asset CSV.
- `rf_helpers.py`: event extraction, labeling, features, sample weights, position/backtest helpers.
- `train_rf.py`: full training + evaluation + artifact/report save.
- `rf_trade_filter_multi_asset.py`: multi-asset EWMA trend-following + RF meta-labeling with OHLCV inputs.
- `run_train_test.py`: tiny synthetic runner.
- `tests/test_multi_asset_pipeline.py`: synthetic integration test script.

## Install

```powershell
pip install -r "g:/Mi unidad/2026/Tesis-1/Códigos/Predicciones/FollowingStrategy/requirements.txt"
```

## Run Training

```powershell
$env:PYTHONPATH='g:/Mi unidad/2026/Tesis-1/Códigos/Predicciones/FollowingStrategy'
python "g:/Mi unidad/2026/Tesis-1/Códigos/Predicciones/FollowingStrategy/train_rf.py" --data "PATH_TO_STACKED_CSV" --out "g:/Mi unidad/2026/Tesis-1/Códigos/Predicciones/FollowingStrategy/models"
```

## Run Multi-Asset OHLCV Training

- Input: CSV with columns `asset`, `timestamp` (or `Date`), `Open`, `High`, `Low`, `Close`, `Volume`, or a directory of per-asset OHLCV CSV files.

```powershell
$env:PYTHONPATH='g:/Mi unidad/2026/Tesis-1/Códigos/Predicciones/FollowingStrategy'
python "g:/Mi unidad/2026/Tesis-1/Códigos/Predicciones/FollowingStrategy/rf_trade_filter_multi_asset.py" --data-path "PATH_TO_OHLCV_CSV_OR_DIR"
```

Outputs in `--out`:
- `rf_pooled.joblib`
- `report.json`

## Run Synthetic End-to-End Check

```powershell
$env:PYTHONPATH='g:/Mi unidad/2026/Tesis-1/Códigos/Predicciones/FollowingStrategy'
python "g:/Mi unidad/2026/Tesis-1/Códigos/Predicciones/FollowingStrategy/run_train_test.py"
python "g:/Mi unidad/2026/Tesis-1/Códigos/Predicciones/FollowingStrategy/tests/test_multi_asset_pipeline.py"
```

## Notes

- Current split is by asset to reduce leakage across assets.
- If you later want strict time-based CV, add a walk-forward splitter over event timestamps per asset.
