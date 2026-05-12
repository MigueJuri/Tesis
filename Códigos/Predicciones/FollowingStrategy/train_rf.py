import argparse
import json
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from rf_helpers import (
    build_features,
    build_position,
    compute_sample_weights,
    get_crossover_events,
    get_labeled_events,
    summarize_strategy,
)
from data_loader import load_stacked_csv


def orchestrate(data_path: str, out_dir: str = 'models', test_asset_fraction: float = 0.3):
    assets = load_stacked_csv(data_path)
    asset_list = sorted(assets.keys())
    # build events, labels, features per asset
    all_rows = []
    for asset in asset_list:
        close = assets[asset]
        events = get_crossover_events(close)
        labeled = get_labeled_events(close, events)
        if labeled.empty:
            continue
        X = build_features(close, labeled)
        y = labeled['label'].reindex(X.index).fillna(0).astype(int)
        w = compute_sample_weights(labeled)
        # align weights
        w = w.reindex(X.index).fillna(1.0)
        X['asset'] = asset
        X['event_time'] = X.index
        # combine into single DataFrame to preserve row alignment
        combined = X.copy()
        combined['label'] = y.values
        combined['weight'] = w.values
        all_rows.append(combined)
    if len(all_rows) == 0:
        raise RuntimeError('No labeled events found in any asset')
    # concat with ignore_index to avoid duplicate timestamp index issues
    df_all = pd.concat(all_rows, ignore_index=True)
    # encode asset as categorical numeric feature for pooled model
    df_all['asset'] = df_all['asset'].astype('category')
    asset_mapping = {str(name): int(i) for i, name in enumerate(df_all['asset'].cat.categories)}
    # create asset_code column
    df_all['asset_code'] = df_all['asset'].cat.codes.astype(int)

    # split assets into train/test by asset
    assets_unique = df_all['asset'].cat.categories.tolist()
    n_train = max(1, int(len(assets_unique) * (1 - test_asset_fraction)))
    train_assets = set(assets_unique[:n_train])
    test_assets = set(assets_unique[n_train:])
    train_mask = df_all['asset'].isin(train_assets)
    test_mask = df_all['asset'].isin(test_assets)
    train_df = df_all.loc[train_mask].copy()
    test_df = df_all.loc[test_mask].copy()

    meta_cols = ['asset', 'event_time', 'label', 'weight']
    feature_cols = [c for c in train_df.columns if c not in meta_cols]

    X_train = train_df[feature_cols].fillna(0)
    X_test = test_df[feature_cols].fillna(0)
    y_train = train_df['label'].astype(int)
    y_test = test_df['label'].astype(int)
    w_train = train_df['weight'].astype(float)

    if X_train.empty:
        raise RuntimeError('Training set is empty after asset split; reduce test_asset_fraction.')
    if X_test.empty:
        raise RuntimeError('Test set is empty after asset split; increase test_asset_fraction or number of assets.')

    # scale
    scaler = StandardScaler()
    X_train_values = scaler.fit_transform(X_train.fillna(0).values)
    X_test_values = scaler.transform(X_test.fillna(0).values)
    # train pooled RF
    rf = RandomForestClassifier(n_estimators=200, max_depth=3, class_weight='balanced', random_state=42)
    rf.fit(X_train_values, y_train.values, sample_weight=w_train.values)

    y_pred = rf.predict(X_test_values)
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'f1_macro': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'train_assets': sorted([str(a) for a in train_assets]),
        'test_assets': sorted([str(a) for a in test_assets]),
    }

    # Backtest on test assets using predicted labels as event signals.
    test_df = test_df.copy()
    test_df['pred'] = y_pred
    per_asset_summary = {}
    per_asset_returns = []
    for asset in sorted(test_df['asset'].astype(str).unique()):
        asset_rows = test_df[test_df['asset'].astype(str) == asset].copy()
        if asset_rows.empty:
            continue
        signal_series = pd.Series(asset_rows['pred'].values, index=pd.to_datetime(asset_rows['event_time']))
        signal_series = signal_series.groupby(level=0).last().sort_index()
        close = assets[asset]
        pos = build_position(close, signal_series)
        summary = summarize_strategy(asset, pos, close)
        per_asset_summary[asset] = {
            'total_return': summary['total_return'],
            'max_drawdown': summary['max_drawdown'],
            'sharpe': summary['sharpe'],
        }
        per_asset_returns.append(summary['returns'].rename(asset))

    if per_asset_returns:
        portfolio_returns = pd.concat(per_asset_returns, axis=1).fillna(0).mean(axis=1)
        portfolio_equity = (1 + portfolio_returns).cumprod()
        portfolio_summary = {
            'total_return': float(portfolio_equity.iloc[-1] - 1),
            'max_drawdown': float((portfolio_equity.cummax() - portfolio_equity).max()),
            'sharpe': float(
                (portfolio_returns.mean() / portfolio_returns.std()) * (252 ** 0.5)
            ) if float(portfolio_returns.std()) > 0 else 0.0,
        }
    else:
        portfolio_summary = {'total_return': 0.0, 'max_drawdown': 0.0, 'sharpe': 0.0}

    os.makedirs(out_dir, exist_ok=True)
    artifact = {
        'model': rf,
        'scaler': scaler,
        'feature_columns': feature_cols,
        'asset_mapping': asset_mapping,
        'metrics': metrics,
        'per_asset_summary': per_asset_summary,
        'portfolio_summary': portfolio_summary,
    }
    joblib.dump(artifact, os.path.join(out_dir, 'rf_pooled.joblib'))
    with open(os.path.join(out_dir, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump({'metrics': metrics, 'per_asset_summary': per_asset_summary, 'portfolio_summary': portfolio_summary}, f, indent=2)

    print('Trained pooled RF model saved to', out_dir)
    print('Metrics:', metrics)
    print('Portfolio summary:', portfolio_summary)
    return artifact


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Path to stacked CSV with asset,timestamp,Close')
    p.add_argument('--out', default='models', help='Output directory for model')
    args = p.parse_args()
    orchestrate(args.data, args.out)
