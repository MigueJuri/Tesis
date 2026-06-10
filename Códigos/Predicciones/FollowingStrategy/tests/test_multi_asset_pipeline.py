import pandas as pd
import numpy as np
import tempfile
import os
from rf_helpers import get_crossover_events, build_features
from train_rf import orchestrate


def make_synthetic_asset(name, n=200, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range('2020-01-01', periods=n, freq='D')
    price = 100 + np.cumsum(rng.normal(0, 1, size=n))
    df = pd.DataFrame({'timestamp': t, 'asset': name, 'Close': price})
    return df


def test_basic_pipeline():
    a = make_synthetic_asset('A', n=300, seed=1)
    b = make_synthetic_asset('B', n=300, seed=2)
    c = make_synthetic_asset('C', n=300, seed=3)
    df = pd.concat([a, b, c], ignore_index=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    tmp.close()
    try:
        df.to_csv(tmp.name, index=False)
        # load and run small pieces
        from data_loader import load_stacked_csv
        assets = load_stacked_csv(tmp.name)
        for asset, close in assets.items():
            events = get_crossover_events(close)
            # features should not crash
            if not events.empty:
                _ = build_features(close, events)

        out_dir = os.path.join(tempfile.gettempdir(), 'rf_models_test')
        artifact = orchestrate(tmp.name, out_dir=out_dir, test_asset_fraction=0.34)
        model_path = os.path.join(out_dir, 'rf_pooled.joblib')
        report_path = os.path.join(out_dir, 'report.json')
        assert os.path.exists(model_path), 'Model artifact was not saved'
        assert os.path.exists(report_path), 'JSON report was not saved'
        assert 'metrics' in artifact and 'portfolio_summary' in artifact
        assert artifact['metrics']['n_train'] > 0
        assert artifact['metrics']['n_test'] > 0
        print('synthetic pipeline OK')
    finally:
        os.unlink(tmp.name)


if __name__ == '__main__':
    test_basic_pipeline()
