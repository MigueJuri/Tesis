import pandas as pd
import numpy as np
import tempfile
import os
from train_rf import orchestrate


def make_synthetic_asset(name, n=300, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range('2020-01-01', periods=n, freq='D')
    price = 100 + np.cumsum(rng.normal(0, 1, size=n))
    df = pd.DataFrame({'timestamp': t, 'asset': name, 'Close': price})
    return df


def main():
    a = make_synthetic_asset('A', seed=1)
    b = make_synthetic_asset('B', seed=2)
    c = make_synthetic_asset('C', seed=3)
    df = pd.concat([a, b, c], ignore_index=True)
    tmp = os.path.join(tempfile.gettempdir(), 'synthetic_multi_asset.csv')
    df.to_csv(tmp, index=False)
    print('Wrote synthetic data to', tmp)
    orchestrate(tmp, out_dir='models_test')

if __name__ == '__main__':
    main()
