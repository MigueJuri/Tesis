import pandas as pd
from typing import Dict


def load_stacked_csv(path: str, timestamp_col: str = 'timestamp', asset_col: str = 'asset', price_col: str = 'Close') -> Dict[str, pd.Series]:
    """Load a stacked CSV with an `asset` column and return dict of asset -> close Series (datetime indexed).
    Expects timestamp parseable by pandas.
    """
    df = pd.read_csv(path)
    if asset_col not in df.columns:
        raise ValueError(f"Asset column '{asset_col}' not found")
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found")

    if timestamp_col not in df.columns:
        # try common names
        if 'Date' in df.columns:
            timestamp_col = 'Date'
        else:
            raise ValueError('Timestamp column not found')
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values([asset_col, timestamp_col])
    assets = {}
    for asset, g in df.groupby(asset_col):
        s = pd.Series(g[price_col].values, index=g[timestamp_col].values)
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        assets[asset] = s
    return assets
