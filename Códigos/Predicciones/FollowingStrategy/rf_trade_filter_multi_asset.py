from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler


@dataclass
class Config:
    fast_span: int = 5
    slow_span: int = 20
    burn_in: int = 20
    daily_vol_span: int = 100
    min_ret: float = 0.0
    train_frac: float = 0.7
    test_asset_fraction: float = 0.3
    random_state: int = 42
    pt_sl: tuple[float, float] = (1.0, 2.0)
    vertical_days: int = 1
    feature_range: tuple[float, float] = (0.0, 1.0)


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not df.empty and str(df.iloc[0, 0]).strip().lower() == "ticker":
        df = df.iloc[2:].copy()

    timestamp_col = None
    for col in df.columns:
        if str(col).strip().lower() in {"timestamp", "date"}:
            timestamp_col = col
            break
    if timestamp_col is None:
        timestamp_col = df.columns[0]

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col])

    def find_col(target: str) -> str | None:
        target_lower = target.lower()
        for col in df.columns:
            if str(col).strip().lower() == target_lower:
                return col
        return None

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    column_map = {name: find_col(name) for name in required_cols}
    missing = [name for name, col in column_map.items() if col is None]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {', '.join(missing)}")

    out = df[
        [
            timestamp_col,
            column_map["Open"],
            column_map["High"],
            column_map["Low"],
            column_map["Close"],
            column_map["Volume"],
        ]
    ].copy()
    out.columns = ["timestamp", "Open", "High", "Low", "Close", "Volume"]
    out = out.set_index("timestamp").sort_index()
    out = out.apply(pd.to_numeric, errors="coerce")
    return out.dropna(subset=["Close"])


def load_ohlcv_assets(path: Path) -> dict[str, pd.DataFrame]:
    if path.is_dir():
        assets = {}
        for csv_path in sorted(path.glob("*.csv")):
            asset_name = csv_path.stem
            assets[asset_name] = _normalize_ohlcv_columns(pd.read_csv(csv_path))
        if not assets:
            raise ValueError(f"No CSV files found in directory: {path}")
        return assets

    df = pd.read_csv(path)
    if "asset" in df.columns:
        assets = {}
        for asset, group in df.groupby("asset"):
            group = group.drop(columns=["asset"])
            assets[str(asset)] = _normalize_ohlcv_columns(group)
        if not assets:
            raise ValueError("No assets found in stacked CSV.")
        return assets

    return {path.stem: _normalize_ohlcv_columns(df)}


def get_ewma_crossover_events(close: pd.Series, fast_span: int, slow_span: int, burn_in: int) -> pd.DataFrame:
    fast = close.ewm(span=fast_span, adjust=False).mean()
    slow = close.ewm(span=slow_span, adjust=False).mean()
    fast = fast.iloc[burn_in:]
    slow = slow.iloc[burn_in:]

    diff = fast - slow
    diff_prev = diff.shift(1)
    bullish = (diff_prev <= 0) & (diff > 0)
    bearish = (diff_prev >= 0) & (diff < 0)

    out = pd.DataFrame(index=diff.index)
    out["side"] = np.where(bullish, 1, np.where(bearish, -1, 0))
    return out[out["side"] != 0]


def get_daily_vol(close: pd.Series, volatility_span: int = 100) -> pd.Series:
    prev_day_pos = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    prev_day_pos = prev_day_pos[prev_day_pos > 0]
    prev_idx = pd.Series(
        close.index[prev_day_pos - 1],
        index=close.index[close.shape[0] - prev_day_pos.shape[0] :],
    )
    daily_ret = close.loc[prev_idx.index] / close.loc[prev_idx.values].values - 1
    return daily_ret.ewm(span=volatility_span).std()


def add_vertical_barrier(t_events: pd.Index, close: pd.Series, num_days: int = 1) -> pd.Series:
    t1 = {}
    for ts in t_events:
        t_end = ts + pd.Timedelta(days=num_days)
        idx = close.index.searchsorted(t_end)
        if idx < len(close.index):
            t1[ts] = close.index[idx]
    return pd.Series(t1)


def apply_pt_sl_on_t1(close: pd.Series, events: pd.DataFrame, pt_sl: tuple[float, float]) -> pd.DataFrame:
    out = pd.DataFrame(index=events.index, columns=["pt", "sl"], dtype="datetime64[ns]")
    for loc, event in events.iterrows():
        try:
            loc_pos = close.index.get_loc(loc)
        except KeyError:
            continue

        end_time = event["t1"] if pd.notna(event["t1"]) else close.index[-1]
        end_pos = close.index.get_indexer([end_time], method="bfill")[0]
        start_pos = loc_pos + 1
        if start_pos > end_pos:
            continue

        path = close.iloc[start_pos : end_pos + 1]
        if path.empty:
            continue
        entry = path.iloc[0]
        path_after_entry = path.iloc[1:]
        if path_after_entry.empty:
            continue
        path_ret = (path_after_entry / entry - 1.0) * event["side"]

        pt_mult, sl_mult = pt_sl
        if pt_mult > 0:
            pt_hit = path_ret[path_ret >= pt_mult * event["trgt"]]
            if not pt_hit.empty:
                out.loc[loc, "pt"] = pt_hit.index[0]
        if sl_mult > 0:
            sl_hit = path_ret[path_ret <= -sl_mult * event["trgt"]]
            if not sl_hit.empty:
                out.loc[loc, "sl"] = sl_hit.index[0]
    return out


def get_meta_labeled_events(
    close: pd.Series,
    events_cross: pd.DataFrame,
    pt_sl: tuple[float, float],
    target: pd.Series,
    min_ret: float,
    t1: pd.Series,
) -> pd.DataFrame:
    target = target.reindex(events_cross.index).dropna()
    target = target[target > min_ret]
    if target.empty:
        return pd.DataFrame(columns=["t1", "trgt", "side", "t_hit", "meta_label"])

    events = pd.DataFrame(index=target.index)
    events["t1"] = t1.reindex(events.index)
    events["trgt"] = target
    events["side"] = events_cross.loc[events.index, "side"]

    hits = apply_pt_sl_on_t1(close=close, events=events, pt_sl=pt_sl)
    candidates = pd.concat([events["t1"], hits["pt"], hits["sl"]], axis=1)
    candidates.columns = ["t1", "pt", "sl"]
    events["t_hit"] = candidates.min(axis=1)

    is_pt_first = pd.notna(hits["pt"]) & (hits["pt"] == events["t_hit"])
    events["meta_label"] = is_pt_first.astype(int)
    return events


def build_features(
    ohlcv: pd.DataFrame, events_cross: pd.DataFrame, fast_span: int, slow_span: int
) -> pd.DataFrame:
    close = ohlcv["Close"]
    open_price = ohlcv["Open"]
    high = ohlcv["High"]
    low = ohlcv["Low"]
    volume = ohlcv["Volume"]

    log_ret = np.log(close / close.shift(1))
    ret_1 = log_ret
    ret_5 = log_ret.rolling(5).sum()
    ret_20 = log_ret.rolling(20).sum()

    vol_20 = log_ret.ewm(span=20, adjust=False).std()
    vol_100 = log_ret.ewm(span=100, adjust=False).std()
    vol_ratio = (vol_20 / vol_100).replace([np.inf, -np.inf], np.nan)

    fast = close.ewm(span=fast_span, adjust=False).mean()
    slow = close.ewm(span=slow_span, adjust=False).mean()
    ewma_gap = (fast - slow) / close

    hl_range = (high - low) / close
    co_ret = (close - open_price) / open_price.replace(0, np.nan)
    vol_chg = np.log(volume / volume.shift(1)).replace([np.inf, -np.inf], np.nan)

    idx = events_cross.index
    X = pd.DataFrame(
        {
            "ret_1": ret_1.reindex(idx),
            "ret_5": ret_5.reindex(idx),
            "ret_20": ret_20.reindex(idx),
            "vol_20": vol_20.reindex(idx),
            "vol_ratio": vol_ratio.reindex(idx),
            "ewma_gap": ewma_gap.reindex(idx),
            "hl_range": hl_range.reindex(idx),
            "co_ret": co_ret.reindex(idx),
            "vol_chg": vol_chg.reindex(idx),
            "side": events_cross["side"].astype(float),
        },
        index=idx,
    )
    return X


def _split_assets(
    df_all: pd.DataFrame, test_asset_fraction: float, train_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    assets = sorted(df_all["asset"].unique())
    if len(assets) <= 1:
        df_all = df_all.sort_values("event_time")
        n_train = max(1, int(len(df_all) * train_frac))
        train_df = df_all.iloc[:n_train].copy()
        test_df = df_all.iloc[n_train:].copy()
        return train_df, test_df, assets, assets

    n_train_assets = max(1, int(len(assets) * (1 - test_asset_fraction)))
    n_train_assets = min(n_train_assets, len(assets) - 1)
    train_assets = assets[:n_train_assets]
    test_assets = assets[n_train_assets:]
    train_df = df_all[df_all["asset"].isin(train_assets)].copy()
    test_df = df_all[df_all["asset"].isin(test_assets)].copy()
    return train_df, test_df, train_assets, test_assets


def orchestrate(data_path: Path, out_dir: Path, config: Config) -> dict:
    assets = load_ohlcv_assets(data_path)
    all_rows = []

    for asset, ohlcv in sorted(assets.items()):
        close = ohlcv["Close"]
        events_cross = get_ewma_crossover_events(close, config.fast_span, config.slow_span, config.burn_in)
        if events_cross.empty:
            continue
        target = get_daily_vol(close, volatility_span=config.daily_vol_span)
        t1 = add_vertical_barrier(events_cross.index, close, num_days=config.vertical_days)
        labeled = get_meta_labeled_events(
            close=close,
            events_cross=events_cross,
            pt_sl=config.pt_sl,
            target=target,
            min_ret=config.min_ret,
            t1=t1,
        )
        if labeled.empty:
            continue

        X = build_features(ohlcv, events_cross, config.fast_span, config.slow_span)
        data = X.join(labeled[["meta_label"]], how="inner").dropna()
        if data.empty:
            continue

        data["asset"] = asset
        data["event_time"] = data.index
        all_rows.append(data.reset_index(drop=True))

    if not all_rows:
        raise RuntimeError("No labeled events found in any asset.")

    df_all = pd.concat(all_rows, ignore_index=True)
    train_df, test_df, train_assets, test_assets = _split_assets(
        df_all, config.test_asset_fraction, config.train_frac
    )

    if train_df.empty:
        raise RuntimeError("Training set is empty after asset split.")
    if test_df.empty:
        raise RuntimeError("Test set is empty after asset split.")

    meta_cols = ["asset", "event_time", "meta_label"]
    feature_cols = [c for c in train_df.columns if c not in meta_cols]

    X_train = train_df[feature_cols].fillna(0)
    X_test = test_df[feature_cols].fillna(0)
    y_train = train_df["meta_label"].astype(int)
    y_test = test_df["meta_label"].astype(int)

    scaler = MinMaxScaler(feature_range=config.feature_range)
    X_train_values = scaler.fit_transform(X_train.values)
    X_test_values = scaler.transform(X_test.values)

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=4,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=config.random_state,
    )
    rf.fit(X_train_values, y_train.values)

    y_pred = rf.predict(X_test_values)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_assets": [str(a) for a in train_assets],
        "test_assets": [str(a) for a in test_assets],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": rf,
        "scaler": scaler,
        "feature_columns": feature_cols,
        "metrics": metrics,
    }
    joblib.dump(artifact, out_dir / "rf_ohlcv_multi_asset.joblib")
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    labeled_out = df_all[meta_cols + feature_cols].copy()
    labeled_out.to_csv(out_dir / "ohlcv_multi_asset_labeled_events.csv", index=False)
    pred_df = test_df[["asset", "event_time"]].copy()
    pred_df["y_true"] = y_test.values
    pred_df["y_pred"] = y_pred
    pred_df.to_csv(out_dir / "ohlcv_multi_asset_test_predictions.csv", index=False)

    print("Trained pooled RF model saved to", out_dir)
    print("Metrics:", metrics)
    return artifact


def parse_args() -> argparse.Namespace:
    cfg = Config()
    parser = argparse.ArgumentParser(
        description="Multi-asset EWMA trend following + RF meta-labeling with OHLCV inputs."
    )
    parser.add_argument("--data-path", type=Path, required=True, help="CSV file or directory with OHLCV data.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs_multi_asset_ohlcv",
        help="Directory for output artifacts.",
    )
    parser.add_argument("--fast-span", type=int, default=cfg.fast_span)
    parser.add_argument("--slow-span", type=int, default=cfg.slow_span)
    parser.add_argument("--burn-in", type=int, default=cfg.burn_in)
    parser.add_argument("--daily-vol-span", type=int, default=cfg.daily_vol_span)
    parser.add_argument("--min-ret", type=float, default=cfg.min_ret)
    parser.add_argument("--train-frac", type=float, default=cfg.train_frac)
    parser.add_argument("--test-asset-fraction", type=float, default=cfg.test_asset_fraction)
    parser.add_argument("--random-state", type=int, default=cfg.random_state)
    parser.add_argument("--pt-multiplier", type=float, default=cfg.pt_sl[0])
    parser.add_argument("--sl-multiplier", type=float, default=cfg.pt_sl[1])
    parser.add_argument("--vertical-days", type=int, default=cfg.vertical_days)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        fast_span=args.fast_span,
        slow_span=args.slow_span,
        burn_in=args.burn_in,
        daily_vol_span=args.daily_vol_span,
        min_ret=args.min_ret,
        train_frac=args.train_frac,
        test_asset_fraction=args.test_asset_fraction,
        random_state=args.random_state,
        pt_sl=(args.pt_multiplier, args.sl_multiplier),
        vertical_days=args.vertical_days,
    )
    orchestrate(data_path=args.data_path, out_dir=args.output_dir, config=config)


if __name__ == "__main__":
    main()
