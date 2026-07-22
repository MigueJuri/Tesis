from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score


@dataclass
class Config:
    fast_span: int = 5
    slow_span: int = 20
    burn_in: int = 20
    daily_vol_span: int = 100
    min_ret: float = 0.0
    train_frac: float = 0.7
    random_state: int = 42
    pt_sl: tuple[float, float] = (1.0, 2.0)
    vertical_days: int = 1


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_data_path() -> Path:
    return get_repo_root() / "Códigos" / "Data" / "sp500_data_only_1993-01-29_to_2026-01-02.csv"


def load_close_series(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)

    if {"timestamp", "Close"}.issubset(df.columns):
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        close = pd.to_numeric(df["Close"], errors="coerce")
        out = pd.Series(close.values, index=ts, name="Close").dropna()
        return out.sort_index()

    if {"Date", "Close"}.issubset(df.columns):
        if not df.empty and str(df.iloc[0, 0]).strip().lower() == "ticker":
            df = df.iloc[2:].copy()
        ts = pd.to_datetime(df["Date"], errors="coerce")
        close = pd.to_numeric(df["Close"], errors="coerce")
        out = pd.Series(close.values, index=ts, name="Close").dropna()
        return out.sort_index()

    if "Close" in df.columns:
        first_col = df.columns[0]
        ts = pd.to_datetime(df[first_col], errors="coerce")
        close = pd.to_numeric(df["Close"], errors="coerce")
        out = pd.Series(close.values, index=ts, name="Close").dropna()
        return out.sort_index()

    raise ValueError(
        f"Unsupported CSV schema for {csv_path}. Expected either "
        "[timestamp, Close] or [Date, Close]."
    )


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


def get_daily_vol(close: pd.Series, span0: int = 100) -> pd.Series:
    prev_day_pos = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    prev_day_pos = prev_day_pos[prev_day_pos > 0]
    prev_idx = pd.Series(
        close.index[prev_day_pos - 1],
        index=close.index[close.shape[0] - prev_day_pos.shape[0] :],
    )
    daily_ret = close.loc[prev_idx.index] / close.loc[prev_idx.values].values - 1
    return daily_ret.ewm(span=span0).std()


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
    trgt: pd.Series,
    min_ret: float,
    t1: pd.Series,
) -> pd.DataFrame:
    trgt = trgt.reindex(events_cross.index).dropna()
    trgt = trgt[trgt > min_ret]
    if trgt.empty:
        return pd.DataFrame(columns=["t1", "trgt", "side", "t_hit", "meta_label"])

    events = pd.DataFrame(index=trgt.index)
    events["t1"] = t1.reindex(events.index)
    events["trgt"] = trgt
    events["side"] = events_cross.loc[events.index, "side"]

    hits = apply_pt_sl_on_t1(close=close, events=events, pt_sl=pt_sl)
    candidates = pd.concat([events["t1"], hits["pt"], hits["sl"]], axis=1)
    candidates.columns = ["t1", "pt", "sl"]
    events["t_hit"] = candidates.min(axis=1)

    is_pt_first = pd.notna(hits["pt"]) & (hits["pt"] == events["t_hit"])
    events["meta_label"] = is_pt_first.astype(int)
    return events


def build_features(close: pd.Series, events_cross: pd.DataFrame, fast_span: int, slow_span: int) -> pd.DataFrame:
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

    idx = events_cross.index
    X = pd.DataFrame(
        {
            "ret_1": ret_1.reindex(idx),
            "ret_5": ret_5.reindex(idx),
            "ret_20": ret_20.reindex(idx),
            "vol_20": vol_20.reindex(idx),
            "vol_ratio": vol_ratio.reindex(idx),
            "ewma_gap": ewma_gap.reindex(idx),
            "side": events_cross["side"].astype(float),
        },
        index=idx,
    )
    return X


def train_rf_classifier(X: pd.DataFrame, y: pd.Series, train_frac: float, random_state: int) -> tuple[RandomForestClassifier, pd.Index, pd.Index]:
    n = len(X)
    n_train = int(n * train_frac)
    train_idx = X.index[:n_train]
    test_idx = X.index[n_train:]

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=4,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X.loc[train_idx], y.loc[train_idx])
    return rf, train_idx, test_idx


def run(config: Config, data_path: Path, output_dir: Path) -> None:
    close = load_close_series(data_path)
    events_cross = get_ewma_crossover_events(close, config.fast_span, config.slow_span, config.burn_in)
    if events_cross.empty:
        raise RuntimeError("No crossover events detected with current parameters.")

    trgt = get_daily_vol(close, span0=config.daily_vol_span)
    t1 = add_vertical_barrier(events_cross.index, close, num_days=config.vertical_days)
    labeled = get_meta_labeled_events(
        close=close,
        events_cross=events_cross,
        pt_sl=config.pt_sl,
        trgt=trgt,
        min_ret=config.min_ret,
        t1=t1,
    )
    if labeled.empty:
        raise RuntimeError("No labeled events available after filtering.")

    X = build_features(close, events_cross, config.fast_span, config.slow_span)
    data = X.join(labeled[["meta_label"]], how="inner").dropna()
    if data.empty:
        raise RuntimeError("No rows remaining after feature/label alignment.")

    y = data["meta_label"].astype(int)
    X_final = data.drop(columns=["meta_label"])

    rf, train_idx, test_idx = train_rf_classifier(X_final, y, config.train_frac, config.random_state)
    y_pred_train = rf.predict(X_final.loc[train_idx])
    y_pred_test = rf.predict(X_final.loc[test_idx])

    print("AFML Exercise 3.4 — RF Trade Filter")
    print(f"Data points: {len(X_final)} | Train: {len(train_idx)} | Test: {len(test_idx)}")
    print(f"Label positive rate: {y.mean():.3f}")
    print(f"Train F1: {f1_score(y.loc[train_idx], y_pred_train, zero_division=0):.4f}")
    print(f"Test  F1: {f1_score(y.loc[test_idx], y_pred_test, zero_division=0):.4f}")
    print(f"Test Precision: {precision_score(y.loc[test_idx], y_pred_test, zero_division=0):.4f}")
    print(f"Test Recall:    {recall_score(y.loc[test_idx], y_pred_test, zero_division=0):.4f}")
    print(classification_report(y.loc[test_idx], y_pred_test, target_names=["Skip (0)", "Trade (1)"], zero_division=0))

    output_dir.mkdir(parents=True, exist_ok=True)
    labeled_out = labeled.join(X, how="left")
    labeled_out.to_csv(output_dir / "afml_ex3_4_labeled_events.csv", index=True)
    pred_df = pd.DataFrame(
        {
            "y_true": y.loc[test_idx],
            "y_pred": pd.Series(y_pred_test, index=test_idx),
            "side": X_final.loc[test_idx, "side"],
        },
        index=test_idx,
    )
    pred_df.to_csv(output_dir / "afml_ex3_4_test_predictions.csv", index=True)
    print(f"Saved outputs to: {output_dir}")


def parse_args() -> argparse.Namespace:
    cfg = Config()
    parser = argparse.ArgumentParser(
        description="AFML Exercise 3.4: trend-following side model + RF meta-label trade filter."
    )
    parser.add_argument("--data-path", type=Path, default=default_data_path(), help="Path to input CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=get_repo_root() / "Códigos" / "Predicciones" / "FollowingStrategy" / "outputs_ex3_4",
        help="Directory for output CSV files.",
    )
    parser.add_argument("--fast-span", type=int, default=cfg.fast_span)
    parser.add_argument("--slow-span", type=int, default=cfg.slow_span)
    parser.add_argument("--burn-in", type=int, default=cfg.burn_in)
    parser.add_argument("--daily-vol-span", type=int, default=cfg.daily_vol_span)
    parser.add_argument("--min-ret", type=float, default=cfg.min_ret)
    parser.add_argument("--train-frac", type=float, default=cfg.train_frac)
    parser.add_argument("--random-state", type=int, default=cfg.random_state)
    parser.add_argument("--pt-multiplier", type=float, default=cfg.pt_sl[0], help="Profit-take multiplier.")
    parser.add_argument("--sl-multiplier", type=float, default=cfg.pt_sl[1], help="Stop-loss multiplier.")
    parser.add_argument("--vertical-days", type=int, default=cfg.vertical_days, help="Vertical barrier in days.")
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
        random_state=args.random_state,
        pt_sl=(args.pt_multiplier, args.sl_multiplier),
        vertical_days=args.vertical_days,
    )
    run(config=config, data_path=args.data_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
