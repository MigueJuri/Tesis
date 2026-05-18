import pandas as pd


def load_data_from_csv(csv_file: str) -> pd.DataFrame:
    # yfinance CSV export uses a 2-row header for MultiIndex columns and a standalone index-name row.
    # data = pd.read_csv(csv_file, header=[0, 1], skiprows=[2], index_col=0)

    # data.index = pd.to_datetime(data.index)
    # data.index.name = "timestamp"
    df = pd.read_csv(csv_file, parse_dates=['timestamp'])
    df.sort_values('timestamp', inplace=True)

    return df


# output_file = r"G:\Mi unidad\2026\Tesis\Códigos\Data\sp500_2015_to_2025_1Min.csv"

# df = load_data_from_csv(output_file)

import numpy as np
import pandas as pd


def form_bars(
    df: pd.DataFrame,
    bar_type: str,
    ewma_window: int = 100,
) -> pd.DataFrame:
    """
    Forms standard or imbalance bars using Prado's exact conditional
    expectation formulation via an autonomous event-driven state machine.

    Daily calibration policy for 1-minute ticks:
    - Compute mean ticks per day from all dates.
    - Burn-in uses round(ewma_window * mean_ticks_per_day) ticks.
    - Standard-bar threshold is mean_ticks_per_day * EWMA(magnitude per tick).
    - Bars start strictly after burn-in.
    """
    _EMPTY = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "VWAP"])

    if df.empty:
        return _EMPTY

    # ------------------------------------------------------------------
    # 1. Resolve bar type & identify the seed metric
    # ------------------------------------------------------------------
    bt = bar_type.strip().lower()
    is_imbalance = "imbalance" in bt

    if is_imbalance:
        base_metric = "dollar"
    else:
        if bt not in {"tick", "volume", "dollar"}:
            raise ValueError("bar_type must be one of: 'tick', 'volume', 'dollar', 'imbalance'")
        base_metric = bt

    # ------------------------------------------------------------------
    # 2. Daily calibration & burn-in length in ticks
    # ------------------------------------------------------------------
    ts = pd.to_datetime(df["timestamp"])
    ticks_per_day = ts.dt.normalize().value_counts().astype(float)
    if ticks_per_day.empty:
        return _EMPTY

    mean_ticks_per_day = float(ticks_per_day.mean())
    if not np.isfinite(mean_ticks_per_day) or mean_ticks_per_day <= 0:
        return _EMPTY

    burn_in_ticks = int(max(1, round(ewma_window * mean_ticks_per_day)))

    # ------------------------------------------------------------------
    # 3. Build seed metric and initialize expectation from burn-in
    # ------------------------------------------------------------------
    if base_metric == "tick":
        full_seed_values = df["trade_count"].astype(float)
    elif base_metric == "volume":
        full_seed_values = df["volume"].astype(float)
    else:  # dollar
        full_seed_values = df["vwap"].astype(float) * df["volume"].astype(float)

    if len(full_seed_values) <= burn_in_ticks + 1:
        return _EMPTY

    # Use day-scale EWMA decay with tick-scale burn-in horizon.
    mapped_seeds = full_seed_values.ewm(
        span=ewma_window,
        adjust=False,
        min_periods=burn_in_ticks + 1,
    ).mean()
    
    initial_magnitude_per_tick = full_seed_values.median() #mapped_seeds.iloc[burn_in_ticks]
    if pd.isna(initial_magnitude_per_tick):
        return _EMPTY

    initial_threshold_standard = mean_ticks_per_day * float(initial_magnitude_per_tick)

    start_idx = burn_in_ticks + 1
    df = df.iloc[start_idx:].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. Extract raw arrays after burn-in
    # ------------------------------------------------------------------
    times = df["timestamp"].values
    opens = df["open"].values.astype(float)
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    volumes = df["volume"].values.astype(float)
    vwaps = df["vwap"].values.astype(float)

    n = len(volumes)
    if n == 0:
        return _EMPTY

    # ------------------------------------------------------------------
    # 5. Driving variable magnitude
    # ------------------------------------------------------------------
    if base_metric == "tick":
        magnitudes = df["trade_count"].astype(float).to_numpy()
    elif base_metric == "volume":
        magnitudes = df["volume"].astype(float).to_numpy()
    else:  # dollar
        magnitudes = (df["vwap"].astype(float) * df["volume"].astype(float)).to_numpy()

    if is_imbalance:
        dp = np.empty(n)
        dp[0] = np.nan
        dp[1:] = vwaps[1:] - vwaps[:-1]

        raw_signs = np.sign(dp)
        sign_series = pd.Series(np.where(raw_signs == 0, np.nan, raw_signs))
        b = sign_series.ffill().fillna(1.0).to_numpy()

        values = b * magnitudes
    else:
        values = magnitudes

    # ------------------------------------------------------------------
    # 6. Bar formation loop (Autonomous event-driven state machine)
    # ------------------------------------------------------------------
    out_open = np.empty(n)
    out_high = np.empty(n)
    out_low = np.empty(n)
    out_close = np.empty(n)
    out_vol = np.empty(n)
    out_vwap = np.empty(n)
    out_times = np.empty(n, dtype=times.dtype)

    cumulative_sum = 0.0
    bar_idx = 0
    current_high = -np.inf
    current_low = np.inf
    current_open = opens[0]
    current_vol = 0.0
    current_dollar = 0.0

    # Intrabar tracking for the current bar
    bar_actual_magnitude = 0.0
    bar_ticks = 0.0
    bar_upticks = 0.0
    bar_downticks = 0.0
    bar_mag_up = 0.0
    bar_mag_down = 0.0

    # ------------------------------------------------------------------
    # Event-driven EWMA states & priors
    # ------------------------------------------------------------------
    alpha = 2.0 / (ewma_window + 1.0)

    # Standard bars: threshold = mean_ticks_per_day * EWMA(magnitude_per_tick)
    ewma_magnitude_per_tick = float(initial_magnitude_per_tick)

    # Keep imbalance initialization based on bar-scale magnitude and existing priors.
    expected_magnitude = float(initial_threshold_standard)

    if is_imbalance:
        expected_T_per_bar = 20.0
        expected_p_up = 0.5
        expected_p_down = 0.5
        expected_v_up = expected_magnitude / expected_T_per_bar
        expected_v_down = expected_magnitude / expected_T_per_bar

        expected_imbalance_per_tick = abs(expected_p_up * expected_v_up - expected_p_down * expected_v_down)
        bar_threshold = expected_T_per_bar * expected_imbalance_per_tick
    else:
        bar_threshold = max(initial_threshold_standard, 1e-12)

    for i in range(n):
        cumulative_sum += values[i]
        bar_actual_magnitude += magnitudes[i]
        bar_ticks += 1.0

        if is_imbalance:
            if b[i] == 1.0:
                bar_upticks += 1.0
                bar_mag_up += magnitudes[i]
            else:
                bar_downticks += 1.0
                bar_mag_down += magnitudes[i]

        current_vol += volumes[i]
        current_dollar += vwaps[i] * volumes[i]

        if highs[i] > current_high:
            current_high = highs[i]
        if lows[i] < current_low:
            current_low = lows[i]

        if is_imbalance:
            trigger = abs(cumulative_sum) >= bar_threshold
        else:
            trigger = cumulative_sum >= bar_threshold

        if trigger:
            out_open[bar_idx] = current_open
            out_high[bar_idx] = current_high
            out_low[bar_idx] = current_low
            out_close[bar_idx] = closes[i]
            out_vol[bar_idx] = current_vol
            out_vwap[bar_idx] = current_dollar / current_vol if current_vol > 0 else closes[i]
            out_times[bar_idx] = times[i]
            bar_idx += 1

            # Update event-driven expectations
            if is_imbalance:
                p_up = bar_upticks / bar_ticks if bar_ticks > 0 else 0.5
                p_down = bar_downticks / bar_ticks if bar_ticks > 0 else 0.5

                v_up = bar_mag_up / bar_upticks if bar_upticks > 0 else expected_v_up
                v_down = bar_mag_down / bar_downticks if bar_downticks > 0 else expected_v_down

                expected_T_per_bar = (alpha * bar_ticks) + ((1.0 - alpha) * expected_T_per_bar)
                expected_p_up = (alpha * p_up) + ((1.0 - alpha) * expected_p_up)
                expected_p_down = (alpha * p_down) + ((1.0 - alpha) * expected_p_down)
                expected_v_up = (alpha * v_up) + ((1.0 - alpha) * expected_v_up)
                expected_v_down = (alpha * v_down) + ((1.0 - alpha) * expected_v_down)

                expected_imbalance_per_tick = abs(
                    expected_p_up * expected_v_up - expected_p_down * expected_v_down
                )

                expected_total_mag = expected_T_per_bar * (
                    expected_p_up * expected_v_up + expected_p_down * expected_v_down
                )
                floor = max(0.01 * expected_total_mag, 1e-5)

                bar_threshold = max(expected_T_per_bar * expected_imbalance_per_tick, floor)
            else:
                bar_mag_per_tick = (
                    bar_actual_magnitude / bar_ticks if bar_ticks > 0 else ewma_magnitude_per_tick
                )
                ewma_magnitude_per_tick = (
                    (alpha * bar_mag_per_tick) + ((1.0 - alpha) * ewma_magnitude_per_tick)
                )

                bar_threshold = max(mean_ticks_per_day * ewma_magnitude_per_tick, 1e-12)

            # Reset internal accumulators
            cumulative_sum = 0.0
            bar_actual_magnitude = 0.0
            bar_ticks = 0.0
            bar_upticks = 0.0
            bar_downticks = 0.0
            bar_mag_up = 0.0
            bar_mag_down = 0.0
            current_high = -np.inf
            current_low = np.inf
            current_vol = 0.0
            current_dollar = 0.0

            if i + 1 < n:
                current_open = opens[i + 1]

    # ------------------------------------------------------------------
    # 7. Assemble output
    # ------------------------------------------------------------------
    bars_df = pd.DataFrame(
        {
            "Open": out_open[:bar_idx],
            "High": out_high[:bar_idx],
            "Low": out_low[:bar_idx],
            "Close": out_close[:bar_idx],
            "Volume": out_vol[:bar_idx],
            "VWAP": out_vwap[:bar_idx],
        },
        index=out_times[:bar_idx],
    )

    bars_df.index.name = "timestamp"
    return bars_df

# ewma_window = 252
# _dates = pd.to_datetime(df["timestamp"]).dt.normalize()
# _ticks_per_day = _dates.value_counts().sort_index()
# _mean_ticks_per_day = float(_ticks_per_day.mean())
# _burn_in_ticks = int(max(1, round(ewma_window * _mean_ticks_per_day)))

# print(f"days={len(_ticks_per_day)}")
# print(f"mean_ticks_per_day={_mean_ticks_per_day:.2f}")
# print(f"burn_in_ticks(ewma_window={ewma_window})={_burn_in_ticks}")

# dollar_bars = form_bars(df, bar_type="dollar", ewma_window=ewma_window)
# dollar_bars.to_csv(r"G:\Mi unidad\2026\Tesis\Códigos\Data\sp500_2015_to_2025_dollar_bars.csv")