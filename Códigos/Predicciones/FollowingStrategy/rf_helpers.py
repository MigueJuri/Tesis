import numpy as np
import pandas as pd


def get_crossover_events(close: pd.Series, fast_span: int = 3, slow_span: int = 10, burn_in: int = 10) -> pd.DataFrame:
    """Detect EWMA crossover events on a price Series.
    Returns DataFrame indexed by event timestamp with columns ['fast','slow','side'] where side is +1/-1.
    """
    fast = close.ewm(span=fast_span, adjust=False).mean()
    slow = close.ewm(span=slow_span, adjust=False).mean()
    diff = fast - slow
    sign = np.sign(diff)
    sign_shift = sign.shift(1).fillna(0)
    cross = (sign != sign_shift) & (sign != 0)
    events_idx = close.index[cross.values]
    sides = sign[cross]
    df = pd.DataFrame({'fast': fast.loc[events_idx], 'slow': slow.loc[events_idx], 'side': sides}, index=events_idx)
    return df


def _apply_barriers(close: pd.Series, events: pd.DataFrame, pt_sl: tuple = (1.0, 1.0), num_bars: int = 20) -> pd.DataFrame:
    """Simple triple-barrier: for each event index, compute t1 as event_pos+num_bars (bounded).
    Checks whether future high/low reaches PT/SL relative to event price (using absolute returns).
    """
    ix = close.index
    index_pos = {t: i for i, t in enumerate(ix)}
    rows = []
    for ts, ev in events.iterrows():
        i = index_pos.get(ts, None)
        if i is None:
            continue
        t1_pos = min(i + num_bars, len(ix) - 1)
        t1 = ix[t1_pos]
        start_price = close.loc[ts]
        # compute max/min within window
        fut_slice = close.iloc[i:t1_pos + 1]
        max_ret = (fut_slice.max() - start_price) / start_price
        min_ret = (fut_slice.min() - start_price) / start_price
        pt, sl = pt_sl
        hit = None
        if max_ret >= pt:
            hit = 'pt'
        elif min_ret <= -sl:
            hit = 'sl'
        else:
            hit = 'none'
        rows.append({'t0': ts, 't1': t1, 'start_price': start_price, 'max_ret': max_ret, 'min_ret': min_ret, 'hit': hit})
    return pd.DataFrame(rows).set_index('t0')


def get_labeled_events(close: pd.Series, events: pd.DataFrame, pt_sl: tuple = (0.01, 0.01), vol_span: int = 20, min_ret: float = 1e-8, num_bars: int = 20) -> pd.DataFrame:
    """Return labeled events DataFrame with columns: t1, trgt, side, label.
    Label: +1 (profit in direction), -1 (loss), 0 (no-touch or opposite)
    """
    # volatility target (simple rolling std of log returns)
    logret = np.log(close / close.shift(1)).fillna(0)
    trgt = logret.rolling(vol_span).std().fillna(logret.std())
    applied = _apply_barriers(close, events, pt_sl=pt_sl, num_bars=num_bars)
    rows = []
    for idx, ev in events.iterrows():
        if idx not in applied.index:
            continue
        a = applied.loc[idx]
        side = int(ev['side'])
        # determine label
        if a['hit'] == 'pt':
            label = side
        elif a['hit'] == 'sl':
            label = -side
        else:
            label = 0
        rows.append({'t0': idx, 't1': a['t1'], 'trgt': trgt.loc[idx] if idx in trgt.index else trgt.iloc[0], 'side': side, 'label': label})
    labeled = pd.DataFrame(rows).set_index('t0')
    return labeled


def build_features(close: pd.Series, events: pd.DataFrame) -> pd.DataFrame:
    """Compute simple features at event timestamps. Returns DataFrame indexed by event timestamp.
    Features: ret_1, ret_5, ret_20, vol_20, ewma_gap, rsi_14 (simple), bb_pct
    """
    price = close
    logret = np.log(price / price.shift(1))
    ret_1 = logret
    ret_5 = logret.rolling(5).sum()
    ret_20 = logret.rolling(20).sum()
    vol_20 = logret.rolling(20).std()
    ewma_fast = price.ewm(span=3, adjust=False).mean()
    ewma_slow = price.ewm(span=10, adjust=False).mean()
    ewma_gap = (ewma_fast - ewma_slow) / price
    # RSI 14 (simplified)
    up = logret.clip(lower=0).rolling(14).mean()
    down = -logret.clip(upper=0).rolling(14).mean()
    rsi = 100 - 100 / (1 + up / down.replace(0, np.nan))
    # Bollinger pct
    ma = price.rolling(20).mean()
    std = price.rolling(20).std()
    bb_pct = (price - (ma - 2 * std)) / (4 * std)
    feat_idx = [ts for ts in events.index if ts in price.index]
    X = pd.DataFrame(index=feat_idx)
    X['ret_1'] = ret_1.loc[feat_idx].values
    X['ret_5'] = ret_5.loc[feat_idx].values
    X['ret_20'] = ret_20.loc[feat_idx].values
    X['vol_20'] = vol_20.loc[feat_idx].values
    X['ewma_gap'] = ewma_gap.loc[feat_idx].values
    X['rsi_14'] = rsi.loc[feat_idx].values
    X['bb_pct'] = bb_pct.loc[feat_idx].values
    return X.fillna(0)


def compute_sample_weights(labeled: pd.DataFrame) -> pd.Series:
    """Compute sample weights per event using overlap uniqueness within the labeled set.
    For each event interval [t0, t1], weight = 1 / (number of intervals that overlap this interval).
    Returns a Series indexed by t0.
    """
    if labeled.empty:
        return pd.Series(dtype=float)
    intervals = []
    for t0, row in labeled.iterrows():
        t1 = row['t1']
        intervals.append((t0, t1))
    # count overlaps
    counts = {}
    for i, (a0, a1) in enumerate(intervals):
        cnt = 0
        for j, (b0, b1) in enumerate(intervals):
            # overlap if start before other's end and end after other's start
            if (a0 <= b1) and (a1 >= b0):
                cnt += 1
        counts[a0] = cnt
    # weight = 1 / count
    weights = {t0: 1.0 / max(c, 1) for t0, c in counts.items()}
    return pd.Series(weights)


def build_position(close: pd.Series, signals: pd.Series) -> pd.Series:
    """Given a price series and a signals Series indexed by timestamps with values {1,0,-1}, produce position Series aligned with close index.
    This is a naive 'hold until next opposite signal or end' position builder.
    """
    pos = pd.Series(0, index=close.index)
    current = 0
    sig_iter = signals.sort_index()
    for ts, sig in sig_iter.items():
        if ts not in pos.index:
            # align to nearest index >= ts
            candidates = pos.index[pos.index >= ts]
            if len(candidates) == 0:
                continue
            ts_aligned = candidates[0]
        else:
            ts_aligned = ts
        current = sig
        pos.loc[ts_aligned:] = current
    return pos


def summarize_strategy(name: str, pos: pd.Series, close: pd.Series) -> dict:
    returns = close.pct_change().fillna(0) * pos.shift(1).fillna(0)
    equity = (1 + returns).cumprod()
    total_return = equity.iloc[-1] - 1
    drawdown = equity.cummax() - equity
    max_dd = drawdown.max()
    ann_factor = 252
    ret_std = float(returns.std())
    sharpe = float(np.sqrt(ann_factor) * returns.mean() / ret_std) if ret_std > 0 else 0.0
    summary = {
        'name': name,
        'total_return': float(total_return),
        'max_drawdown': float(max_dd),
        'sharpe': sharpe,
        'equity': equity,
        'returns': returns,
    }
    return summary
