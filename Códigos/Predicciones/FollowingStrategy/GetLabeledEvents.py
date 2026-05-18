import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics  import (classification_report, precision_score, recall_score, f1_score,
                               ConfusionMatrixDisplay)
from sklearn.utils    import check_random_state

warnings.filterwarnings("ignore")

# ── Load data ──────────────────────────────────────────────────────────────
DATA_PATH = r'G:\Mi unidad\2026\Tesis\Códigos\Data\sp500_2015_to_2025_dollar_bars.csv'
def read_data(path):
    raw   = pd.read_csv(DATA_PATH)
    raw['timestamp'] = pd.to_datetime(raw['timestamp'])
    raw   = raw.set_index('timestamp').sort_index()
    close = raw['Close'].astype(float)
    high  = raw['High'].astype(float)
    low   = raw['Low'].astype(float)
    volume = raw['Volume'].astype(float) if 'Volume' in raw.columns else None
    return close, high, low, volume

# close, high, low, volume = read_data(DATA_PATH)
# print(f"Loaded {len(close):,} bars  "
#       f"| {close.index[0].date()} → {close.index[-1].date()}")


# ── Hyper-parameters (mirror v2) ──────────────────────────────────────────
FAST_SPAN = 3
SLOW_SPAN = 5
BURN_IN   = SLOW_SPAN * 3
NUM_BARS  = 21          # vertical barrier
PT_SL     = [1.0, 1.0]  # symmetric barriers
MIN_RET   = 0.001
VOL_SPAN  = 20

# ── EWMA crossover events ─────────────────────────────────────────────────
def get_crossover_events(close, fast_span=FAST_SPAN, slow_span=SLOW_SPAN,
                         burn_in=BURN_IN):
    fast = close.ewm(span=fast_span, adjust=False).mean()
    slow = close.ewm(span=slow_span, adjust=False).mean()
    fast, slow = fast.iloc[burn_in:], slow.iloc[burn_in:]
    diff = fast - slow
    bullish = (diff.shift(1) <= 0) & (diff > 0)
    bearish = (diff.shift(1) >= 0) & (diff < 0)
    df = pd.DataFrame({'fast_ewma': fast, 'slow_ewma': slow})
    df['side'] = np.where(bullish, 1, np.where(bearish, -1, 0))
    return df[df['side'] != 0].copy()

# events_cross = get_crossover_events(close)
# print(f"Crossover events: {len(events_cross)} "
#       f"(bull={( events_cross['side']==1).sum()}, "
#       f"bear={(events_cross['side']==-1).sum()})")

# vol_20 = close.pct_change().ewm(span=20, adjust=False).std()

def _apply_barriers(close, events, pt_sl):
    """Return DataFrame(pt, sl) — first time each barrier is touched."""
    # Prepare an output DataFrame indexed by event entry times with two
    # columns: 'pt' (profit-taking touch time) and 'sl' (stop-loss touch time).
    out = pd.DataFrame(index=events.index,
                       columns=['pt', 'sl'],
                       dtype='datetime64[ns]')
    # Iterate over labeled events; 'loc' is the event timestamp (entry).
    for loc, ev in events.iterrows():
        try:
            # Find the integer location of the event in the price index.
            lp = close.index.get_loc(loc)
        except KeyError:
            # If the event timestamp is not present in the price index, skip.
            continue
        # Determine the index position of the vertical barrier (t1).
        ep = close.index.get_indexer(
            [ev['t1'] if pd.notna(ev['t1']) else close.index[-1]],
            method='bfill')[0]
        # Start checking from the next bar after the entry.
        sp = lp + 1
        # If the entry is after the barrier, nothing to check.
        if sp > ep:
            continue
        # Extract the price path between entry+1 and the barrier (inclusive).
        path = close.iloc[sp: ep + 1]
        # Need at least two bars (entry reference + one subsequent bar) to compute returns.
        if len(path) < 2:
            continue
        # Compute returns relative to the first bar in the path, then align to the
        # signal direction by multiplying by ev['side'] (+1/-1).
        rets = (path.iloc[1:] / path.iloc[0] - 1.0) * ev['side']
        # Profit-taking: first time return >= pt threshold * target volatility.
        if pt_sl[0] > 0:
            hit = rets[rets >=  pt_sl[0] * ev['trgt']]
            if not hit.empty:
                # Record the timestamp of the first hit (earliest index in hit).
                out.loc[loc, 'pt'] = hit.index[0]
        # Stop-loss: first time return <= -sl threshold * target volatility.
        if pt_sl[1] > 0:
            hit = rets[rets <= -pt_sl[1] * ev['trgt']]
            if not hit.empty:
                out.loc[loc, 'sl'] = hit.index[0]
    # Ensure returned timestamps are proper datetimes (NaT preserved where no hit).
    out['pt'] = pd.to_datetime(out['pt'])
    out['sl'] = pd.to_datetime(out['sl'])
    return out


def get_labeled_events(close, events_cross, pt_sl, vol_span,
                       min_ret, num_bars, target_multiplier = 3):
    """
    Build the canonical labeled-event table.

    Returns columns: t1, trgt, side, t_hit, label
        label = +1 (PT hit)
                -1 (SL hit)
                 0 (vertical barrier / timeout)
    """
    # All event timestamps from the crossover detector
    tEvents = events_cross.index
    # Target (trgt) is the EWM standard deviation of 1-bar returns (volatility estimate).
    trgt    = target_multiplier * close.pct_change().ewm(span=vol_span, adjust=False).std()
    # Keep only targets for times where we actually had events; drop NaNs.
    trgt    = trgt.reindex(tEvents).dropna()
    # Discard events whose volatility target is too small (filter by min_ret).
    trgt    = trgt[trgt > min_ret]

    # Vertical barrier: map each event to its barrier timestamp (num_bars ahead).
    pos = close.index.get_indexer(trgt.index, method='backfill')
    barrier_pos = np.minimum(pos + num_bars, len(close) - 1)
    t1 = pd.Series(close.index[barrier_pos], index=trgt.index)

    # Build base event table with t1 and volatility target
    ev = pd.DataFrame({'t1': t1, 'trgt': trgt})
    # Attach the EWMA side signal (+1/-1) to each event (reindex to align).
    ev['side'] = events_cross['side'].reindex(ev.index).astype(float)

    # Find first touch times for pt/sl within each event window.
    hits = _apply_barriers(close, ev, pt_sl)

    # Combine the vertical-barrier time with any pt/sl hit times so we can
    # compute the earliest time that ended the event (t_hit).
    cand = pd.concat([ev['t1'], hits['pt'], hits['sl']], axis=1)
    cand.columns = ['t1', 'pt', 'sl']
    # Stack columns to a single Series per event and take the earliest timestamp.
    ev['t_hit'] = cand.stack(dropna=True).groupby(level=0).min()

    # Determine label: +1 if PT occurred in signal direction, -1 if SL occurred
    # (opposite direction), 0 otherwise (should include vertical/timeouts).
    def _label(row):
        if pd.isna(row['t_hit']):
            return 0
        # If profit-taking timestamp exists and is the one that ended the event,
        # label according to the signal side (correct trade).
        if pd.notna(hits.loc[row.name, 'pt']) and row['t_hit'] == hits.loc[row.name, 'pt']:
            return int(row['side'])
        # If stop-loss timestamp exists and ended the event, label opposite sign.
        if pd.notna(hits.loc[row.name, 'sl']) and row['t_hit'] == hits.loc[row.name, 'sl']:
            return -int(row['side'])
        return 0

    ev['label'] = ev.apply(_label, axis=1)
    ev['meta_label'] = (ev['label'] * ev['side'] > 0).astype(int)
    return ev


# labeled = get_labeled_events(close, events_cross,
#                              PT_SL, VOL_SPAN, MIN_RET, NUM_BARS)
# print(f"Labeled events : {len(labeled)}")
# print(labeled['label'].value_counts()
#       .rename({1:'PT (+1)', -1:'SL (-1)', 0:'Vertical (0)'}))

def build_features(close, events_cross, fast_span=FAST_SPAN,
                   slow_span=SLOW_SPAN):
    """
    Compute features at each event timestamp using only look-back information.

    All rolling/EWM computations are performed on the full price series first;
    then we simply index into the result at event timestamps — no forward bias.
    """
    log_ret = np.log(close / close.shift(1))

    # ── Returns ──────────────────────────────────────────────────────────────
    ret_1  = log_ret
    ret_5  = log_ret.rolling(5).sum()
    ret_20 = log_ret.rolling(20).sum()

    # ── Volatility ───────────────────────────────────────────────────────────
    vol_5  = log_ret.ewm(span=5,  adjust=False).std()
    vol_20 = log_ret.ewm(span=20, adjust=False).std()
    vol_ratio = (vol_5 / vol_20).replace([np.inf, -np.inf], np.nan)

    # ── EWMA gap (normalized) ─────────────────────────────────────────────────
    fast   = close.ewm(span=fast_span, adjust=False).mean()
    slow   = close.ewm(span=slow_span, adjust=False).mean()
    ewma_gap = (fast - slow) / close

    # ── RSI (14) ──────────────────────────────────────────────────────────────
    delta  = close.diff()
    gain   = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss   = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rs     = gain / loss.replace(0, np.nan)
    rsi_14 = 100 - 100 / (1 + rs)

    # ── Bollinger %B (20, 2σ) ─────────────────────────────────────────────────
    bb_mid   = close.rolling(50).mean()
    bb_std   = close.rolling(50).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    # bb_pct   = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    bb_pct = (close - bb_mid)/(2*bb_std)

    # ── Assemble at event timestamps ──────────────────────────────────────────
    idx = events_cross.index
    X = pd.DataFrame({
        # 'ret_1'    : ret_1.reindex(idx),
        # 'ret_5'    : ret_5.reindex(idx),
        # 'ret_20'   : ret_20.reindex(idx),
        # 'vol_20'   : vol_20.reindex(idx),
        # 'vol_ratio': vol_ratio.reindex(idx),
        # 'ewma_gap' : ewma_gap.reindex(idx),
        'rsi_14'   : rsi_14.reindex(idx),
        # 'bb_pct'   : bb_pct.reindex(idx),
        'side'     : events_cross['side'].astype(float),
    }, index=idx)

    return X