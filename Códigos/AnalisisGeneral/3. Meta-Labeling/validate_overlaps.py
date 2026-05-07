import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score

# Load data (same path used in notebook)
DATA_PATH = r'G:\Mi unidad\2026\Tesis\Códigos\Data\sp500_2015_to_2025_dollar_bars.csv'

data = pd.read_csv(DATA_PATH)
data['timestamp'] = pd.to_datetime(data['timestamp'])
if not pd.api.types.is_datetime64_any_dtype(data.index):
    data = data.set_index('timestamp')
close = data['Close']

# Simple EWMA crossing event generator

def get_EWMA(series, span):
    return series.ewm(span=span, adjust=False).mean()

def get_crossing_moving_averages_events(close, fast_span=10, slow_span=40):
    fast = get_EWMA(close, fast_span)
    slow = get_EWMA(close, slow_span)
    signal = (fast - slow).apply(np.sign)
    # event when sign changes (exclude zeros)
    event = signal[signal != 0].diff().fillna(0)
    tEvents = event[event != 0].index
    events = pd.DataFrame(index=tEvents)
    events['event'] = signal.reindex(tEvents).fillna(0).astype(int)
    return events

# applyPtSlOnT1 copied/ported from notebook
def applyPtSlOnT1(close: pd.Series, events: pd.DataFrame, ptSl, molecule):
    if not isinstance(close, pd.Series):
        raise ValueError("close must be a pd.Series")
    if not isinstance(events, pd.DataFrame):
        raise ValueError("events must be a pd.DataFrame")

    events_ = events.loc[pd.Index(molecule)].copy()
    out = pd.DataFrame(index=events_.index, columns=["pt", "sl"])
    out["pt"] = pd.NaT
    out["sl"] = pd.NaT
    last_close_time = close.index[-1]

    for loc, event in events_.iterrows():
        try:
            loc_pos = close.index.get_loc(loc)
        except KeyError:
            continue

        end_time = event["t1"] if ("t1" in event and pd.notna(event["t1"])) else last_close_time
        if pd.isna(end_time):
            end_time = last_close_time
        if end_time > last_close_time:
            end_time = last_close_time
        end_pos = close.index.get_indexer([end_time], method="bfill")[0]

        start_pos = loc_pos + 1
        if start_pos > end_pos:
            continue

        path = close.iloc[start_pos : end_pos + 1]
        if len(path) < 1:
            continue

        entry_price = path.iloc[0]
        if len(path) > 1:
            path_ret = (path.iloc[1:] / entry_price - 1.0) * event.get("side", 1.0)
        else:
            path_ret = pd.Series(dtype=float)

        if ptSl[0] and ptSl[0] > 0 and not path_ret.empty:
            pt_level = ptSl[0] * event["trgt"]
            hit = path_ret[path_ret >= pt_level]
            if not hit.empty:
                out.loc[loc, "pt"] = hit.index[0]

        if ptSl[1] and ptSl[1] > 0 and not path_ret.empty:
            sl_level = -ptSl[1] * event["trgt"]
            hit = path_ret[path_ret <= sl_level]
            if not hit.empty:
                out.loc[loc, "sl"] = hit.index[0]

    out["pt"] = pd.to_datetime(out["pt"])
    out["sl"] = pd.to_datetime(out["sl"])
    return out

# getEvents ported
def getEvents(close: pd.Series, tEvents, ptSl, trgt: pd.Series, minRet: float, t1=False):
    if not isinstance(close, pd.Series):
        raise ValueError("close must be a pd.Series")
    tEvents = pd.Index(tEvents).sort_values()
    last_close_time = close.index[-1]

    trgt = trgt.reindex(tEvents)
    trgt = trgt.dropna()
    trgt = trgt[trgt > minRet]

    if t1 is False:
        t1_series = pd.Series(pd.NaT, index=tEvents, dtype="datetime64[ns]")
    else:
        if not isinstance(t1, pd.Series):
            raise ValueError("t1 must be a pd.Series aligned to tEvents or False")
        t1_series = t1.reindex(tEvents)

    t1_series = t1_series.clip(upper=last_close_time)

    if trgt.empty:
        return pd.DataFrame(columns=["t1", "t_hit", "trgt", "side"]) 

    events = pd.DataFrame(index=trgt.index)
    events["t1"] = t1_series.reindex(events.index)
    events["trgt"] = trgt
    events["side"] = 1.0

    barrier_hits = applyPtSlOnT1(close=close, events=events, ptSl=ptSl, molecule=events.index)

    candidates = pd.concat([events["t1"], barrier_hits["pt"], barrier_hits["sl"]], axis=1)
    candidates.columns = ["t1", "pt", "sl"]
    stacked = candidates.stack(dropna=True)
    if stacked.empty:
        events["t_hit"] = pd.NaT
    else:
        events["t_hit"] = stacked.groupby(level=0).min().reindex(events.index)

    return events

# Build events and compute tb_events
events = get_crossing_moving_averages_events(close)

ptSl = [1,1]
minRet = 0.0

# trgt: volatility proxy
trgt = close.pct_change().abs().rolling(window=50).mean()
trgt = trgt.reindex(events.index).bfill()

# no vertical barrier for simplicity
try:
    t1_series
except NameError:
    t1_series = pd.Series(pd.NaT, index=events.index, dtype='datetime64[ns]')

if events.empty:
    print('No crossing events found; nothing to validate')
    exit()

tb_events = getEvents(close=close, tEvents=events.index, ptSl=ptSl, trgt=trgt, minRet=minRet, t1=t1_series)
if tb_events.empty:
    print('No tb_events after applying minRet')
    exit()

# recompute barrier hits and t_hit
barrier_hits = applyPtSlOnT1(close=close, events=tb_events, ptSl=ptSl, molecule=tb_events.index)
tb_events = tb_events.join(barrier_hits)

# t_hit and barrier
tb_events['t_hit'] = tb_events[[ 't1', 'pt', 'sl']].min(axis=1)

def which_bar(row):
    if pd.isna(row['t_hit']):
        return 't1'
    if pd.notna(row.get('pt')) and row['t_hit'] == row['pt']:
        return 'pt'
    if pd.notna(row.get('sl')) and row['t_hit'] == row['sl']:
        return 'sl'
    return 't1'

tb_events['barrier'] = tb_events.apply(which_bar, axis=1)
# label: pt -> 1, else 0 (t1 counts negative)
tb_events['label'] = (tb_events['barrier'] == 'pt').astype(int)

# weights
starts = pd.to_datetime(tb_events.index)
ends = pd.to_datetime(tb_events['t_hit']).fillna(close.index[-1])
durations = (ends - starts).dt.total_seconds()
weights = []
for i in range(len(tb_events)):
    s_i = starts[i]
    e_i = ends[i]
    dur_i = durations.iloc[i]
    if dur_i <= 0 or pd.isna(e_i):
        weights.append(0.0)
        continue
    overlap = 0.0
    for j in range(len(tb_events)):
        if i == j:
            continue
        s_j = starts[j]
        e_j = ends[j]
        latest_start = max(s_i, s_j)
        earliest_end = min(e_i, e_j)
        delta = (earliest_end - latest_start).total_seconds()
        if delta > 0:
            overlap += delta
    overlap = min(overlap, dur_i)
    frac = overlap / dur_i if dur_i > 0 else 0.0
    weights.append(max(0.0, 1.0 - frac))

tb_events['weight'] = weights

# Find overlapping events indices
tb = tb_events.copy()
tb['start'] = starts
tb['end'] = ends
tb['duration_s'] = durations

overlap_idx = []
for i, row in tb.iterrows():
    s_i = row['start']; e_i = row['end']; dur = row['duration_s']
    if dur <= 0 or pd.isna(e_i):
        continue
    for j, rowj in tb.iterrows():
        if i == j:
            continue
        s_j = rowj['start']; e_j = rowj['end']
        latest = max(s_i, s_j); earliest = min(e_i, e_j)
        delta = (earliest - latest).total_seconds()
        if delta > 0:
            overlap_idx.append(i)
            break

overlap_idx = list(dict.fromkeys(overlap_idx))
print(f'Found {len(overlap_idx)} overlapping events; showing up to 5')
show_idx = overlap_idx[:5]

# Save plots
out_files = []
for idx in show_idx:
    s = tb.loc[idx,'start'] - pd.Timedelta(days=5)
    e = tb.loc[idx,'end'] + pd.Timedelta(days=5)
    window = close.loc[s:e]
    plt.figure(figsize=(10,3))
    plt.plot(window.index, window.values, label='close')
    plt.axvline(tb.loc[idx,'start'], color='black', linestyle='--', label='entry')
    plt.axvline(tb.loc[idx,'end'], color='red', linestyle='--', label='exit')
    plt.title('Event {} barrier={} weight={:.2f}'.format(idx, tb.loc[idx,'barrier'], tb.loc[idx,'weight']))
    plt.legend()
    fname = f"overlap_event_{str(idx).replace(':','_').replace(' ','_')}.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    out_files.append(fname)
    print('Saved', fname)

# Save summary table
summary = tb.loc[show_idx][['start','end','t1','pt','sl','t_hit','barrier','label','weight']]
summary.to_csv('overlap_summary.csv')
print('Saved overlap_summary.csv')

# Print a small table to stdout
print(summary.head())

print('Done')
