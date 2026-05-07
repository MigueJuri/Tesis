import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data_from_csv(csv_file: str) -> pd.DataFrame:
    # yfinance CSV export uses a 2-row header for MultiIndex columns and a standalone index-name row.
    data = pd.read_csv(csv_file, header=[0, 1], skiprows=[2], index_col=0)

    data.index = pd.to_datetime(data.index)
    data.index.name = "Date"

    return data


script_dir = Path(__file__).resolve().parent
output_file = script_dir / "sp500_data_only_1993-01-29_to_2026-01-02.csv"

if not output_file.exists():
    raise FileNotFoundError(
        f"CSV not found: {output_file}. Place the file next to this script or provide an absolute path."
    )

data_from_csv = load_data_from_csv(output_file)

def get_weights_ffd(d, thres=1e-4):
    """Calculate FFD weights."""
    w = [1.]
    k = 1
    while abs(w[-1]) >= thres:
        w_new = -w[-1] / k * (d - k + 1)
        w.append(w_new)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series, d, thres=1e-4):
    w = get_weights_ffd(d, thres)
    width = len(w) - 1
    df = {}
    for name in series.columns:
        seriesF = series[[name]].ffill().dropna()  # Fix Bug 2
        df_ = pd.Series(dtype=float)
        for iloc1 in range(width, seriesF.shape[0]):
            loc0 = seriesF.index[iloc1 - width]
            loc1 = seriesF.index[iloc1]
            slice_ = seriesF.loc[loc0:loc1]
            if len(slice_) != len(w):             # Fix Bug 4
                continue
            if not np.isfinite(slice_[name]).all(): # Fix Bug 3
                continue
            df_[loc1] = np.dot(w.T, slice_)[0, 0]
        df[name] = df_.copy(deep=True)
    return pd.concat(df, axis=1)

log_prices = np.log(data_from_csv["Adj Close"][["SPY"]])
# --- 1. Define the Grid ---
# Assume `log_prices` is a pandas DataFrame with your 1990-2026 data
d_values = np.round(np.arange(0.1, 1.05, 0.1), 2)
# T in trading days: 1 yr, 2 yrs, 3 yrs, 4 yrs, 5 yrs, 10 yrs
T_values = [252]#, 504, 756, 1008, 1260, 2520] 
step_size = 21 # Sample a window every 1 month to save compute time

results_matrix = np.zeros((len(d_values), len(T_values)))

# --- 2. Compute Ensemble Stationarity ---
for i, d in enumerate(d_values):
    # Differentiate the whole series ONCE per d to save computation
    diff_series = frac_diff_ffd(log_prices, d=d).dropna().iloc[:, 0]
    
    for j, T in enumerate(T_values):
        adf_sum = 0
        total_windows = 0
        
        # Roll through the differentiated series
        for start_idx in range(0, len(diff_series) - T, step_size):
            window_data = diff_series.iloc[start_idx : start_idx + T]
            
            if window_data.std() < 1e-8:
                continue

            try:
                adf_stat, p_val, _, _, _, _ = adfuller(window_data)
                adf_sum += adf_stat
            except Exception as e:
                print(f"ERROR: {e}")

            total_windows += 1

        # ← print is HERE, outside the loop (one level back)
        if total_windows > 0:
            print(f"d={d}, T={T}: stat={adf_stat:.4f}, p={p_val:.6f}, windows={total_windows}")
        else:
            print(f"d={d}, T={T}: no valid windows for ADF")

        # Record the average ADF statistic for this (d, T)
        results_matrix[i, j] = adf_sum / total_windows if total_windows > 0 else 0
            
# --- 3. Plot the Heatmap ---
plt.figure(figsize=(10, 6))
sns.heatmap(results_matrix, annot=True, fmt=".2f", cmap="YlGnBu", 
            xticklabels=T_values, yticklabels=d_values)
plt.title(r"Ensemble Stationarity: ADF Pass Rate $\pi(T, d)$ (1990-2026)")
plt.xlabel("ADF Window Size $T$ (Trading Days)")
plt.ylabel("Fractional Differentiation Order $d$")
plt.show()