import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import warnings
import requests
from io import StringIO
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller 
from itertools import combinations
from pathlib import Path

CSV_FILENAME = "sp500_data_2015-01-12_to_2026-01-02.csv"
CSV_PATH = Path(__file__).resolve().parent / CSV_FILENAME

START_DATE = '2023-01-01'
END_DATE = '2026-01-02'


def load_data_as_downloaded(csv_path: str) -> pd.DataFrame:
    """Load a yfinance-style CSV and keep its two-level column layout."""
    data = pd.read_csv(csv_path, header=[0, 1], skiprows=[2])

    # First column is the date index in this exported format.
    date_col = data.columns[0]
    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    data = data.dropna(subset=[date_col]).set_index(date_col)
    data.index.name = "Date"

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data.sort_index(inplace=True)
    return data

def test_cointegration(y: pd.Series, x: pd.Series) -> tuple:
    """
    Tests for cointegration between two price series using the Engle-Granger approach.
    
    Steps:
    1. Run OLS regression: y = α + β*x
    2. Calculate residuals (spread): ε = y - (α + β*x)
    3. Test residuals for stationarity using ADF test
    
    Returns:
    - adf_stat: ADF test statistic (more negative = stronger cointegration)
    - p_value: p-value of the ADF test
    - beta: hedge ratio from OLS regression
    """
    # Remove NaN values
    valid_idx = ~(y.isna() | x.isna())
    y_clean = y[valid_idx].values.reshape(-1, 1)
    x_clean = x[valid_idx].values.reshape(-1, 1)
    
    # Run OLS regression
    model = LinearRegression()
    model.fit(x_clean, y_clean)
    beta = model.coef_[0][0]
    
    # Calculate residuals (spread)
    spread = y_clean.flatten() - (model.predict(x_clean).flatten())
    
    # Run ADF test on the spread
    adf_result = adfuller(spread, maxlag=1, regression='c', autolag=None)
    adf_stat = adf_result[0]
    p_value = adf_result[1]
    
    return adf_stat, p_value, beta

def find_cointegrated_pairs(prices: pd.DataFrame, max_pairs: int = 50) -> pd.DataFrame:
    """
    Tests all possible pairs for cointegration and returns the top N most cointegrated pairs.
    
    Args:
        prices: DataFrame of price series
        max_pairs: Number of top pairs to return
    
    Returns:
        DataFrame with columns: ticker1, ticker2, adf_stat, p_value, beta
    """
    tickers = prices.columns.tolist()
    n_tickers = len(tickers)
    n_pairs = n_tickers * (n_tickers - 1) // 2
    
    print(f"\nTesting {n_pairs} pairs for cointegration...")
    print("This may take several minutes...\n")
    
    results = []
    
    for i, (ticker1, ticker2) in enumerate(combinations(tickers, 2)):
        if (i + 1) % 1000 == 0:
            print(f"Progress: {i+1}/{n_pairs} pairs tested ({100*(i+1)/n_pairs:.1f}%)")
        
        try:
            # Test both directions and take the best
            adf1, pval1, beta1 = test_cointegration(prices[ticker1], prices[ticker2])
            adf2, pval2, beta2 = test_cointegration(prices[ticker2], prices[ticker1])
            
            # Use the direction with stronger cointegration (more negative ADF)
            if adf1 < adf2:
                results.append({
                    'ticker1': ticker1,
                    'ticker2': ticker2,
                    'adf_stat': adf1,
                    'p_value': pval1,
                    'beta': beta1
                })
            else:
                results.append({
                    'ticker1': ticker2,
                    'ticker2': ticker1,
                    'adf_stat': adf2,
                    'p_value': pval2,
                    'beta': beta2
                })
        except Exception as e:
            # Skip pairs that cause errors
            continue
    
    print(f"\nTesting complete. {len(results)} valid pairs found.")
    
    # Convert to DataFrame and sort by ADF statistic (most negative = most cointegrated)
    if not results:
        print("No pairs could be tested. Check that the price DataFrame has at least 2 columns.")
        return pd.DataFrame(columns=['ticker1', 'ticker2', 'adf_stat', 'p_value', 'beta'])
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('adf_stat', ascending=True).reset_index(drop=True)
    
    # Return top N pairs
    return results_df.head(max_pairs)


def load_tickers_from_file(file_path: str) -> list:
    """
    Loads tickers from a text file, one ticker per line.
    """
    with open(file_path, 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]
    return tickers

# Load tickers from a text file
TICKERS = load_tickers_from_file(r"G:\Mi unidad\2026\Tesis\Códigos\Data\tickers\sp500_tickers_Some.txt")

downloaded_data = load_data_as_downloaded(CSV_PATH)

# Handle both MultiIndex orientations produced by different yfinance versions:
#   (price_type, ticker)  -- older format: level 0 = 'Adj Close' / 'Close'
#   (ticker, price_type)  -- newer format: level 1 = 'Adj Close' / 'Close'
level0 = downloaded_data.columns.get_level_values(0).unique().tolist()
level1 = downloaded_data.columns.get_level_values(1).unique().tolist()

PRICE_KEYS = ('Adj Close', 'Close')
if any(k in level0 for k in PRICE_KEYS):
    # (price_type, ticker) orientation
    price_key = next(k for k in PRICE_KEYS if k in level0)
    price_data = downloaded_data[price_key]
elif any(k in level1 for k in PRICE_KEYS):
    # (ticker, price_type) orientation — swap levels first
    price_key = next(k for k in PRICE_KEYS if k in level1)
    price_data = downloaded_data.swaplevel(axis=1)[price_key]
else:
    raise KeyError(
        f"No price column found.\n  Level-0 values: {level0[:10]}\n  Level-1 values: {level1[:10]}"
    )

print(f"Using price key: '{price_key}'")
price_data

# Keep only tickers that are actually present in the CSV
available_tickers = [t for t in TICKERS if t in price_data.columns]
missing_tickers = set(TICKERS) - set(available_tickers)
if missing_tickers:
    print(f"Warning: {len(missing_tickers)} tickers not found in CSV (skipped): {sorted(missing_tickers)}")

data = price_data[available_tickers].dropna(how='all')
data = data.loc[START_DATE:END_DATE]


# Vectorized close-price matrix (rows=dates, cols=tickers).
data_vector = data.to_numpy()

print(f"Loaded data from CSV: {downloaded_data.shape[0]} dates, {data.shape[1]} tickers")

print("Calculating Gaussian MI Matrix...")
# We use k=3 as it provides the optimal bias-variance tradeoff in financial time series
mi_matrix = find_cointegrated_pairs(data, max_pairs=50)

labels = list(mi_matrix.columns)

# plt.figure(figsize=(10, 8))
# plt.imshow(mi_matrix, interpolation='nearest')
# plt.colorbar(label='Mutual Information (KSG Estimator)')
# plt.xticks(range(len(labels)), labels, rotation=45)
# plt.yticks(range(len(labels)), labels)
# plt.title('Non-Linear Asset Coupling (Continuous State-Space)')
# plt.tight_layout()
# plt.show()

# Extract and rank the couplings
mi_pairs = []
for i in range(len(mi_matrix)):
    for j in range(i + 1, len(mi_matrix)):
        mi_value = mi_matrix.iloc[i, j]
        if not np.isnan(mi_value):
            mi_pairs.append((mi_matrix.index[i], mi_matrix.columns[j], mi_value))

top_20_pairs = sorted(mi_pairs, key=lambda x: x[2], reverse=True)[:50]

print("Top 20 Most Coupled Pairs (KSG Continuous Mutual Information):")
for idx, (t1, t2, val) in enumerate(top_20_pairs, 1):
    print(f"{idx}. {t1} - {t2}: {val:.4f}")