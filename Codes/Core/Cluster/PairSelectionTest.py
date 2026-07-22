import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import warnings
import requests
from io import StringIO
from pathlib import Path

CSV_FILENAME = "sp500_data_2015-01-12_to_2026-01-02.csv"
CSV_PATH = Path(__file__).resolve().parent / CSV_FILENAME

def get_gaussian_mi_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the Pairwise Mutual Information matrix under a Gaussian assumption.
    Complexity is strictly O(N^2) correlation, avoiding expensive neighbor searches.
    """
    if returns_df.empty:
        raise ValueError("Returns DataFrame is empty.")
        
    # 1. Compute the Pearson correlation matrix (Highly optimized)
    rho_matrix = returns_df.corr(method='pearson')
    
    # 2. Bound the correlation to avoid mathematically undefined log(0) at rho=1 (the diagonal)
    rho_clipped = rho_matrix.clip(lower=-0.99999, upper=0.99999)
    
    # 3. Apply the Gaussian Mutual Information transformation
    mi_matrix = -0.5 * np.log(1 - rho_clipped**2)
    
    # 4. Remove self-information (the diagonal)
    np.fill_diagonal(mi_matrix.values, np.nan)
    
    return mi_matrix


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

def calculate_mi_matrix_ksg(returns_df: pd.DataFrame, k_neighbors: int = 3) -> pd.DataFrame:
    """
    Computes the pairwise Mutual Information matrix using the Kraskov-Stogbauer-Grassberger (KSG)
    k-Nearest Neighbors estimator. Treats the returns as continuous observables in a complex 
    adaptive system, avoiding the discretization bias of histograms.
    """
    if returns_df is None or returns_df.empty:
        raise ValueError("returns_df is empty. No data available to compute MI.")

    # Keep rows with at least one valid return. Pairwise cleaning is applied per asset pair.
    clean_returns = returns_df.dropna(how='all')
    if clean_returns.empty:
        raise ValueError("No valid return history after NaN cleaning.")

    assets = clean_returns.columns
    n_assets = len(assets)

    mi_matrix = np.full((n_assets, n_assets), np.nan)
    
    # Suppress warnings regarding exact zero-variance ties in finite samples
    warnings.filterwarnings("ignore", category=UserWarning)

    print(f"Calculating KSG Mutual Information matrix for {n_assets} assets...")
    
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            pair_data = clean_returns.iloc[:, [i, j]].dropna(how='any')

            # Need enough paired observations for k-NN estimation.
            if len(pair_data) <= k_neighbors + 1:
                continue

            x = pair_data.iloc[:, 0].to_numpy()
            y = pair_data.iloc[:, 1].to_numpy()

            # Guard against degenerate (zero-variance) series
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            
            # Reshape X for sklearn's expected feature matrix format
            X = x.reshape(-1, 1)
            Y = y.reshape(-1, 1)
            
            # KSG estimation. We use a fixed random_state because the algorithm 
            # adds a microscopic jitter (~1e-10) to break exact duplicate ties.
            mi_xy = mutual_info_regression(X, y, n_neighbors=k_neighbors, random_state=42)[0]
            
            # KSG is theoretically symmetric, but finite-sample k-NN searches can 
            # induce microscopic asymmetries. We enforce strict physical symmetry.
            mi_yx = mutual_info_regression(Y, x, n_neighbors=k_neighbors, random_state=42)[0]
            mi_symmetric = (mi_xy + mi_yx) / 2.0
            
            mi_matrix[i, j] = mi_symmetric
            mi_matrix[j, i] = mi_symmetric

    return pd.DataFrame(mi_matrix, index=assets, columns=assets)



# --- Execution Pipeline ---
#SPY
#tickers = ['NVDA','AAPL','MSFT','AMZN','GOOG','META','AVGO','TSLA','BRK.B','WMT','LLY','JPM','XOM','V','JNJ','MA']

downloaded_data = load_data_as_downloaded(CSV_PATH)
close_prices = downloaded_data.xs('Close', axis=1, level=0)
returns_df = close_prices.pct_change(fill_method=None).dropna(how='all')

# Vectorized close-price matrix (rows=dates, cols=tickers).
data_vector = close_prices.to_numpy()

print(f"Loaded data from CSV: {downloaded_data.shape[0]} dates, {close_prices.shape[1]} tickers")

print("Calculating Gaussian MI Matrix...")
# We use k=3 as it provides the optimal bias-variance tradeoff in financial time series
mi_matrix = calculate_mi_matrix_ksg(returns_df, k_neighbors=3)

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