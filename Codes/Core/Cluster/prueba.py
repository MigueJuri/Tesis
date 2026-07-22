import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.feature_selection import mutual_info_regression
import warnings
import requests

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

def get_log_returns(tickers, start_date, end_date):
    """
    Fetches adjusted close prices and computes log returns, enforcing stationarity.
    """
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise ValueError("yfinance returned no data for the requested date range.")

    if isinstance(raw.columns, pd.MultiIndex):
        if 'Adj Close' not in raw.columns.get_level_values(0):
            raise ValueError("'Adj Close' not found in downloaded data.")
        price_data = raw['Adj Close']
    else:
        price_data = raw.rename(columns={'Adj Close': tickers[0] if isinstance(tickers, list) else tickers})
        if 'Adj Close' in price_data.columns:
            price_data = price_data[['Adj Close']]

    if isinstance(price_data, pd.Series):
        ticker_name = tickers[0] if isinstance(tickers, list) else str(tickers)
        price_data = price_data.to_frame(name=ticker_name)

    valid_cols = [col for col in price_data.columns if price_data[col].notna().sum() > 2]
    
    if len(valid_cols) < 2:
        raise ValueError(f"Need at least 2 valid tickers to compute MI. Found: {valid_cols}")

    price_data = price_data[valid_cols]

    # Compute log returns to ensure weak stationarity of the observables
    log_returns = np.log(price_data / price_data.shift(1)).dropna(how='any')

    if log_returns.empty:
        raise ValueError("No overlapping return observations after cleaning.")

    return log_returns

# --- Execution Pipeline ---
#SPY
#tickers = ['NVDA','AAPL','MSFT','AMZN','GOOG','META','AVGO','TSLA','BRK.B','WMT','LLY','JPM','XOM','V','JNJ','MA']

def get_sp500_tickers_by_sector(target_sector: str = "Energy") -> list:
    """
    Scrapes the S&P 500 constituents and filters them by GICS Sector.
    This restricts our state-space to a highly coupled sub-manifold.
    
    Args:
        target_sector: GICS Sector name to filter by, or "All" to get all S&P 500 tickers
    
    Returns:
        List of tickers in the specified sector(s)
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    # Required header to bypass CDN bot-protection
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise ConnectionError(f"HTTP Error: {response.status_code}")
        
    # Read the first table into a DataFrame
    df = pd.read_html(response.text)[0]
    
    # If "All" is requested, return all tickers without filtering
    if target_sector.lower() == "all":
        tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
        return tickers
    
    # Check available sectors to prevent silent failures
    available_sectors = df['GICS Sector'].unique()
    if target_sector not in available_sectors:
        raise ValueError(f"Sector '{target_sector}' not found. Valid sectors are: {available_sectors}")
    
    # Apply the topological constraint (boolean filtering)
    sector_df = df[df['GICS Sector'] == target_sector]
    
    # Format tickers for yfinance compatibility
    tickers = sector_df['Symbol'].str.replace('.', '-', regex=False).tolist()
    
    return tickers

# Create your large ticker vector
tickers_sp500 = get_sp500_tickers_by_sector("All")

# Note: Changed 'BRK.' to 'BRK.B' as yfinance requires the exact ticker
returns_df = get_log_returns(tickers_sp500, '2020-01-01', '2026-01-03')

print("Calculating Gaussian MI Matrix...")
mi_matrix_fast = get_gaussian_mi_matrix(returns_df)

# --- Extract the Top 10 Pairs ---
pairs = []
for i in range(len(mi_matrix_fast.columns)):
    for j in range(i + 1, len(mi_matrix_fast.columns)):
        val = mi_matrix_fast.iloc[i, j]
        if not np.isnan(val):
            pairs.append((mi_matrix_fast.index[i], mi_matrix_fast.columns[j], val))

# Sort descending by Mutual Information
top_10_fast_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:10]

print("\nTop 10 Pairs by Gaussian Mutual Information (Ready for KSG validation):")
for rank, (t1, t2, mi_val) in enumerate(top_10_fast_pairs, 1):
    print(f"{rank}. {t1} - {t2} : {mi_val:.4f} nats")