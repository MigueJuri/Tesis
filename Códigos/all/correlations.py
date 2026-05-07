import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Fetch Data
tickers = ['MSFT', 'INTC', 'GLD']
data = yf.download(tickers, start='1994-01-01', end='2024-12-31', interval='1mo', auto_adjust=False)['Adj Close']

# 2. Calculate PRICE Correlation (The "Trend" view)
price_corr = data.corr()

# 3. Calculate RETURNS Correlation (The "market" view)
# We use pct_change() to get the monthly movements
returns = data.pct_change().dropna()
returns_corr = returns.corr()

# 4. Display Side-by-Side
print("--- PRICE Correlation (Long-term Trend) ---")
print(price_corr)
print("\n--- RETURNS Correlation (Monthly Co-movement) ---")
print(returns_corr)

# 5. Visual Comparison
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(price_corr, annot=True, cmap='Greens', vmin=0, vmax=1, ax=ax[0])
ax[0].set_title('Price Correlation\n(Do they trend up together?)')

sns.heatmap(returns_corr, annot=True, cmap='Blues', vmin=0, vmax=1, ax=ax[1])
ax[1].set_title('Returns Correlation\n(Do they react together?)')

plt.show()