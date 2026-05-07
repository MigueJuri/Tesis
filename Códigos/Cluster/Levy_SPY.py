import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# --- 1. Carga de Datos Estáticos ---
def load_local_data(filename):
    # Leer el CSV generado en la etapa 1
    data = pd.read_csv(filename, index_col='Date', parse_dates=True)
    prices = data['Adj Close']
    
    # Log-retornos son aditivos, esenciales para GBM
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return prices, log_returns

# Leer desde el archivo estático en lugar de yfinance
data, log_ret = load_local_data('spy_data.csv')
# Setup x-axis range for plotting
x = np.linspace(-0.1, 0.1, 500)

# Empirical density estimation using KDE
kde = stats.gaussian_kde(log_ret, bw_method='scott')
empirical_density = kde(x)

# Fit distributions
mu, std = stats.norm.fit(log_ret)
print(f"Ajustando Lévy estable...")
# params_levy = stats.levy_stable.fit(log_ret)
# alpha, beta, loc, scale = params_levy
# print(f"Lévy: α={alpha:.3f}, β={beta:.3f}, loc={loc:.5f}, scale={scale:.5f}")

# Calculate theoretical densities
normal_density = stats.norm.pdf(x, mu, std)
# levy_density = stats.levy_stable.pdf(x, *params_levy)

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))

ax.scatter(x, empirical_density, s=10, alpha=0.7, color='green', 
           label='Densidad Empírica (KDE)', zorder=3)
ax.plot(x, normal_density, 'k-', linewidth=2, 
        label=f'Normal (μ={mu:.4f}, σ={std:.4f})', zorder=2)
# ax.plot(x, levy_density, 'r-', linewidth=2, 
#         label=f'Lévy Estable (α={alpha:.3f}, β={beta:.3f})', zorder=1)

ax.set_yscale('log')
ax.set_xlim(-0.1, 0.1)
ax.set_ylim(bottom=1e-2)
ax.set_xlabel('Retorno Logarítmico')
ax.set_ylabel('Densidad (log)')
ax.set_title('Distribución de Retornos Logarítmicos: Empírica vs Teórica')
ax.legend(loc='best')
ax.grid(True, alpha=0.3, which="both", linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()