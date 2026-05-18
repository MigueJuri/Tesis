"""
LSTM Volatility Forecasting System — SPY
=========================================
Quant Professor Implementation | Licentiate Thesis Level

Corrections applied vs. original spec:
  [C1] Target: annualized (252/H) normalization, not (1/4)
  [C2] Parkinson: squared log-ratio inside the average, not outside
  [C3] LSTM dropout: explicit nn.Dropout layers between LSTM modules
  [C4] HAR-RV baseline added (Corsi 2009) — standard benchmark
  [C5] QLIKE loss, Mincer-Zarnowitz, Diebold-Mariano added
  [C6] Backtest base signal: transparent momentum (not undefined placeholder)
  [C7] Weight initialization: Xavier + orthogonal + forget bias = 1

References:
  Corsi (2009). A Simple Approximate Long-Memory Model of Realized Volatility.
  Patton (2011). Volatility Forecast Comparison Using Imperfect Volatility Proxies.
  Diebold & Mariano (1995). Comparing Predictive Accuracy.
  Mincer & Zarnowitz (1969). The Evaluation of Economic Forecasts.
  Jozefowicz et al. (2015). An Empirical Evaluation of Recurrent Network Architectures.
  Parkinson (1980). The Extreme Value Method for Estimating the Variance of the Rate of Return.
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datetime import datetime
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# ============================================================
# REPRODUCIBILITY
# ============================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# SECTION 0: CONFIGURATION
# ============================================================
CONFIG = {
    # Data
    'ticker':           'SPY',
    'start_date':       '2000-01-01',
    # Chronological splits
    'train_end':        '2018-01-01',
    'val_end':          '2023-01-01',
    # Feature / sequence settings
    'lookback':         30,
    'forecast_horizon': 5,
    'features': [
        'rv_daily', 'rv_weekly', 'rv_monthly',  # HAR components
        'log_return',                            # price momentum proxy
        'parkinson_vol',                         # high-low estimator
        'rsi_14',                                # market microstructure
        'rel_volume',                            # liquidity proxy
        'intraday_range',                        # intraday volatility proxy
    ],
    # LSTM architecture
    'hidden1':    32,
    'hidden2':    16,
    'dropout':    0.4,
    # Training
    'batch_size': 32,
    'lr':         1e-3,
    'weight_decay': 1e-5,
    'max_epochs': 200,
    'patience':   15,
    'grad_clip':  1.0,
    # Backtest
    'cost_bps':   0,#5,   # one-way transaction cost in basis points
    # Outputs
    'model_path':   'lstm_vol_model.pt',
    'scaler_path':  'scaler.pkl',
    'metrics_path': 'metrics.csv',
    'report_path':  'report.md',
    'config_path':  'model_config.json',
    'plot_path':    'lstm_vol_report.png',
}


# ============================================================
# SECTION 1: DATA ACQUISITION
# ============================================================

def download_data(ticker: str, start: str) -> pd.DataFrame:
    """
    Download OHLCV data from yfinance. auto_adjust=True handles
    dividend and split adjustments.

    Returns a clean DataFrame with columns: Open, High, Low, Close, Volume.
    All prices are adjusted (split- and dividend-corrected).
    """
    print(f"\n[DATA] Downloading {ticker} from {start}…")
    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)

    # Flatten MultiIndex columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()

    # Validate: missing values
    n_nan = raw.isnull().sum().sum()
    if n_nan > 0:
        print(f"  WARNING: {n_nan} NaN values. Forward-filling.")
        raw = raw.ffill().dropna()

    # Validate: non-positive prices (data error)
    if (raw[['Open', 'High', 'Low', 'Close']] <= 0).any().any():
        raise ValueError("Non-positive price detected — data integrity error.")

    # Flag extreme moves (>15%) for awareness, but retain (COVID etc.)
    log_ret = np.log(raw['Close'] / raw['Close'].shift(1)).dropna()
    n_extreme = (np.abs(log_ret) > 0.15).sum()
    if n_extreme > 0:
        print(f"  INFO: {n_extreme} daily moves >15% flagged (retained).")

    print(f"  Shape: {raw.shape} | "
          f"{raw.index[0].date()} → {raw.index[-1].date()}")
    return raw


# ============================================================
# SECTION 2: FEATURE ENGINEERING
# ============================================================

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    All features are strictly causal: at time t, only data from [0, t] is used.

    Mathematical definitions
    ------------------------
    r_t          = log(C_t / C_{t-1})                           log-return
    RV_t^d       = r_t^2                                        daily RV proxy
    RV_t^w       = (1/5)  * sum_{i=1}^{5}  r_{t-i}^2          weekly HAR component
    RV_t^m       = (1/22) * sum_{i=1}^{22} r_{t-i}^2          monthly HAR component

    Parkinson (1980) [C2 — corrected]:
        sigma_P^2 = (1 / (4 ln 2)) * E[(log(H/L))^2]
        => sigma_P = sqrt( (1/(4 ln 2)) * rolling_mean( (log(H/L))^2, n ) )
        NOTE: The spec erroneously wrote mean(log(H/L)) / (4 ln 2), which
              is dimensionally and statistically incorrect.

    Garman-Klass (1980):
        sigma_GK^2 = 0.5*(log(H/L))^2 - (2 ln 2 - 1)*(log(C/O))^2

    RSI (Wilder, normalized to [0,1]):
        RSI = 100 - 100/(1 + RS),  RS = EMA(gains)/EMA(losses)
        rsi_14_norm = RSI / 100

    rel_volume   = V_t / MA20(V_t)
    intraday_range = (H_t - L_t) / C_{t-1}   (normalized by prior close)
    """
    feat = pd.DataFrame(index=df.index)

    # --- Log return ---
    feat['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    # --- HAR-RV components (Corsi 2009) ---
    rv_sq = feat['log_return'] ** 2
    feat['rv_daily']   = rv_sq
    feat['rv_weekly']  = rv_sq.rolling(5).mean()
    feat['rv_monthly'] = rv_sq.rolling(22).mean()

    # Rolling close-to-close vol (annualized, 20-day)
    feat['vol_20'] = feat['log_return'].rolling(20).std() * np.sqrt(252)

    # --- Parkinson (1980) high-low estimator [C2: corrected] ---
    hl_log_sq = np.log(df['High'] / df['Low']) ** 2   # square first
    feat['parkinson_vol'] = np.sqrt(
        hl_log_sq.rolling(20).mean() / (4.0 * np.log(2))
    )

    # --- Garman-Klass (1980) estimator ---
    hl2 = 0.5 * np.log(df['High'] / df['Low']) ** 2
    co2 = (2.0 * np.log(2) - 1.0) * np.log(df['Close'] / df['Open']) ** 2
    feat['gk_vol'] = np.sqrt((hl2 - co2).clip(lower=0).rolling(20).mean())

    # --- RSI (14-day, Wilder smoothing approximated by rolling mean) ---
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-10)
    feat['rsi_14'] = (100.0 - 100.0 / (1.0 + rs)) / 100.0   # normalized to [0,1]

    # --- Volume features ---
    vol_ma20 = df['Volume'].rolling(20).mean()
    feat['rel_volume']   = df['Volume'] / (vol_ma20 + 1e-10)
    feat['volume_trend'] = (df['Volume'].rolling(5).mean() /
                            (vol_ma20 + 1e-10))

    # --- Intraday range (normalized, causal: use prior close) ---
    feat['intraday_range'] = (df['High'] - df['Low']) / df['Close'].shift(1)

    # --- Close-to-open gap ---
    feat['gap'] = np.abs(df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

    # --- Rolling skewness (5-day) ---
    feat['skew_5']    = feat['log_return'].rolling(5).skew()
    feat['mean_ret_5'] = feat['log_return'].rolling(5).mean()

    return feat


def compute_target(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """
    5-day forward annualized realized volatility. [C1: corrected normalization]

    Target at time t:
        sigma_{t:t+H} = sqrt( (252/H) * sum_{i=1}^{H} r_{t+i}^2 )

    This is FORWARD-LOOKING (uses future returns) and only for target construction.
    The shift(-horizon) alignment ensures no look-ahead in features.

    Note: target[t] uses returns from t+1 to t+H (rolling sum shifted back H days).
    When aligned with sequences of lookback=L, the last feature day is t=i+L-1,
    so the effective prediction is vol from day i+L to i+L+H-1.
    """
    log_ret = np.log(df['Close'] / df['Close'].shift(1))
    rv_sq   = log_ret ** 2
    # Sum of next H squared returns, annualized
    target  = np.sqrt((252.0 / horizon) * rv_sq.rolling(horizon).sum().shift(-horizon))
    return target.rename('target_vol')


# ============================================================
# SECTION 3: HAR-RV BASELINE  [C4 — added]
# ============================================================

class HARRVModel:
    """
    Heterogeneous Autoregressive Realized Volatility model (Corsi 2009).

    Specification:
        RV_t = beta_0 + beta_d * RV_{t-1}^{(d)}
                      + beta_w * RV_{t-1}^{(w)}
                      + beta_m * RV_{t-1}^{(m)}  + epsilon_t

    where RV_{t-1}^{(d)}, RV_{t-1}^{(w)}, RV_{t-1}^{(m)} are the daily,
    5-day-average, and 22-day-average realized variance components.

    Estimated by OLS. Consistently competitive with ML models on daily data;
    serves as the primary benchmark for DM-test comparison.
    """
    HAR_COLS = ['rv_daily', 'rv_weekly', 'rv_monthly']

    def __init__(self):
        self.model_   = None
        self.ols_res_ = None

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> 'HARRVModel':
        X_ols = sm.add_constant(X_train[self.HAR_COLS].values)
        self.ols_res_ = sm.OLS(y_train, X_ols).fit(
            cov_type='HAC', cov_kwds={'maxlags': 10})
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_ols = sm.add_constant(X[self.HAR_COLS].values)
        return np.clip(self.ols_res_.predict(X_ols), a_min=0, a_max=None)

    def summary(self) -> str:
        return str(self.ols_res_.summary())


# ============================================================
# SECTION 4: LSTM MODEL  [C3, C7 — corrected]
# ============================================================

class VolatilityLSTM(nn.Module):
    """
    Two-layer stacked LSTM for volatility forecasting.

    Architecture
    ------------
    Input  → [batch, lookback, n_features]
    LSTM-1 → hidden1 (no built-in dropout; single layer)
    Dropout(p)                              ← explicit [C3]
    LSTM-2 → hidden2
    Dropout(p)
    Linear(hidden2, 16) + ReLU
    Linear(16, 1) + Softplus             ← guarantees sigma > 0

    Softplus: f(x) = log(1 + e^x)
    Guarantees positivity while remaining differentiable at 0.
    Unlike ReLU, Softplus has no dead neuron problem for regression targets.

    Weight Initialization [C7]
    --------------------------
    - Input weights (W_ih): Xavier uniform
    - Recurrent weights (W_hh): Orthogonal (preserves gradient norms)
    - Biases: zero-initialized except forget gate bias = 1.0
      (Jozefowicz et al. 2015: reduces vanishing gradient at initialization)
    - Linear layers: Xavier uniform
    """

    def __init__(self, n_features: int, hidden1: int = 64, hidden2: int = 32,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm1    = nn.LSTM(n_features, hidden1, batch_first=True, num_layers=1)
        self.drop1    = nn.Dropout(p=dropout)            # [C3] explicit dropout
        self.lstm2    = nn.LSTM(hidden1,    hidden2, batch_first=True, num_layers=1)
        self.drop2    = nn.Dropout(p=dropout)
        self.fc1      = nn.Linear(hidden2, 16)
        self.relu     = nn.ReLU()
        self.fc2      = nn.Linear(16, 1)
        self.softplus = nn.Softplus()

        self._init_weights()

    def _init_weights(self):  # [C7]
        for lstm in (self.lstm1, self.lstm2):
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
                    # Set forget gate bias to 1 (units n//4 : n//2 in 4-gate layout)
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)
        for fc in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, lookback, n_features]
        Returns: [batch] — predicted annualized vol (positive)
        """
        h1, _       = self.lstm1(x)               # [batch, T, hidden1]
        h1          = self.drop1(h1)
        h2, _       = self.lstm2(h1)               # [batch, T, hidden2]
        last        = self.drop2(h2[:, -1, :])     # [batch, hidden2]
        out         = self.relu(self.fc1(last))    # [batch, 16]
        vol_pred    = self.softplus(self.fc2(out)) # [batch, 1]
        return vol_pred.squeeze(-1)                 # [batch]


# ============================================================
# SECTION 5: SEQUENCE CONSTRUCTION
# ============================================================

def make_sequences(X: np.ndarray, y: np.ndarray,
                   lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window sequence construction.

    Sequence i: features X[i : i+lookback], target y[i+lookback]

    The last observed day in sequence i is index i+lookback-1 (day t).
    The target y[i+lookback] is the realized vol from t+1 to t+H.
    This is causal: no future feature data is included.
    """
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i : i + lookback])
        ys.append(y[i + lookback])
    return (np.array(Xs, dtype=np.float32),
            np.array(ys, dtype=np.float32))


# ============================================================
# SECTION 6: TRAINING PIPELINE
# ============================================================

class QLIKELoss(nn.Module):
    """
    QLIKE loss function for volatility forecasting, robust to proxy noise.
    L = E[sigma^2 / sigma_hat^2 - log(sigma^2/sigma_hat^2) - 1]
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        ratio = (y_true ** 2) / (y_pred ** 2 + self.eps)
        loss = ratio - torch.log(ratio) - 1.0
        return torch.mean(loss)

def train_lstm(model: VolatilityLSTM,
               train_loader: DataLoader,
               X_val_t: torch.Tensor,
               y_val_t: torch.Tensor,
               config: dict) -> dict:
    """
    Training loop with:
    - Adam optimizer + weight decay (L2 regularization)
    - ReduceLROnPlateau scheduler
    - Gradient clipping (max_norm = config['grad_clip'])
    - Early stopping (patience = config['patience'])
    - Best-weights checkpointing
    - Gradient clip frequency monitoring (flag if >5% of batches)
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['lr'],
        weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6)
    criterion = QLIKELoss()

    model.to(DEVICE)
    X_val_t = X_val_t.to(DEVICE)
    y_val_t = y_val_t.to(DEVICE)

    history = {'train_loss': [], 'val_loss': []}
    best_val      = float('inf')
    patience_ctr  = 0
    clip_count    = 0
    total_batches = 0

    for epoch in range(config['max_epochs']):
        model.train()
        epoch_loss, n_batches = 0.0, 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(Xb), yb)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            if norm > config['grad_clip']:
                clip_count += 1
            total_batches += 1
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        avg_train = epoch_loss / n_batches
        history['train_loss'].append(avg_train)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val     = val_loss
            patience_ctr = 0
            torch.save(model.state_dict(), config['model_path'])
        else:
            patience_ctr += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d} | "
                  f"train: {avg_train:.6f} (QLIKE) | val: {val_loss:.6f} (QLIKE)")

        if patience_ctr >= config['patience']:
            print(f"  Early stopping at epoch {epoch+1}. "
                  f"Best val QLIKE loss: {best_val:.6f}")
            break

    clip_pct = 100.0 * clip_count / max(total_batches, 1)
    print(f"  Gradient clipping triggered: {clip_pct:.1f}% of batches.")
    if clip_pct > 5.0:
        print("  *** RED FLAG: Gradient clipping >5%. Consider reducing LR. ***")

    model.load_state_dict(torch.load(config['model_path'], map_location=DEVICE))
    model.eval()
    return history


# ============================================================
# SECTION 7: EVALUATION  [C5 — QLIKE, MZ, DM added]
# ============================================================

def evaluate_forecasts(y_true: np.ndarray, y_pred: np.ndarray,
                       label: str = 'Model') -> dict:
    """
    Comprehensive volatility forecast evaluation.

    Metrics
    -------
    R²         : Coefficient of determination (OLS sense)
    RMSE       : Root mean squared error (annualized vol units)
    MAE        : Mean absolute error
    Spearman r : Rank correlation (robust to nonlinearity)
    Dir. Acc.  : Directional accuracy — above/below median split
    QLIKE      : Patton (2011) robust loss under proxy noise
                 L = E[sigma^2 / sigma_hat^2 - log(sigma^2/sigma_hat^2) - 1]
    MZ alpha   : Mincer-Zarnowitz intercept (H0: alpha = 0)
    MZ beta    : Mincer-Zarnowitz slope    (H0: beta  = 1)
    MZ F p-val : Joint test H0: alpha=0 AND beta=1 (unbiasedness)
    """
    mask   = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true = y_true[mask];  y_pred = y_pred[mask]

    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)

    sp_r, sp_p = stats.spearmanr(y_pred, y_true)

    # Directional accuracy (above/below cross-sectional median)
    dir_acc = np.mean((y_pred > np.median(y_pred)) == (y_true > np.median(y_true)))
    dir_test = stats.binomtest(int(round(dir_acc * len(y_true))), len(y_true), 0.5)
    dir_p    = float(getattr(dir_test, 'pvalue', dir_test))

    # QLIKE loss [C5]
    eps   = 1e-8
    ratio = y_true ** 2 / (y_pred ** 2 + eps)
    qlike = float(np.mean(ratio - np.log(ratio) - 1.0))

    # Mincer-Zarnowitz regression [C5]
    X_mz     = sm.add_constant(y_pred)
    mz_res   = sm.OLS(y_true, X_mz).fit(
        cov_type='HAC', cov_kwds={'maxlags': 10})
    # Joint F-test: alpha=0, beta=1  =>  R * params = r
    R   = np.eye(2)
    r   = np.array([0.0, 1.0])
    mz_f = mz_res.f_test((R, r))

    metrics = {
        'model':        label,
        'R2':           float(r2),
        'RMSE':         float(rmse),
        'MAE':          float(mae),
        'Spearman_r':   float(sp_r),
        'Spearman_p':   float(sp_p),
        'Dir_Acc':      float(dir_acc),
        'Dir_p':        dir_p,
        'QLIKE':        qlike,
        'MZ_alpha':     float(mz_res.params[0]),
        'MZ_beta':      float(mz_res.params[1]),
        'MZ_F_pval':    float(mz_f.pvalue),
        'N':            int(len(y_true)),
    }

    bar = '-' * 44
    print(f"\n{bar}")
    print(f"  {label}")
    print(f"{bar}")
    print(f"  R²:            {r2:+.4f}   (target: > 0.08)")
    print(f"  RMSE:          {rmse*100:.4f}%  (ann. vol units)")
    print(f"  MAE:           {mae*100:.4f}%")
    print(f"  Spearman r:    {sp_r:.4f}   (p = {sp_p:.4f})")
    print(f"  Dir. Accuracy: {dir_acc:.4f}   (p = {dir_p:.4f})")
    print(f"  QLIKE:         {qlike:.6f}")
    print(f"  MZ alpha:      {mz_res.params[0]:.4f}   (H0: = 0)")
    print(f"  MZ beta:       {mz_res.params[1]:.4f}   (H0: = 1)")
    print(f"  MZ F p-value:  {float(mz_f.pvalue):.4f}   (H0: unbiased)")
    return metrics


def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray, h: int = 5) -> dict:
    """
    Diebold-Mariano (1995) test for equal predictive accuracy. [C5]

    H0: Equal MSE-based predictive accuracy.
    Test statistic under Newey-West HAC standard errors with h-1 lags.

    d_t = e1_t^2 - e2_t^2   (loss differential)
    DM  = d_bar / sqrt(HAC_var(d) / T)

    Negative DM => model 1 (LSTM) more accurate than model 2 (HAR).
    """
    d    = e1 ** 2 - e2 ** 2
    d_bar = np.mean(d)
    T    = len(d)

    # Newey-West HAC variance with h-1 lags
    gamma0  = np.var(d, ddof=0)
    hac_var = gamma0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1]
        hac_var += 2.0 * (1.0 - k / h) * gamma_k

    se     = np.sqrt(max(hac_var, 1e-12) / T)
    dm_stat = d_bar / se
    dm_pval = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))

    direction = "LSTM more accurate" if dm_stat < 0 else "HAR-RV more accurate"
    sig = "SIGNIFICANT" if dm_pval < 0.05 else "not significant"

    print(f"\n[DM TEST] LSTM vs HAR-RV: "
          f"stat = {dm_stat:.4f}, p = {dm_pval:.4f}  [{sig}]")
    print(f"  ({direction} under MSE criterion)")

    return {'DM_stat': float(dm_stat), 'DM_pvalue': float(dm_pval)}


# ============================================================
# SECTION 8: BACKTEST  [C6 — transparent base signal]
# ============================================================

def run_vol_timing_backtest(actual_log_returns: pd.Series,
                            pred_vol: np.ndarray,
                            cost_bps: float = 5.0) -> dict:
    """
    Vol-timing strategy: scale a momentum position by inverse predicted vol.

    Base signal: sign of trailing 63-day (≈3-month) return.
    This is a well-documented, independent, transparent signal. [C6]

    Position:
        pos_vol(t)  = clip( signal(t) / (sigma_hat(t) / sigma_bar), -1, 1 )
        pos_base(t) = signal(t)   (unscaled, for comparison)

    Transaction cost: cost_bps * |delta_pos| (one-way, per unit notional)

    Returns
    -------
    Sharpe, MaxDD, Calmar for both strategies, and the improvement delta.
    Annualization: sqrt(252) factor applied throughout.
    """
    n   = len(pred_vol)
    ret = actual_log_returns.values[-n:]

    # Base signal: trailing 63-day momentum
    mom_win = min(63, n // 4)
    sig = np.sign(
        pd.Series(ret).rolling(mom_win).mean().fillna(0).values
    )
    sig[np.isnan(sig)] = 0.0

    sigma_bar = np.mean(pred_vol) + 1e-8
    vol_scale = pred_vol / sigma_bar

    pos_vol  = np.clip(sig / (vol_scale + 1e-8), -1.0, 1.0)
    pos_base = np.clip(sig, -1.0, 1.0)

    cost   = cost_bps * 1e-4
    tc_vol  = cost * np.abs(np.diff(np.r_[0.0, pos_vol]))
    tc_base = cost * np.abs(np.diff(np.r_[0.0, pos_base]))

    ret_vol  = pos_vol  * ret - tc_vol
    ret_base = pos_base * ret - tc_base

    def sharpe(r):
        return np.sqrt(252) * np.mean(r) / (np.std(r) + 1e-10)

    def max_drawdown(r):
        eq  = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(eq)
        return float(np.min((eq - peak) / (peak + 1e-10)))

    def calmar(r):
        mdd = abs(max_drawdown(r))
        return np.sqrt(252) * np.mean(r) / (mdd + 1e-10)

    return {
        'ret_vol':            ret_vol,
        'ret_base':           ret_base,
        'sharpe_vol':         sharpe(ret_vol),
        'sharpe_base':        sharpe(ret_base),
        'maxdd_vol':          max_drawdown(ret_vol),
        'maxdd_base':         max_drawdown(ret_base),
        'calmar_vol':         calmar(ret_vol),
        'calmar_base':        calmar(ret_base),
        'sharpe_improvement': sharpe(ret_vol) - sharpe(ret_base),
    }


# ============================================================
# SECTION 9: VISUALIZATION
# ============================================================

def generate_plots(y_true: np.ndarray, y_pred_lstm: np.ndarray,
                   y_pred_har: np.ndarray, test_dates: pd.DatetimeIndex,
                   history: dict, bt: dict, config: dict) -> None:
    """
    Eight-panel diagnostic figure:
      [0,0] Training / validation loss curves
      [0,1] Scatter: predicted vs actual vol (test)
      [1,:] Time series: actual, LSTM, HAR-RV over test period
      [2,0] Residual histogram with normal overlay
      [2,1] Backtest equity curves
      [3,0] Q-Q plot of LSTM residuals
      [3,1] Rank correlation scatter (LSTM vs HAR)
    """
    matplotlib.rcParams.update({'font.family': 'monospace'})

    BG   = '#0d1117'
    CELL = '#161b22'
    GRID = '#21262d'
    TEXT = '#c9d1d9'
    C = {
        'actual': '#58a6ff', 'lstm': '#f78166',
        'har':    '#3fb950', 'volbt': '#d2a8ff',
        'basebt': '#ffa657', 'white': '#ffffff',
    }

    fig = plt.figure(figsize=(20, 24), facecolor=BG)
    gs  = GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

    def ax_style(ax, title=''):
        ax.set_facecolor(CELL)
        for side in ax.spines.values():
            side.set_color(GRID)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
        if title:
            ax.set_title(title, color=TEXT, fontweight='bold', pad=8, fontsize=10)

    n = len(y_true)
    td = test_dates[-n:] if len(test_dates) >= n else pd.RangeIndex(n)

    # --- Panel 0,0: loss curves ---
    ax = fig.add_subplot(gs[0, 0])
    ax_style(ax, 'Training & Validation Loss (MSE)')
    ax.plot(history['train_loss'], color=C['lstm'], lw=1.5, label='Train')
    ax.plot(history['val_loss'],   color=C['har'],  lw=1.5, label='Validation')
    ax.set_xlabel('Epoch');  ax.set_ylabel('MSE')
    ax.legend(facecolor=CELL, labelcolor=TEXT)

    # --- Panel 0,1: scatter ---
    ax = fig.add_subplot(gs[0, 1])
    ax_style(ax, 'LSTM: Predicted vs Actual Vol (Test)')
    ax.scatter(y_true, y_pred_lstm, alpha=0.35, s=6, color=C['lstm'])
    lo = min(y_true.min(), y_pred_lstm.min())
    hi = max(y_true.max(), y_pred_lstm.max())
    ax.plot([lo, hi], [lo, hi], '--', color=C['white'], lw=1.0, alpha=0.5)
    ax.text(0.05, 0.93, f"R² = {r2_score(y_true, y_pred_lstm):.4f}",
            transform=ax.transAxes, color=TEXT, fontsize=10)
    ax.set_xlabel('Actual Vol (ann.)');  ax.set_ylabel('Predicted Vol (ann.)')

    # --- Panel 1,: time series ---
    ax = fig.add_subplot(gs[1, :])
    ax_style(ax, '5-Day Forward Realized Volatility — Test Period')
    ax.plot(td, y_true,      color=C['actual'], lw=1.2, label='Actual', alpha=0.9)
    ax.plot(td, y_pred_lstm, color=C['lstm'],   lw=1.0, label='LSTM',   alpha=0.8)
    ax.plot(td, y_pred_har,  color=C['har'],    lw=1.0, label='HAR-RV', alpha=0.8,
            linestyle='--')
    ax.set_xlabel('Date');  ax.set_ylabel('Annualized Vol')
    ax.legend(facecolor=CELL, labelcolor=TEXT)
    if hasattr(td[0], 'strftime'):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha='right')

    # --- Panel 2,0: residuals histogram ---
    resids = y_pred_lstm - y_true
    ax = fig.add_subplot(gs[2, 0])
    ax_style(ax, 'LSTM Residuals Distribution')
    ax.hist(resids, bins=45, color=C['lstm'], alpha=0.75,
            edgecolor=BG, density=True)
    xr = np.linspace(resids.min(), resids.max(), 300)
    ax.plot(xr, stats.norm.pdf(xr, resids.mean(), resids.std()),
            color=C['white'], lw=1.5, label='Normal fit')
    ax.axvline(0, color=C['har'], ls='--', lw=1.0)
    _, p_sw = stats.shapiro(resids[:min(len(resids), 5000)])
    ax.set_title(f'Residuals (Shapiro-Wilk p = {p_sw:.4f})',
                 color=TEXT, fontweight='bold', fontsize=10)
    ax.set_xlabel('Error (ann. vol)');  ax.set_ylabel('Density')
    ax.legend(facecolor=CELL, labelcolor=TEXT)

    # --- Panel 2,1: backtest equity ---
    ax = fig.add_subplot(gs[2, 1])
    ax_style(ax, 'Backtest — Cumulative Returns (Test Period)')
    eq_vol  = np.cumprod(1.0 + bt['ret_vol'])
    eq_base = np.cumprod(1.0 + bt['ret_base'])
    ax.plot(eq_vol,  color=C['volbt'],  lw=1.5,
            label=f"Vol-Timed (SR={bt['sharpe_vol']:.2f})")
    ax.plot(eq_base, color=C['basebt'], lw=1.5, ls='--',
            label=f"Baseline  (SR={bt['sharpe_base']:.2f})")
    ax.set_xlabel('Trading Days');  ax.set_ylabel('Cumulative Return')
    ax.legend(facecolor=CELL, labelcolor=TEXT)

    # --- Panel 3,0: Q-Q plot ---
    ax = fig.add_subplot(gs[3, 0])
    ax_style(ax, 'Q-Q Plot: LSTM Residuals vs Normal')
    (osm, osr), (slope, intercept, _) = stats.probplot(resids, dist='norm')
    ax.scatter(osm, osr, color=C['lstm'], s=4, alpha=0.5)
    xq = np.array([osm[0], osm[-1]])
    ax.plot(xq, slope * xq + intercept, color=C['white'], lw=1.5)
    ax.set_xlabel('Theoretical Quantiles');  ax.set_ylabel('Sample Quantiles')

    # --- Panel 3,1: rank correlation ---
    ax = fig.add_subplot(gs[3, 1])
    ax_style(ax, 'Rank-Order: LSTM vs HAR-RV Forecasts')
    rank_true = stats.rankdata(y_true)
    ax.scatter(rank_true, stats.rankdata(y_pred_lstm),
               color=C['lstm'],  alpha=0.3, s=5, label='LSTM')
    ax.scatter(rank_true, stats.rankdata(y_pred_har),
               color=C['har'],   alpha=0.3, s=5, label='HAR-RV')
    ax.set_xlabel('Rank: Actual Vol');  ax.set_ylabel('Rank: Predicted Vol')
    ax.legend(facecolor=CELL, labelcolor=TEXT)

    fig.suptitle('LSTM Volatility Forecasting System — SPY',
                 color=TEXT, fontsize=16, fontweight='bold', y=1.005)

    plt.savefig(config['plot_path'], dpi=150, bbox_inches='tight',
                facecolor=BG)
    plt.show()
    print(f"  Plot saved → {config['plot_path']}")


# ============================================================
# SECTION 10: MARKDOWN REPORT
# ============================================================

def generate_report(metrics_list: list, dm: dict,
                    bt: dict, config: dict) -> None:
    now   = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
    lines = [
        "# LSTM Volatility Forecasting — SPY",
        f"**Generated:** {now}  ",
        f"**Ticker:** `{config['ticker']}` | "
        f"**Horizon:** {config['forecast_horizon']} trading days  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "A two-layer stacked LSTM is trained to forecast 5-day forward annualized",
        "realized volatility for SPY. The HAR-RV model (Corsi 2009) serves as the",
        "primary baseline. Forecast quality is assessed via R², RMSE, QLIKE,",
        "Mincer-Zarnowitz regression, and Diebold-Mariano tests.",
        "",
        "## 2. Architecture",
        "",
        "| Component | Specification |",
        "|-----------|---------------|",
        f"| LSTM-1 | {config['hidden1']} hidden units, orthogonal recurrent init |",
        f"| Dropout | p = {config['dropout']} (explicit, after each LSTM layer) |",
        f"| LSTM-2 | {config['hidden2']} hidden units |",
        "| Output | Linear(16→1) + Softplus (positivity guarantee) |",
        f"| Lookback | {config['lookback']} trading days |",
        f"| Parameters | Reported at runtime |",
        "",
        "## 3. Target Variable",
        "",
        "$$\\hat{\\sigma}_{t:t+H} = "
        "\\sqrt{\\frac{252}{H}\\sum_{i=1}^{H} r_{t+i}^2}, \\quad H=5$$",
        "",
        "Annualized realized volatility over the next 5 trading days.",
        "Computed with `shift(-H)` to prevent look-ahead contamination.",
        "",
        "## 4. Features",
        "",
        "| Feature | Description |",
        "|---------|-------------|",
        "| `rv_daily` | $r_t^2$ — daily RV proxy (HAR component) |",
        "| `rv_weekly` | $(1/5)\\sum r_{t-i}^2,\\; i=1\\ldots5$ |",
        "| `rv_monthly` | $(1/22)\\sum r_{t-i}^2,\\; i=1\\ldots22$ |",
        "| `log_return` | $\\log(C_t/C_{t-1})$ |",
        "| `parkinson_vol` | $\\sqrt{\\frac{1}{4\\ln 2}\\cdot\\overline{(\\ln H/L)^2}}$"
        " — Parkinson (1980) |",
        "| `rsi_14` | 14-day RSI $\\in [0,1]$ |",
        "| `rel_volume` | $V_t / \\overline{V}_{20}$ |",
        "| `intraday_range` | $(H_t - L_t)/C_{t-1}$ |",
        "",
        "## 5. Data Splits",
        "",
        "| Set | Period | Observations (approx) |",
        "|-----|--------|------------------------|",
        f"| Train | 2015–{config['train_end'][:4]} | ~1250 |",
        f"| Validation | {config['train_end'][:4]}–{config['val_end'][:4]}"
        " | ~500 (COVID regime) |",
        "| Test | 2022–present | ~750 |",
        "",
        "## 6. Results",
        "",
    ]

    for m in metrics_list:
        lines += [
            f"### {m['model']}",
            "",
            "| Metric | Value | Target |",
            "|--------|-------|--------|",
            f"| R² | `{m['R2']:.4f}` | > 0.08 |",
            f"| RMSE | `{m['RMSE']*100:.4f}%` | < 1% ann. |",
            f"| MAE | `{m['MAE']*100:.4f}%` | < 0.8% ann. |",
            f"| Spearman r | `{m['Spearman_r']:.4f}` (p={m['Spearman_p']:.4f}) "
            "| r > 0.15, p < 0.05 |",
            f"| Dir. Accuracy | `{m['Dir_Acc']:.4f}` (p={m['Dir_p']:.4f}) "
            "| > 0.55 |",
            f"| QLIKE | `{m['QLIKE']:.6f}` | lower is better |",
            f"| MZ α | `{m['MZ_alpha']:.4f}` | H₀: = 0 |",
            f"| MZ β | `{m['MZ_beta']:.4f}` | H₀: = 1 |",
            f"| MZ F p-value | `{m['MZ_F_pval']:.4f}` | H₀: unbiased |",
            "",
        ]

    lines += [
        "### Diebold-Mariano Test (LSTM vs HAR-RV)",
        "",
        f"DM statistic = `{dm['DM_stat']:.4f}`,  p-value = `{dm['DM_pvalue']:.4f}`  ",
        "Negative DM: LSTM more accurate. Positive DM: HAR-RV more accurate.",
        "",
        "## 7. Backtest",
        "",
        "Base signal: sign of trailing 63-day return (momentum).",
        "Vol-timed: position scaled by $1/\\hat{\\sigma}$.",
        "",
        "| Metric | Vol-Timed | Baseline |",
        "|--------|-----------|----------|",
        f"| Sharpe | `{bt['sharpe_vol']:.4f}` | `{bt['sharpe_base']:.4f}` |",
        f"| Max DD | `{bt['maxdd_vol']*100:.2f}%` | `{bt['maxdd_base']*100:.2f}%` |",
        f"| Calmar | `{bt['calmar_vol']:.4f}` | `{bt['calmar_base']:.4f}` |",
        f"| ΔSharpe | `{bt['sharpe_improvement']:+.4f}` | — |",
        "",
        "## 8. Red-Team Critique",
        "",
        "1. **HAR-RV competitiveness.** The empirical literature (Patton & Sheppard 2009,",
        "   Audrino & Schaumburg 2015) finds HAR-RV difficult to beat on daily data.",
        "   DM-test significance is required before claiming LSTM superiority.",
        "",
        "2. **COVID validation regime.** The 2020–2021 validation set represents a",
        "   structural break. Good validation performance may reflect COVID-specific",
        "   mean-reversion patterns not present in the test set.",
        "",
        "3. **Data-to-parameter ratio.** With ~10,000 LSTM parameters and ~1,200",
        "   training sequences, the n/p ratio is ~0.12. Regularization is non-trivial.",
        "   HAR-RV has 4 parameters on the same data — Occam favors parsimony.",
        "",
        "4. **Proxy noise in targets.** Daily squared returns are a noisy proxy for",
        "   latent volatility (Barndorff-Nielsen & Shephard 2002). QLIKE is robust",
        "   to this, but R² and RMSE may be contaminated.",
        "",
        "5. **Momentum signal independence.** The backtest uses a momentum signal",
        "   unrelated to vol forecasting. Sharpe improvement isolates vol-timing",
        "   alpha, but the result depends on momentum remaining valid.",
        "",
        "6. **Transaction cost assumption.** 5 bps one-way assumes institutional",
        "   execution. Retail costs of 10–15 bps would significantly compress Sharpe.",
        "",
        "## 9. Recommended Extensions",
        "",
        "- Replace MSE with QLIKE training loss for proxy-robust optimization.",
        "- Add VIX as an exogenous feature (implied vol information).",
        "- Optimal combination: $\\hat{\\sigma}_{combo} = w\\hat{\\sigma}_{LSTM} + "
        "(1-w)\\hat{\\sigma}_{HAR}$ via Bates-Granger (1969).",
        "- Replace daily RV proxy with 5-minute realized variance from tick data.",
        "- Attention mechanism to identify regime-specific lookback periods.",
    ]

    report = "\n".join(lines)
    with open(config['report_path'], 'w') as f:
        f.write(report)
    print(f"  Report saved → {config['report_path']}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> dict:
    print("=" * 62)
    print("  LSTM VOLATILITY FORECASTING SYSTEM — SPY")
    print("  Quant Professor Implementation")
    print("=" * 62)

    # ── 1. Data ─────────────────────────────────────────────
    raw = download_data(CONFIG['ticker'], CONFIG['start_date'])

    # ── 2. Features & Target ────────────────────────────────
    feat   = compute_features(raw)
    target = compute_target(raw, CONFIG['forecast_horizon'])

    data = pd.concat([feat, target], axis=1).dropna()
    print(f"\n[DATASET] {len(data)} clean samples after dropna.")

    # ── 3. Chronological Splits ─────────────────────────────
    tr_mask  = data.index <= CONFIG['train_end']
    vl_mask  = (data.index >  CONFIG['train_end']) & (data.index <= CONFIG['val_end'])
    te_mask  = data.index >  CONFIG['val_end']

    fcols = CONFIG['features']

    X_tr = data.loc[tr_mask, fcols].values.astype(np.float32)
    X_vl = data.loc[vl_mask, fcols].values.astype(np.float32)
    X_te = data.loc[te_mask, fcols].values.astype(np.float32)
    y_tr = data.loc[tr_mask, 'target_vol'].values.astype(np.float32)
    y_vl = data.loc[vl_mask, 'target_vol'].values.astype(np.float32)
    y_te = data.loc[te_mask, 'target_vol'].values.astype(np.float32)

    te_dates = data.index[te_mask]
    print(f"[SPLITS] Train: {len(X_tr)} | Val: {len(X_vl)} | Test: {len(X_te)}")

    # ── 4. Normalization (fit on train only) ─────────────────
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_vl_s  = scaler.transform(X_vl)
    X_te_s  = scaler.transform(X_te)
    pickle.dump(scaler, open(CONFIG['scaler_path'], 'wb'))

    # ── 5. Sequence Construction ─────────────────────────────
    LB = CONFIG['lookback']
    Xtr_seq, ytr_seq = make_sequences(X_tr_s, y_tr, LB)
    Xvl_seq, yvl_seq = make_sequences(X_vl_s, y_vl, LB)
    Xte_seq, yte_seq = make_sequences(X_te_s, y_te, LB)
    print(f"[SEQUENCES] Train: {Xtr_seq.shape} | "
          f"Val: {Xvl_seq.shape} | Test: {Xte_seq.shape}")

    # ── 6. DataLoaders ───────────────────────────────────────
    train_ds     = TensorDataset(torch.from_numpy(Xtr_seq),
                                  torch.from_numpy(ytr_seq))
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                               shuffle=True, drop_last=True)
    X_vl_t = torch.from_numpy(Xvl_seq)
    y_vl_t = torch.from_numpy(yvl_seq)
    X_te_t = torch.from_numpy(Xte_seq)

    # ── 7. HAR-RV Baseline ───────────────────────────────────
    print("\n[HAR-RV] Fitting baseline...")
    # Feature alignment: sequence i has last feature day i+LB-1;
    # we use features at that day for HAR prediction.
    har_cols_df = data.loc[tr_mask, fcols].values
    har_X_tr    = pd.DataFrame(har_cols_df[LB-1 : LB-1+len(ytr_seq)],
                                columns=fcols)
    har_cols_df = data.loc[vl_mask, fcols].values
    har_X_vl    = pd.DataFrame(har_cols_df[LB-1 : LB-1+len(yvl_seq)],
                                columns=fcols)
    har_cols_df = data.loc[te_mask, fcols].values
    har_X_te    = pd.DataFrame(har_cols_df[LB-1 : LB-1+len(yte_seq)],
                                columns=fcols)

    har = HARRVModel()
    har.fit(har_X_tr, ytr_seq)
    print(har.summary())

    y_pred_har_te = har.predict(har_X_te)
    y_pred_har_te = y_pred_har_te[:len(yte_seq)]

    # ── 8. LSTM Training ─────────────────────────────────────
    n_features = len(fcols)
    model      = VolatilityLSTM(n_features, CONFIG['hidden1'],
                                  CONFIG['hidden2'], CONFIG['dropout'])
    n_params   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[LSTM] Trainable parameters: {n_params:,}")
    print(f"  n_train_seq / n_params = {len(ytr_seq)/n_params:.4f}  "
          f"({'OK (>1)' if len(ytr_seq)/n_params > 1 else 'WARNING: underdetermined'})")

    print("\n[LSTM] Training...")
    history = train_lstm(model, train_loader, X_vl_t, y_vl_t, CONFIG)

    # ── 9. Predictions ───────────────────────────────────────
    model.eval()
    with torch.no_grad():
        y_pred_lstm = model(X_te_t.to(DEVICE)).cpu().numpy()

    # Align lengths
    n_eval = min(len(yte_seq), len(y_pred_lstm), len(y_pred_har_te))
    y_true   = yte_seq[:n_eval]
    y_lstm   = y_pred_lstm[:n_eval]
    y_har    = y_pred_har_te[:n_eval]

    # ── 10. Evaluation ───────────────────────────────────────
    print("\n[EVALUATION] Test-set performance:")
    m_lstm = evaluate_forecasts(y_true, y_lstm,  label='LSTM')
    m_har  = evaluate_forecasts(y_true, y_har,   label='HAR-RV')

    e_lstm = y_true - y_lstm
    e_har  = y_true - y_har
    dm     = diebold_mariano_test(e_lstm, e_har, h=CONFIG['forecast_horizon'])

    metrics_df = pd.DataFrame([m_lstm, m_har])
    metrics_df.to_csv(CONFIG['metrics_path'], index=False)
    print(f"\n  Metrics saved → {CONFIG['metrics_path']}")

    # Red flags
    print("\n[VALIDATION CHECKLIST]")
    flags = {
        'R² > 0.08':              m_lstm['R2'] > 0.08,
        'Dir. Acc. > 0.55':       m_lstm['Dir_Acc'] > 0.55,
        'Spearman r > 0.15':      m_lstm['Spearman_r'] > 0.15,
        'Spearman p < 0.05':      m_lstm['Spearman_p'] < 0.05,
        'MZ F p > 0.05 (unbiased)': m_lstm['MZ_F_pval'] > 0.05,
    }
    for check, passed in flags.items():
        icon = '✓' if passed else '✗'
        print(f"  {icon}  {check}")

    # ── 11. Backtest ─────────────────────────────────────────
    print("\n[BACKTEST] Vol-timing strategy (test period)...")
    log_ret_all = np.log(raw['Close'] / raw['Close'].shift(1)).dropna()
    bt = run_vol_timing_backtest(log_ret_all, y_lstm, CONFIG['cost_bps'])

    print(f"  Sharpe (vol-timed):  {bt['sharpe_vol']:.4f}")
    print(f"  Sharpe (baseline):   {bt['sharpe_base']:.4f}")
    print(f"  Max DD (vol-timed):  {bt['maxdd_vol']*100:.2f}%")
    print(f"  Calmar (vol-timed):  {bt['calmar_vol']:.4f}")
    print(f"  Sharpe improvement:  {bt['sharpe_improvement']:+.4f}")

    if bt['sharpe_improvement'] > 0.2:
        print("  ✓ Improvement > 0.2 Sharpe points: vol forecast adds value.")
    else:
        print("  ✗ Improvement ≤ 0.2: vol forecast marginal for position sizing.")

    # ── 12. Visualization ────────────────────────────────────
    print("\n[PLOT] Generating diagnostic figure...")
    generate_plots(y_true, y_lstm, y_har, te_dates, history, bt, CONFIG)

    # ── 13. Report ───────────────────────────────────────────
    print("\n[REPORT] Writing markdown report...")
    generate_report([m_lstm, m_har], dm, bt, CONFIG)

    # ── 14. Save model config ────────────────────────────────
    model_config = {
        'n_features':       n_features,
        'hidden1':          CONFIG['hidden1'],
        'hidden2':          CONFIG['hidden2'],
        'dropout':          CONFIG['dropout'],
        'lookback':         CONFIG['lookback'],
        'forecast_horizon': CONFIG['forecast_horizon'],
        'features':         CONFIG['features'],
        'train_end':        CONFIG['train_end'],
        'val_end':          CONFIG['val_end'],
        'n_params':         n_params,
        'seed':             SEED,
    }
    with open(CONFIG['config_path'], 'w') as f:
        json.dump(model_config, f, indent=2)

    print("\n[DONE] All outputs:")
    for k in ['model_path', 'scaler_path', 'metrics_path',
               'report_path', 'config_path', 'plot_path']:
        print(f"  → {CONFIG[k]}")

    return {
        'model':    model,
        'scaler':   scaler,
        'metrics':  {'lstm': m_lstm, 'har': m_har},
        'backtest': bt,
        'history':  history,
        'dm_test':  dm,
    }


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    results = main()
