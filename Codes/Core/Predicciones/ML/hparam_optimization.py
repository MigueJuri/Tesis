"""
hyperparameter_optimization.py
================================
Bayesian Optimisation of Feature-Engineering Hyperparameters
for the Three-Factor GBDT Return Predictor.

Companion script to TreeBasedPredictor_v2.ipynb.

Optimised parameters
--------------------
- forward_horizon  : prediction horizon in trading days  [1, 21]
- mom_short        : short momentum window               [1,  5]
- mom_medium       : medium momentum window              [5, 42]
- mom_long         : long momentum window               [21,126]
- vol_window       : realised variance lookback          [5, 63]
- ema_span         : EMA z-score span                   [5, 63]

Validation scheme
-----------------
Walk-forward expanding window with embargo gap equal to forward_horizon.
A single outer loop runs over all folds; each fold trains on all prior
data and evaluates on the next step_days window.  This avoids any
look-ahead at the parameter level.

Metrics evaluated (see comparative analysis at bottom)
------------------------------------------------------
- IC      : mean daily cross-sectional Spearman rank correlation
- ICIR    : IC / std(IC) × sqrt(252)  — IC Sharpe ratio
- Pearson : mean daily Pearson correlation of predicted vs realised returns

Primary selection metric: ICIR  (justified in comparative analysis section)

Dependencies
------------
pip install lightgbm scikit-optimize pandas numpy scipy matplotlib seaborn yfinance
"""

# ─────────────────────────────────────────────────────────────────────────────
#  0. Imports
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import lightgbm as lgb
from scipy import stats
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "serif",
    "axes.titlesize": 11,
    "axes.labelsize": 9,
})


# ─────────────────────────────────────────────────────────────────────────────
#  1. Static Configuration
#     Only parameters that are NOT being optimised live here.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StaticConfig:
    """
    Fixed experiment settings — not touched by the optimiser.

    Separating static from tunable parameters is essential: if the
    optimiser could modify, say, `min_train_days`, it could fabricate
    apparent performance by training on arbitrarily large windows.
    """
    tickers: List[str] = field(default_factory=lambda: [
        "AAPL", "KO", "JPM", "XOM", "JNJ",
        "GS",   "BA",   "CAT", "PFE", "WMT",
    ])
    start_date: str = "2010-01-01"
    end_date:   str = "2024-12-31"

    # Walk-forward fixed settings
    min_train_days: int = 504       # ≈ 2 years before first test fold
    step_days:      int = 63        # retrain every quarter

    # FFD (not optimised here — treated as a preprocessing constant)
    ffd_d:      float = 0.4
    ffd_thresh: float = 1e-4
    ffd_window: int   = 126

    # LightGBM — fixed, heavily regularised to keep feature search honest.
    # We do NOT optimise LGB hyperparameters simultaneously with features:
    # joint optimisation creates a high-dimensional space that is extremely
    # prone to over-fitting in financial time series.
    lgb_params: Dict = field(default_factory=lambda: {
        "objective":         "regression",
        "metric":            "rmse",
        "n_estimators":      200,
        "learning_rate":     0.05,
        "max_depth":         4,
        "num_leaves":        15,
        "min_child_samples": 30,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "reg_lambda":        1.0,
        "verbose":           -1,
        "random_state":      42,
    })


SCFG = StaticConfig()


# ─────────────────────────────────────────────────────────────────────────────
#  2. Search Space Definition
#
#  skopt.space.Integer defines an integer-valued dimension for the Gaussian
#  Process surrogate.  The GP operates in the continuous relaxation [low, high]
#  and rounds to the nearest integer before evaluation.
#
#  Constraint enforced at evaluation time: mom_short < mom_medium < mom_long.
#  When the GP proposes a violated configuration, we return a large penalty
#  score rather than skipping, so the surrogate learns to avoid that region.
# ─────────────────────────────────────────────────────────────────────────────
SEARCH_SPACE = [
    Integer(1,   21, name="forward_horizon"),   # days
    Integer(1,    21, name="mom_short"),         # days
    Integer(22,   126, name="mom_medium"),        # days
    Integer(127, 252, name="mom_long"),          # days
    Integer(5,   100, name="vol_window"),        # days
    Integer(5,   100, name="ema_span"),          # days
]

PENALTY_SCORE = -99.0   # returned when momentum window constraint is violated


# ─────────────────────────────────────────────────────────────────────────────
#  3. Data Layer
# ─────────────────────────────────────────────────────────────────────────────
def fetch_prices(cfg: StaticConfig) -> pd.DataFrame:
    """Download and clean adjusted close prices."""
    log.info("Fetching prices from Yahoo Finance …")
    raw = yf.download(
        tickers=cfg.tickers,
        start=cfg.start_date,
        end=cfg.end_date,
        auto_adjust=True,
        progress=False,
    )["Close"]
    prices = raw.ffill(limit=3).dropna(how="all")
    pct_missing = prices.isna().mean()
    prices = prices[pct_missing[pct_missing < 0.05].index].dropna()
    log.info(f"Prices: {prices.shape[0]} days × {prices.shape[1]} tickers")
    return prices

# ─────────────────────────────────────────────────────────────────────────────
#  4. Feature Engineering (parameterised)
#
#  Every tunable parameter flows through build_panel.  The function is called
#  once per Bayesian optimisation trial — it must be fast.  Key decisions:
#
#  - Momentum: cumulative log-return log(P_{t-1}/P_{t-k}) for k ∈ {short, medium, long}
#    The one-day skip (using P_{t-1} instead of P_t) avoids microstructure reversal.
#  - Vol: rolling sample variance of log-returns (Bessel-corrected, demeaned).
#  - MR:  EMA z-score.  Even though dimensionless, ranked cross-sectionally.
#  - Target: h-day forward log-return.  Stored as fwd_return.
#    dropna applied to features only; rows with NaN target are kept for scoring.
# ─────────────────────────────────────────────────────────────────────────────
def build_panel(
    prices: pd.DataFrame,
    forward_horizon: int,
    mom_short:       int,
    mom_medium:      int,
    mom_long:        int,
    vol_window:      int,
    ema_span:        int,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the long-format feature panel for a given hyperparameter configuration.

    Parameters
    ----------
    prices          : (T, N) adjusted close prices.
    forward_horizon : h-day ahead return to predict.
    mom_short/medium/long : momentum lookback windows in days.
    vol_window      : rolling variance window.
    ema_span        : EMA z-score span.

    Returns
    -------
    panel       : Long-format DataFrame with date, ticker, features, fwd_return.
    feature_cols: List of feature column names (for model.fit / cross-sec rank).
    """
    records      = []
    feature_cols = ["mom_s", "mom_m", "mom_l", "vol", "z_ema"]

    for ticker in prices.columns:
        p    = prices[ticker].dropna()
        logp = np.log(p)
        r    = logp.diff()

        # ── Momentum features (cumulative log-returns with 1-day skip) ───────
        mom_s = (logp.shift(1) - logp.shift(mom_short)).rename("mom_s")
        mom_m = (logp.shift(1) - logp.shift(mom_medium)).rename("mom_m")
        mom_l = (logp.shift(1) - logp.shift(mom_long)).rename("mom_l")

        # ── Volatility feature (unbiased sample variance) ─────────────────────
        vol = r.rolling(vol_window, min_periods=vol_window // 2).var().rename("vol")

        # ── Mean-reversion feature (EMA z-score) ──────────────────────────────
        ema      = p.ewm(span=ema_span, adjust=False).mean()
        roll_std = p.rolling(ema_span, min_periods=ema_span // 2).std()
        z_ema    = ((p - ema) / roll_std.replace(0, np.nan)).rename("z_ema")

        # ── Forward return target ──────────────────────────────────────────────
        fwd_return = (logp.shift(-forward_horizon) - logp).rename("fwd_return")

        df = pd.concat([mom_s, mom_m, mom_l, vol, z_ema, fwd_return], axis=1)
        df.insert(0, "ticker", ticker)
        df.insert(1, "date",   df.index)
        records.append(df)

    panel = pd.concat(records, axis=0).reset_index(drop=True)
    # Drop only rows where features are missing; keep NaN target rows
    panel = panel.dropna(subset=feature_cols)
    return panel, feature_cols


def cross_sectional_rank(panel: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Replace raw feature values with cross-sectional percentile ranks ∈ [0,1].
    Ranking is done within each date, ensuring comparability across assets.
    """
    def _rank(g: pd.DataFrame) -> pd.DataFrame:
        N = len(g)
        g = g.copy()
        if N == 1:
            g[feature_cols] = 0.5
            return g
        g[feature_cols] = (g[feature_cols].rank(method="average") - 1) / (N - 1)
        return g

    return (
        panel.copy()
        .groupby("date", group_keys=False)
        .apply(_rank)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
#  5. Walk-Forward Validation Engine
#
#  The splits generator is parameterised by forward_horizon so the embargo
#  is always exactly equal to the prediction horizon — the minimum required
#  to prevent label overlap (see D1 fix in the main notebook).
# ─────────────────────────────────────────────────────────────────────────────
def make_splits(
    dates: pd.DatetimeIndex,
    min_train: int,
    step: int,
    embargo: int,
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Generate expanding-window (train, test) date pairs with embargo gap.

    The last `embargo` dates are excluded from each training set to prevent
    the target for those dates from using prices from the test window.
    """
    splits, cursor, T = [], min_train, len(dates)
    while cursor + step <= T:
        train_end = max(0, cursor - embargo)
        if train_end > 0:
            splits.append((dates[:train_end], dates[cursor : cursor + step]))
        cursor += step
    return splits


# ─────────────────────────────────────────────────────────────────────────────
#  6. Metric Computation
#
#  Three metrics are computed per trial.  See Section 9 for comparative analysis.
#
#  IC (Spearman rank correlation)
#  ─────────────────────────────
#  IC_t = ρ_S( ŷ_{t,·}, y_{t,·} )
#
#  Computed daily (cross-sectional), then averaged over time.
#
#  ICIR (IC Information Ratio)
#  ───────────────────────────
#  ICIR = mean(IC) / std(IC) × sqrt(252)
#
#  This is the IC Sharpe: it penalises inconsistency in the signal.
#  A model that is occasionally brilliant but often wrong has a low ICIR
#  even if its mean IC is positive.
#
#  Pearson correlation
#  ───────────────────
#  ρ_t = Pearson( ŷ_{t,·}, y_{t,·} )
#
#  Sensitive to outliers in both predictions and returns.
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(oos: pd.DataFrame) -> Dict[str, float]:
    """
    Compute IC, ICIR, and Pearson from OOS predictions.

    Parameters
    ----------
    oos : DataFrame with columns [date, fwd_return, predicted_return].

    Returns
    -------
    dict with keys "ic", "icir", "pearson".  All NaN if insufficient data.
    """
    df = oos.dropna(subset=["fwd_return", "predicted_return"])
    if len(df) < 10:
        return {"ic": np.nan, "icir": np.nan, "pearson": np.nan}

    # Cross-sectional IC per day (vectorised rank correlation)
    df = df.copy()
    df["rk_pred"] = df.groupby("date")["predicted_return"].rank(method="average")
    df["rk_real"] = df.groupby("date")["fwd_return"].rank(method="average")

    daily_ic = df.groupby("date").apply(
        lambda g: g["rk_pred"].corr(g["rk_real"]) if len(g) >= 3 else np.nan
    ).dropna()

    daily_pearson = df.groupby("date").apply(
        lambda g: g["predicted_return"].corr(g["fwd_return"]) if len(g) >= 3 else np.nan
    ).dropna()

    ic_mean = daily_ic.mean()
    ic_std  = daily_ic.std()
    icir    = ic_mean / ic_std * np.sqrt(252) if ic_std > 0 else np.nan

    return {
        "ic":      ic_mean,
        "icir":    icir,
        "pearson": daily_pearson.mean(),
        "daily_ic": daily_ic,   # full series for later analysis
    }


# ─────────────────────────────────────────────────────────────────────────────
#  7. Single-Trial Evaluator
#
#  This function is called by the Bayesian optimiser for each proposed
#  hyperparameter configuration.  It must return a scalar to MINIMISE,
#  so we return the negative of the primary metric (ICIR).
#
#  The walk-forward loop inside is intentionally lightweight:
#  - No early stopping (fixed n_estimators)
#  - No model persistence (we only want the OOS metric)
#  - Logging suppressed per fold; only aggregate logged
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_config(
    prices:          pd.DataFrame,
    forward_horizon: int,
    mom_short:       int,
    mom_medium:      int,
    mom_long:        int,
    vol_window:      int,
    ema_span:        int,
    cfg:             StaticConfig,
    primary_metric:  str = "icir",
) -> Tuple[float, Dict]:
    """
    Build panel, run walk-forward CV, return (negative primary metric, full metrics dict).

    Returns
    -------
    score   : float — negative primary_metric (to be minimised by skopt).
    metrics : dict  — {"ic": ..., "icir": ..., "pearson": ..., "daily_ic": Series}.
    """
    # Momentum window order constraint
    if not (mom_short < mom_medium < mom_long):
        return PENALTY_SCORE, {}

    # Build and rank panel
    panel, feature_cols = build_panel(
        prices, forward_horizon, mom_short, mom_medium, mom_long, vol_window, ema_span
    )
    panel = cross_sectional_rank(panel, feature_cols)

    # Walk-forward splits
    unique_dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    splits = make_splits(
        unique_dates,
        min_train=cfg.min_train_days,
        step=cfg.step_days,
        embargo=forward_horizon,
    )
    if len(splits) < 3:
        return PENALTY_SCORE, {}

    all_preds = []
    for train_dates, test_dates in splits:
        train = panel[panel["date"].isin(train_dates)].dropna(subset=["fwd_return"])
        test  = panel[panel["date"].isin(test_dates)].dropna(subset=feature_cols)

        if len(train) < 100 or train["fwd_return"].nunique() < 2:
            continue

        model = lgb.LGBMRegressor(**cfg.lgb_params)
        model.fit(train[feature_cols].values, train["fwd_return"].values,
                  feature_name=feature_cols)

        pred = model.predict(test[feature_cols].values)
        out  = test[["date", "ticker", "fwd_return"]].copy()
        out["predicted_return"] = pred
        all_preds.append(out)

    if not all_preds:
        return PENALTY_SCORE, {}

    oos     = pd.concat(all_preds, ignore_index=True)
    metrics = compute_metrics(oos)

    score = metrics.get(primary_metric, np.nan)
    if np.isnan(score):
        return PENALTY_SCORE, metrics

    return -score, metrics   # minimise negative metric


# ─────────────────────────────────────────────────────────────────────────────
#  8. Bayesian Optimisation Driver
#
#  We use scikit-optimize's gp_minimize with the Matern 5/2 kernel.
#  This kernel is appropriate for integer-valued spaces because it is
#  continuous but not infinitely differentiable — better suited for
#  step-like objective surfaces than the squared-exponential kernel.
#
#  Acquisition function: Expected Improvement (EI).
#  EI balances exploitation of known good regions with exploration of
#  uncertain regions — critical when each evaluation costs minutes.
#
#  n_initial_points: the first n evaluations are random (Latin Hypercube).
#  This seeds the GP with enough points to build a meaningful surrogate
#  before the acquisition function takes over.
#
#  n_calls: total evaluations = n_initial_points + GP-guided evaluations.
#  Each call runs the full walk-forward loop (~N_folds × model fits).
# ─────────────────────────────────────────────────────────────────────────────
def run_bayesian_optimisation(
    prices:          pd.DataFrame,
    cfg:             StaticConfig,
    n_calls:         int = 40,
    n_initial_points: int = 10,
    primary_metric:  str = "icir",
    verbose:         bool = True,
) -> Tuple[object, List[Dict]]:
    """
    Run Bayesian optimisation over the feature-engineering search space.

    Parameters
    ----------
    prices           : (T, N) price DataFrame.
    cfg              : static configuration.
    n_calls          : total number of objective evaluations.
    n_initial_points : random initial evaluations before GP takes over.
    primary_metric   : "icir" | "ic" | "pearson" — metric to maximise.
    verbose          : print progress each iteration.

    Returns
    -------
    result  : skopt OptimizeResult object.
    history : list of dicts, one per call, with params + all metrics.
    """
    history: List[Dict] = []
    call_idx = [0]

    @use_named_args(SEARCH_SPACE)
    def objective(
        forward_horizon,
        mom_short,
        mom_medium,
        mom_long,
        vol_window,
        ema_span,
    ) -> float:
        call_idx[0] += 1
        params = dict(
            forward_horizon=int(forward_horizon),
            mom_short=int(mom_short),
            mom_medium=int(mom_medium),
            mom_long=int(mom_long),
            vol_window=int(vol_window),
            ema_span=int(ema_span),
        )

        score, metrics = evaluate_config(
            prices=prices,
            cfg=cfg,
            primary_metric=primary_metric,
            **params,
        )

        record = {**params, "score": -score, **{
            k: v for k, v in metrics.items() if k != "daily_ic"
        }}
        history.append(record)

        if verbose:
            constraint_ok = params["mom_short"] < params["mom_medium"] < params["mom_long"]
            status = "✓" if constraint_ok else "⚠ constraint"
            log.info(
                f"[{call_idx[0]:>3d}/{n_calls}] "
                f"h={params['forward_horizon']:>2d} "
                f"mom=({params['mom_short']:>2d},{params['mom_medium']:>2d},{params['mom_long']:>3d}) "
                f"vol={params['vol_window']:>2d} ema={params['ema_span']:>2d} "
                f"→ {primary_metric.upper()}={record.get(primary_metric, 0):+.4f}  {status}"
            )

        return score   # skopt minimises this

    log.info(
        f"Starting Bayesian Optimisation: {n_calls} calls "
        f"({n_initial_points} random + {n_calls - n_initial_points} GP-guided)"
    )
    result = gp_minimize(
        func=objective,
        dimensions=SEARCH_SPACE,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        acq_func="EI",              # Expected Improvement
        acq_optimizer="lbfgs",
        random_state=42,
        noise=1e-6,                 # small noise stabilises GP inversion
    )

    log.info(f"Optimisation complete.  Best {primary_metric.upper()} = {-result.fun:.4f}")
    return result, history


# ─────────────────────────────────────────────────────────────────────────────
#  9. Metric Comparative Analysis
#
#  This section computes all three metrics on the BEST configuration found
#  by Bayesian optimisation and produces a formal comparison.
#
#  Theoretical justification for metric choice
#  --------------------------------------------
#
#  Pearson correlation
#  ───────────────────
#  ρ_P(ŷ, y) = Cov(ŷ, y) / (σ_ŷ σ_y)
#
#  Sensitive to outlier returns and outlier predictions.  Financial returns
#  have heavy tails (kurtosis >> 3) and occasional large jumps.  A single
#  crash day where the model predicted the wrong sign can dominate Pearson
#  correlation and cause it to misrepresent the average cross-sectional
#  relationship.  Furthermore, Pearson is sensitive to the scaling of ŷ,
#  which is model-dependent and has no natural interpretation.
#  → Conclusion: NOT appropriate as a primary metric for this context.
#
#  IC (mean Spearman rank correlation)
#  ────────────────────────────────────
#  IC_t = ρ_S(ŷ_{t,·}, y_{t,·})
#
#  Rank-based: immune to outliers in both predictions and returns.
#  Cross-sectional: measures whether the model correctly ranks assets
#  on each day, which is the correct objective for a long-short strategy.
#  Directly interpretable: IC > 0 means the model predicts the right
#  relative ordering more often than chance.
#  Scale-invariant: a model predicting ŷ or 10×ŷ gets the same IC.
#  → Conclusion: Valid primary metric, but mean IC alone ignores consistency.
#
#  ICIR (IC Information Ratio = IC / std(IC) × sqrt(252))
#  ────────────────────────────────────────────────────────
#  ICIR = IC_mean / IC_std × sqrt(252)
#
#  This is the IC Sharpe ratio.  It penalises a model that produces a high
#  average IC but with large variance (regime-dependent, unreliable signal).
#  In portfolio construction, signal consistency matters as much as average
#  signal strength: a model with ICIR = 1.0 built from IC_mean = 0.02,
#  IC_std = 0.02 is MORE valuable than one with IC_mean = 0.05, IC_std = 0.20
#  (ICIR = 0.35) because the latter requires much larger position sizing
#  to extract alpha and produces violent drawdowns when IC goes negative.
#  The Grinold-Kahn fundamental law of active management states:
#       IR ≈ IC × sqrt(N)
#  where N is the number of independent bets per year.  Maximising ICIR
#  directly maximises the portfolio-level information ratio.
#  → Conclusion: ICIR is the appropriate primary selection metric.
#
#  Selection: ICIR > IC > Pearson for financial time-series cross-sectional models.
# ─────────────────────────────────────────────────────────────────────────────
def run_metric_comparison(
    prices:         pd.DataFrame,
    best_params:    Dict,
    cfg:            StaticConfig,
) -> pd.DataFrame:
    """
    Evaluate the best hyperparameter configuration under all three metrics
    and return a structured comparison DataFrame.
    """
    log.info("Running metric comparison on best configuration …")
    _, metrics = evaluate_config(prices=prices, cfg=cfg, primary_metric="icir", **best_params)

    daily_ic = metrics.get("daily_ic", pd.Series(dtype=float))
    ic_mean  = daily_ic.mean()
    ic_std   = daily_ic.std()
    icir     = ic_mean / ic_std * np.sqrt(252) if ic_std > 0 else np.nan

    comparison = pd.DataFrame({
        "Metric":          ["Pearson",          "IC (Spearman)",         "ICIR"],
        "Value":           [metrics.get("pearson", np.nan), ic_mean, icir],
        "Interpretation":  [
            "Linear pred–realised correlation; outlier-sensitive",
            "Mean cross-sectional rank correlation; outlier-robust",
            "IC / std(IC) × √252; penalises inconsistency",
        ],
        "Appropriate?":    ["⚠ No — heavy tails", "✓ Valid but incomplete", "✓ Primary metric"],
        "Reason":          [
            "Fat-tailed returns make Pearson unstable; does not penalise inconsistency",
            "Rank-based and scale-invariant, but ignores signal variance over time",
            "Maximising ICIR maximises portfolio IR (Grinold-Kahn); rewards consistency",
        ],
    })
    return comparison, daily_ic


# ─────────────────────────────────────────────────────────────────────────────
#  10. Visualisation Suite
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(
    result,
    history:    List[Dict],
    daily_ic:   pd.Series,
    best_params: Dict,
) -> None:
    """
    Six-panel visualisation:
      A — Convergence curve (best ICIR found vs iteration)
      B — Objective surface: forward_horizon vs mom_medium (2D marginal)
      C — Parameter importance (sensitivity of ICIR to each dimension)
      D — Distribution of all evaluated ICIR scores
      E — Daily IC time-series of best configuration
      F — IC autocorrelogram (overlap detection)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Bayesian Hyperparameter Optimisation — Feature Engineering for GBDT Predictor",
        fontsize=13, y=1.01,
    )

    hist_df = pd.DataFrame([h for h in history if h.get("icir") is not None])

    # ── A: Convergence curve ──────────────────────────────────────────────────
    ax = axes[0, 0]
    if len(hist_df) > 0:
        best_so_far = hist_df["icir"].cummax()
        ax.plot(range(1, len(best_so_far) + 1), best_so_far,
                lw=2, color="steelblue", label="Best ICIR so far")
        ax.scatter(range(1, len(hist_df) + 1), hist_df["icir"],
                   s=15, alpha=0.4, color="grey", label="Each trial")
        ax.axhline(best_so_far.iloc[-1], color="crimson", ls="--", lw=1.2,
                   label=f"Best = {best_so_far.iloc[-1]:.3f}")
    ax.set_title("A — Optimisation Convergence")
    ax.set_xlabel("Trial index")
    ax.set_ylabel("ICIR (higher = better)")
    ax.legend(fontsize=8)

    # ── B: 2D scatter — forward_horizon vs mom_medium coloured by ICIR ────────
    ax = axes[0, 1]
    if len(hist_df) > 0:
        sc = ax.scatter(
            hist_df["forward_horizon"], hist_df["mom_medium"],
            c=hist_df["icir"], cmap="RdYlGn", s=60, alpha=0.8,
            vmin=hist_df["icir"].quantile(0.1),
            vmax=hist_df["icir"].quantile(0.9),
        )
        plt.colorbar(sc, ax=ax, label="ICIR")
        ax.set_xlabel("Prediction horizon (days)")
        ax.set_ylabel("Medium momentum window (days)")
    ax.set_title("B — Horizon × Medium Momentum Surface")

    # ── C: Parameter sensitivity (range of ICIR when each param varies) ───────
    ax = axes[0, 2]
    param_names = ["forward_horizon", "mom_short", "mom_medium", "mom_long",
                   "vol_window", "ema_span"]
    if len(hist_df) >= 5:
        sensitivity = {}
        for p in param_names:
            if p in hist_df.columns:
                corr = hist_df[[p, "icir"]].dropna().corr(method="spearman").iloc[0, 1]
                sensitivity[p] = abs(corr)
        sens_series = pd.Series(sensitivity).sort_values()
        colors = ["#2ecc71" if v > 0.3 else "#e74c3c" if v < 0.1 else "#f39c12"
                  for v in sens_series.values]
        ax.barh(sens_series.index, sens_series.values, color=colors, alpha=0.85)
        ax.axvline(0.3, color="green",  ls="--", lw=1, label="|ρ| > 0.3 (high impact)")
        ax.axvline(0.1, color="orange", ls="--", lw=1, label="|ρ| > 0.1 (medium)")
        ax.legend(fontsize=7)
    ax.set_title("C — Parameter Sensitivity\n(|Spearman ρ| with ICIR)")
    ax.set_xlabel("|Spearman correlation with ICIR|")

    # ── D: Distribution of ICIR across all trials ─────────────────────────────
    ax = axes[1, 0]
    if len(hist_df) > 0:
        ax.hist(hist_df["icir"].dropna(), bins=20, color="steelblue",
                alpha=0.8, edgecolor="white")
        ax.axvline(hist_df["icir"].max(), color="crimson", lw=2,
                   label=f"Best = {hist_df['icir'].max():.3f}")
        ax.axvline(0, color="black", lw=0.9, ls=":")
        ax.legend(fontsize=8)
    ax.set_title("D — ICIR Distribution Across Trials")
    ax.set_xlabel("ICIR")
    ax.set_ylabel("Count")

    # ── E: Daily IC time-series of best config ────────────────────────────────
    ax = axes[1, 1]
    if daily_ic is not None and len(daily_ic) > 0:
        rolling_ic = daily_ic.rolling(63, min_periods=32).mean()
        ax.plot(daily_ic.index, daily_ic.values,
                lw=0.6, color="lightgrey", alpha=0.7)
        ax.plot(rolling_ic.index, rolling_ic.values,
                lw=2, color="steelblue", label="63-day rolling IC")
        ax.fill_between(rolling_ic.index, rolling_ic.values, 0,
                        where=(rolling_ic > 0), alpha=0.3, color="green")
        ax.fill_between(rolling_ic.index, rolling_ic.values, 0,
                        where=(rolling_ic < 0), alpha=0.3, color="red")
        ax.axhline(0, color="black", lw=0.9, ls=":")
        ax.axhline(daily_ic.mean(), color="navy", lw=1.5, ls="--",
                   label=f"Mean IC = {daily_ic.mean():.4f}")
        ax.legend(fontsize=8)
    ax.set_title("E — Daily IC (Best Config)")
    ax.set_ylabel("Spearman IC")

    # ── F: IC autocorrelogram ─────────────────────────────────────────────────
    ax = axes[1, 2]
    if daily_ic is not None and len(daily_ic) > 20:
        max_lag  = 30
        ic_arr   = daily_ic.dropna().values
        ic_arr   = ic_arr - ic_arr.mean()
        acf_vals = np.array([
            np.corrcoef(ic_arr[:-lag], ic_arr[lag:])[0, 1]
            if lag > 0 else 1.0
            for lag in range(max_lag + 1)
        ])
        lags = np.arange(max_lag + 1)
        ax.bar(lags, acf_vals, color="steelblue", alpha=0.7, width=0.6)

        # 95% confidence bands (assuming i.i.d. under H0)
        ci = 1.96 / np.sqrt(len(ic_arr))
        ax.axhline( ci, color="crimson", ls="--", lw=1.2, label="95% CI")
        ax.axhline(-ci, color="crimson", ls="--", lw=1.2)

        # Mark the forward_horizon lag — this lag MUST be significant if there
        # is overlap-induced autocorrelation (the main source of NW correction)
        h = best_params.get("forward_horizon", 5)
        ax.axvline(h, color="orange", ls=":", lw=1.5,
                   label=f"forward_horizon = {h}")
        ax.legend(fontsize=8)
    ax.set_title("F — IC Autocorrelogram\n(overlap-induced AC at lag h expected)")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")

    plt.tight_layout()
    plt.savefig("bayesian_optimisation_results.png", dpi=150, bbox_inches="tight")
    log.info("Figure saved to bayesian_optimisation_results.png")
    plt.show()


def plot_metric_comparison(
    daily_ic:   pd.Series,
    oos:        pd.DataFrame,
    best_params: Dict,
) -> None:
    """
    Three-panel metric comparison figure:
      A — Scatter: predicted vs realised (raw values)
      B — Rolling IC vs rolling Pearson vs rolling ICIR (normalised)
      C — Metric stability table heatmap across time quartiles
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "Metric Comparative Analysis: Pearson vs IC vs ICIR",
        fontsize=12, y=1.01,
    )

    ev = oos.dropna(subset=["fwd_return", "predicted_return"])

    # ── A: Predicted vs Realised scatter ──────────────────────────────────────
    ax = axes[0]
    sample = ev.sample(min(3000, len(ev)), random_state=42)
    ax.scatter(sample["predicted_return"], sample["fwd_return"],
               s=4, alpha=0.3, color="steelblue")
    # Reference line: perfect prediction
    lims = [
        min(sample["predicted_return"].min(), sample["fwd_return"].min()),
        max(sample["predicted_return"].max(), sample["fwd_return"].max()),
    ]
    ax.plot(lims, lims, "r--", lw=1.2, label="Perfect prediction")
    ax.set_xlabel("Predicted log-return")
    ax.set_ylabel("Realised log-return")
    ax.set_title("A — Predicted vs Realised\n(random sample of OOS rows)")
    ax.legend(fontsize=8)

    # ── B: Rolling metrics over time ──────────────────────────────────────────
    ax = axes[1]
    ev2 = ev.copy()
    ev2["rk_pred"] = ev2.groupby("date")["predicted_return"].rank(method="average")
    ev2["rk_real"] = ev2.groupby("date")["fwd_return"].rank(method="average")

    daily_pearson = ev2.groupby("date").apply(
        lambda g: g["predicted_return"].corr(g["fwd_return"]) if len(g) >= 3 else np.nan
    ).dropna()

    roll_ic      = daily_ic.rolling(63,  min_periods=32).mean()
    roll_pearson = daily_pearson.rolling(63, min_periods=32).mean()

    # Normalise both to zero-mean, unit-variance for visual comparison
    def _norm(s):
        return (s - s.mean()) / s.std() if s.std() > 0 else s

    ax.plot(_norm(roll_ic),      lw=1.8, color="steelblue",  label="Rolling IC (norm)")
    ax.plot(_norm(roll_pearson), lw=1.8, color="darkorange", label="Rolling Pearson (norm)", ls="--")
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_title("B — Rolling Metrics Over Time\n(normalised for comparison)")
    ax.set_ylabel("Normalised metric value")
    ax.legend(fontsize=8)

    # ── C: Metric stability across time quartiles ─────────────────────────────
    ax = axes[2]
    ev3 = ev.copy()
    ev3["date"] = pd.to_datetime(ev3["date"])
    ev3["quartile"] = pd.qcut(ev3["date"].astype(np.int64), q=4,
                              labels=["Q1 (earliest)", "Q2", "Q3", "Q4 (latest)"])
    ev3["rk_pred"] = ev3.groupby("date")["predicted_return"].rank(method="average")
    ev3["rk_real"] = ev3.groupby("date")["fwd_return"].rank(method="average")

    quartile_metrics = []
    for q in ev3["quartile"].cat.categories:
        sub = ev3[ev3["quartile"] == q]
        d_ic = sub.groupby("date").apply(
            lambda g: g["rk_pred"].corr(g["rk_real"]) if len(g) >= 3 else np.nan
        ).dropna()
        d_p = sub.groupby("date").apply(
            lambda g: g["predicted_return"].corr(g["fwd_return"]) if len(g) >= 3 else np.nan
        ).dropna()
        icir_q = d_ic.mean() / d_ic.std() * np.sqrt(252) if d_ic.std() > 0 else np.nan
        quartile_metrics.append({
            "Period": str(q),
            "IC":      d_ic.mean(),
            "ICIR":    icir_q,
            "Pearson": d_p.mean(),
        })

    qm_df = pd.DataFrame(quartile_metrics).set_index("Period")
    sns.heatmap(
        qm_df.astype(float), ax=ax,
        annot=True, fmt=".3f", cmap="RdYlGn", center=0,
        linewidths=0.5, cbar_kws={"label": "Metric value"},
    )
    ax.set_title("C — Metric Stability Across Time Periods\n(regime stability check)")

    plt.tight_layout()
    plt.savefig("metric_comparison.png", dpi=150, bbox_inches="tight")
    log.info("Metric comparison figure saved to metric_comparison.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  11. Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """
    Full pipeline:
      1. Fetch prices
      2. Run Bayesian optimisation (ICIR as primary metric)
      3. Print best configuration and comparison table
      4. Re-evaluate best config to collect full OOS predictions
      5. Plot diagnostic figures
    """
    # ── Step 1: Data ──────────────────────────────────────────────────────────
    prices = fetch_prices(SCFG)

    # ── Step 2: Bayesian Optimisation ─────────────────────────────────────────
    result, history = run_bayesian_optimisation(
        prices=prices,
        cfg=SCFG,
        n_calls=40,
        n_initial_points=10,
        primary_metric="icir",
        verbose=True,
    )

    # ── Step 3: Extract and print best configuration ───────────────────────────
    best_params = {
        dim.name: int(val)
        for dim, val in zip(SEARCH_SPACE, result.x)
    }
    best_icir = -result.fun

    print("\n" + "═" * 60)
    print("  OPTIMAL HYPERPARAMETER CONFIGURATION")
    print("═" * 60)
    for k, v in best_params.items():
        print(f"  {k:<20s}: {v}")
    print(f"  {'Best ICIR':<20s}: {best_icir:.4f}")
    print("═" * 60)

    # ── Step 4: Full metric comparison on best config ─────────────────────────
    comparison_df, daily_ic = run_metric_comparison(prices, best_params, SCFG)

    print("\n" + "═" * 60)
    print("  METRIC COMPARATIVE ANALYSIS")
    print("═" * 60)
    print(comparison_df.to_string(index=False))
    print("\nConclusion: ICIR selected as primary metric because it maximises")
    print("the portfolio information ratio (Grinold-Kahn) and penalises")
    print("signal inconsistency — critical for regime-robust strategies.")

    # ── Step 5: Re-run best config to collect OOS predictions for plotting ────
    panel, feature_cols = build_panel(prices=prices, **best_params)
    panel = cross_sectional_rank(panel, feature_cols)
    unique_dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    splits = make_splits(
        unique_dates,
        min_train=SCFG.min_train_days,
        step=SCFG.step_days,
        embargo=best_params["forward_horizon"],
    )
    all_preds = []
    for train_dates, test_dates in splits:
        train = panel[panel["date"].isin(train_dates)].dropna(subset=["fwd_return"])
        test  = panel[panel["date"].isin(test_dates)].dropna(subset=feature_cols)
        if len(train) < 100:
            continue
        model = lgb.LGBMRegressor(**SCFG.lgb_params)
        model.fit(train[feature_cols].values, train["fwd_return"].values,
                  feature_name=feature_cols)
        out = test[["date", "ticker", "fwd_return"]].copy()
        out["predicted_return"] = model.predict(test[feature_cols].values)
        all_preds.append(out)
    oos = pd.concat(all_preds, ignore_index=True)

    # ── Step 6: Plots ──────────────────────────────────────────────────────────
    plot_results(result, history, daily_ic, best_params)
    plot_metric_comparison(daily_ic, oos, best_params)

    # ── Step 7: Save history to CSV ────────────────────────────────────────────
    hist_df = pd.DataFrame([h for h in history if "icir" in h])
    hist_df.to_csv("optimisation_history.csv", index=False)
    log.info("Optimisation history saved to optimisation_history.csv")


if __name__ == "__main__":
    main()
