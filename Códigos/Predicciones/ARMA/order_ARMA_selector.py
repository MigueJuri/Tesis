import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns


# ── standard library ──────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")
import itertools
from pathlib import Path
from statsmodels.tsa.stattools       import adfuller, kpss
from statsmodels.tsa.arima.model     import ARIMA
from statsmodels.tsa.stattools       import arma_order_select_ic
from statsmodels.stats.diagnostic    import acorr_ljungbox
from statsmodels.graphics.tsaplots   import plot_acf, plot_pacf

Q_MAX = 20
P_MAX = 20

def load_data_from_csv(csv_file: str) -> pd.DataFrame:
    # yfinance CSV export uses a 2-row header for MultiIndex columns and a standalone index-name row.
    data = pd.read_csv(csv_file, header=[0, 1], skiprows=[2], index_col=0)

    data.index = pd.to_datetime(data.index)
    data.index.name = "Date"

    return data


script_dir = Path(__file__).resolve().parent
# output_file = Path(r"G:\Mi unidad\2026\Tesis\Códigos\Data\sp500_data_only_1993-01-29_to_2026-01-02.csv")  # script_dir / "sp500_data_only_1993-01-29_to_2026-01-02.csv"

# if not output_file.exists():
#     raise FileNotFoundError(
#         f"CSV not found: {output_file}. Place the file next to this script or provide an absolute path."
#     )

# data_from_csv = load_data_from_csv(output_file)
data_from_csv = load_data_from_csv(script_dir / "sp500_data_only_1993-01-29_to_2026-01-02.csv")

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

# 1. Fetch Data
log_prices = np.log(data_from_csv["Adj Close"][["SPY"]])

diff_log_prices = frac_diff_ffd(log_prices[["SPY"]], d=0.6)


import logging
import itertools
from typing import Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import arma_order_select_ic

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
P_MAX: int = 5
Q_MAX: int = 5
CRITERION = Literal["aic", "bic", "aicc"]


# ── AICc ─────────────────────────────────────────────────────────────────────
def compute_aicc(aic: float, T: int, k: int) -> float:
    """
    Hurvich & Tsai (1989) small-sample correction to AIC.

        AICc = AIC + 2k(k+1) / (T - k - 1)

    Parameters
    ----------
    aic : float   — AIC value from a fitted model.
    T   : int     — Effective sample size (observations used in fit).
    k   : int     — Number of free parameters  (p + q + 2 for ARMA with
                    intercept and variance).

    Returns
    -------
    float — AICc, or np.inf when the correction denominator T - k - 1 ≤ 0.

    Notes
    -----
    AICc → AIC as T → ∞.  Use whenever T / k < 40.
    """
    denom = T - k - 1
    return np.inf if denom <= 0 else aic + 2.0 * k * (k + 1) / denom


# ── Grid builders ─────────────────────────────────────────────────────────────
def _grid_from_statsmodels(
    s: pd.Series, T: int, p_max: int, q_max: int
) -> pd.DataFrame | None:
    """
    Fast path: delegate the full (p, q) grid fit to arma_order_select_ic,
    which avoids Python-level looping by building the IC matrices internally.
    Vectorise the AICc column with numpy after the fact.

    Returns a tidy DataFrame or None on failure.
    """
    try:
        ic_res = arma_order_select_ic(
            s,
            max_ar=p_max,
            max_ma=q_max,
            ic=["aic", "bic"],
            trend="c",
            fit_kw={"method_kwargs": {"warn_convergence": False}},
        )
    except Exception as exc:
        logger.warning("arma_order_select_ic failed (%s); falling back.", exc)
        return None

    # Stack both grids into a single tidy frame at once — no Python loop.
    aic_df = ic_res.aic.stack().rename("aic")           # MultiIndex (p, q)
    bic_df = ic_res.bic.stack().rename("bic")

    grid = pd.concat([aic_df, bic_df], axis=1)
    grid.index.names = ["p", "q"]
    grid = grid.reset_index()

    # Drop (0, 0) and any non-finite rows in one vectorised mask.
    valid = (
        ~((grid["p"] == 0) & (grid["q"] == 0))
        & np.isfinite(grid["aic"])
        & np.isfinite(grid["bic"])
    )
    grid = grid.loc[valid].copy()
    if grid.empty:
        return None

    # Vectorised AICc — no loop, operates on entire column at once.
    k_vec = grid["p"] + grid["q"] + 2          # shape (n_models,)
    denom_vec = T - k_vec - 1
    correction = np.where(
        denom_vec > 0,
        2.0 * k_vec * (k_vec + 1) / denom_vec,
        np.inf,
    )
    grid["aicc"] = grid["aic"] + correction

    return grid.reset_index(drop=True)


def _grid_manual(
    s: pd.Series, T: int, p_max: int, q_max: int
) -> pd.DataFrame | None:
    """
    Slow-path fallback: fit each ARIMA(p, 0, q) individually.
    Used only when arma_order_select_ic raises an exception.
    """
    records: list[dict] = []
    for p, q in itertools.product(range(p_max + 1), range(q_max + 1)):
        if p == 0 and q == 0:
            continue
        try:
            fit = ARIMA(s, order=(p, 0, q), trend="c").fit(
                method_kwargs={"warn_convergence": False}
            )
        except Exception as exc:
            logger.debug("ARIMA(%d,0,%d) failed: %s", p, q, exc)
            continue

        k = p + q + 2
        records.append(
            {
                "p": p,
                "q": q,
                "aic": fit.aic,
                "bic": fit.bic,
                "aicc": compute_aicc(fit.aic, T, k),
            }
        )

    return pd.DataFrame(records) if records else None


# ── Public API ────────────────────────────────────────────────────────────────
def select_order(
    series: pd.Series | pd.DataFrame | np.ndarray,
    p_max: int = P_MAX,
    q_max: int = Q_MAX,
    criterion: CRITERION = "bic",
    top_n: int = 10,
) -> tuple[int, int]:
    """
    Select the best ARMA(p, q) order over the grid [0..p_max] × [0..q_max]
    by minimising the chosen information criterion.

    Parameters
    ----------
    series    : array-like — univariate return series (NaNs are dropped).
    p_max     : int        — maximum AR order.
    q_max     : int        — maximum MA order.
    criterion : str        — one of {"aic", "bic", "aicc"}.
    top_n     : int        — number of models to log in the summary table.

    Returns
    -------
    (p, q) : tuple[int, int] — best order under the chosen criterion.

    Raises
    ------
    ValueError   — bad input types or unknown criterion.
    RuntimeError — no model converged across the entire grid.
    """
    # ── Input normalisation ──────────────────────────────────────────────────
    if isinstance(series, pd.DataFrame):
        if series.shape[1] != 1:
            raise ValueError("DataFrame input must contain exactly one column.")
        s = series.iloc[:, 0].dropna()
    elif isinstance(series, pd.Series):
        s = series.dropna()
    else:
        s = pd.Series(series).dropna()

    T = len(s)

    if criterion not in {"aic", "bic", "aicc"}:
        raise ValueError(f"criterion must be 'aic', 'bic', or 'aicc'; got {criterion!r}")

    # ── Grid construction ────────────────────────────────────────────────────
    grid = _grid_from_statsmodels(s, T, p_max, q_max) or _grid_manual(s, T, p_max, q_max)

    if grid is None or grid.empty:
        raise RuntimeError("No valid ARMA model converged in the search grid.")

    # ── Selection ────────────────────────────────────────────────────────────
    grid = grid.sort_values(criterion).reset_index(drop=True)
    best_row = grid.iloc[0]
    best_p, best_q = int(best_row["p"]), int(best_row["q"])

    # ── Logging ──────────────────────────────────────────────────────────────
    header = f"Order selection ({criterion.upper()}) — top {top_n} models:"
    divider = "-" * 48
    col_header = f"{'(p,q)':>8}  {'AIC':>10}  {'BIC':>10}  {'AICc':>10}"
    lines = [header, divider, col_header, divider]

    for _, row in grid.head(top_n).iterrows():
        tag = " ◄ best" if (int(row["p"]), int(row["q"])) == (best_p, best_q) else ""
        lines.append(
            f"({int(row['p'])},{int(row['q'])})      "
            f"{row['aic']:>10.2f}  {row['bic']:>10.2f}  {row['aicc']:>10.2f}{tag}"
        )

    logger.info("\n".join(lines))

    return best_p, best_q


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # ret = pd.Series(...)  ← inject your return series here
    order = select_order(diff_log_prices, criterion="bic")
    print(f"\nSelected order: ARMA{order}")