

---



## 9. ACF and PACF Analysis

### 9.1 The Theoretical ACF/PACF Signatures

The autocorrelation function (ACF) and partial autocorrelation function (PACF)
are the primary visual diagnostic tools for identifying ARMA order. Their theoretical
signatures are:

| Process | ACF | PACF |
|:--------|:----|:-----|
| AR$(p)$ | Decays exponentially (oscillating or monotone) | Cuts off sharply after lag $p$ |
| MA$(q)$ | Cuts off sharply after lag $q$ | Decays exponentially |
| ARMA$(p,q)$ | Decays exponentially after lag $q$ | Decays exponentially after lag $p$ |
| White Noise | All zero (within bands) | All zero (within bands) |
| $I(1)$ process | Decays very slowly, near 1 at all lags | Large spike at lag 1 |

**The Bartlett confidence band** for the ACF under the null $H_0: \rho(k) = 0$ is:

$$\pm \frac{1.96}{\sqrt{T}}$$

For $T = 5000$ (twenty years of daily data), this band is approximately
$\pm 0.028$. Any spike inside this band is statistically indistinguishable from
zero at the 5% significance level.

### 9.2 The Squared Returns ACF — Volatility Clustering

Beyond the ACF of $r_t$, we examine the ACF of $r_t^2$. If the ACF of $r_t$ is
near zero at all lags (consistent with the EMH), but the ACF of $r_t^2$ has
significant positive autocorrelation at many lags, this reveals that:

$$\text{Corr}(r_t, r_{t-k}) \approx 0 \quad \text{but} \quad \text{Corr}(r_t^2, r_{t-k}^2) \gg 0$$

This is the **stylised fact of volatility clustering**: large returns (of either sign)
tend to cluster together. It is the central empirical motivation for GARCH models
and is entirely invisible to ARMA, which models only the conditional mean.

```python
def plot_diagnostics(series, title, lags=40):
    T    = len(series.dropna())
    band = 1.96 / np.sqrt(T)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    plot_acf( series.dropna(),    lags=lags, ax=axes[0], alpha=0.05,
              title="ACF of $r_t$")
    plot_pacf(series.dropna(),    lags=lags, ax=axes[1], method="ywm",
              alpha=0.05, title="PACF of $r_t$")
    plot_acf( series.dropna()**2, lags=lags, ax=axes[2], alpha=0.05,
              title="ACF of $r_t^2$ (volatility clustering)")

    for ax in axes:
        ax.axhline(band,  color="red", ls="--", lw=0.8,
                   label=f"±{band:.3f}")
        ax.axhline(-band, color="red", ls="--", lw=0.8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

plot_diagnostics(ret, "SPY Daily Log Returns — ACF / PACF Diagnostics")
```

### 9.3 Reading the Output

When examining the three panels, look for:

1. **ACF of $r_t$:** Virtually all spikes inside the $\pm 1.96/\sqrt{T}$ band.
   This is the empirical face of the weak-form EMH. Any spike that does cross the
   band warrants investigation but should not be over-interpreted — with 40 lags
   plotted, we expect approximately $40 \times 0.05 = 2$ false positives under the
   null by chance alone.

2. **PACF of $r_t$:** Similar conclusion. If a single significant spike appears at
   lag 1 (in either the ACF or PACF), it is likely the bid-ask bounce effect.

3. **ACF of $r_t^2$:** This panel will look strikingly different — significant
   positive autocorrelation at lags 1 through 20 or more. The decay is slow and
   hyperbolic rather than exponential, consistent with long-memory volatility.
   This pattern cannot be captured by ARMA and is the primary motivation for GARCH.

---

## 10. Order Selection via Information Criteria

### 10.1 The Grid Search Procedure

We conduct a grid search over all $(p, q)$ pairs with $p \leq 5$, $q \leq 5$,
$(p,q) \neq (0,0)$, fitting an ARMA$(p, q)$ model with intercept to the full
log return series and recording AIC, BIC, and AICc.

```python
def compute_aicc(aic, T, k):
    """
    AICc corrects AIC for small-sample bias.
    k = p + q + 2 (intercept + variance).
    Reduces to AIC as T → ∞; use when T/k < 40.
    """
    denom = T - k - 1
    return np.inf if denom <= 0 else aic + 2*k*(k+1)/denom

def select_order(series, p_max=P_MAX, q_max=Q_MAX, criterion="bic"):
    s, T = series.dropna(), len(series.dropna())
    rows, best = [], {"score": np.inf, "order": (1,0)}
    col = {"aic":1, "bic":2, "aicc":3}[criterion]

    for p, q in itertools.product(range(p_max+1), range(q_max+1)):
        if p == 0 and q == 0:
            continue
        try:
            fit  = ARIMA(s, order=(p,0,q), trend="c").fit(
                       method_kwargs={"warn_convergence": False})
            k    = p + q + 2
            aicc = compute_aicc(fit.aic, T, k)
            rows.append((p, q, fit.aic, fit.bic, aicc))
            if rows[-1][col] < best["score"]:
                best = {"score": rows[-1][col], "order": (p,q)}
        except:
            continue

    rows.sort(key=lambda x: x[col])
    print(f"Order selection ({criterion.upper()}) — top 10 models:")
    print(f"{'(p,q)':>8s}  {'AIC':>10s}  {'BIC':>10s}  {'AICc':>10s}")
    for row in rows[:10]:
        tag = " ◄ best" if (row[0],row[1]) == best["order"] else ""
        print(f"({row[0]},{row[1]}){' ':4s}  {row[2]:>10.2f}  "
              f"{row[3]:>10.2f}  {row[4]:>10.2f}{tag}")
    return best["order"]

order = select_order(ret, criterion="bic")
```

### 10.2 What the Output Will Show — and Why

The BIC grid will almost certainly select a small model: ARMA$(0,1)$, ARMA$(1,0)$,
or ARMA$(1,1)$, with the winning order varying slightly across data periods. This
result has the following interpretation:

- **If BIC selects ARMA$(0,0)$:** The return series is indistinguishable from
  white noise. The ARMA model adds nothing beyond the constant mean. This is the
  strongest form of the EMH null.

- **If BIC selects ARMA$(1,0)$:** There is weak AR(1) structure. The coefficient
  $\hat{\phi}_1$ will be small (typically $|\hat{\phi}_1| < 0.05$) and may or may
  not be statistically significant. This could reflect micro-structural effects
  (non-synchronous trading) or genuine but weak short-term momentum.

- **If BIC selects ARMA$(0,1)$:** There is weak MA(1) structure. The coefficient
  $\hat{\theta}_1$ negative and small suggests bid-ask bounce; positive and small
  suggests lagged price discovery.

The AIC will select a slightly larger model than BIC in most cases — this difference
illustrates concretely why BIC is preferred for near-white-noise financial returns.

---

## 11. Window Sensitivity Analysis

### 11.1 Motivation

Before committing to $W_{\text{train}} = 504$ days, we verify empirically that
this choice is reasonable by computing the out-of-sample Information Coefficient
(IC) for each candidate window. The IC is the Pearson correlation between
predicted and realised returns:

$$\text{IC}(W) = \text{Corr}\!\left(\hat{r}_{t+1}^{(W)},\, r_{t+1}\right)$$

The window $W^*$ that maximises $|\text{IC}(W)|$ out-of-sample is the empirically
optimal choice. If $|\text{IC}(W)|$ is near zero for all $W$, the ARMA mean has no
predictive power — and the window choice is irrelevant.

```python
def window_sensitivity(log_ret, order, window_grid=WINDOW_GRID,
                        horizon=1, step=21):
    print(f"{'Window':>8s}  {'OOS IC':>10s}  {'t-stat':>10s}  {'N':>6s}")
    rows = []
    for W in window_grid:
        preds, reals = [], []
        T = len(log_ret)
        t = W
        while t + horizon <= T:
            train = log_ret.iloc[t-W:t]
            try:
                fit   = ARIMA(train, order=(order[0],0,order[1]),
                              trend="c").fit(
                                  method_kwargs={"warn_convergence":False})
                pred  = fit.forecast(steps=horizon).sum()
            except:
                pred = train.mean() * horizon
            reals.append(log_ret.iloc[t:t+horizon].sum())
            preds.append(pred)
            t += step

        preds, reals = np.array(preds), np.array(reals)
        N  = len(preds)
        ic = np.corrcoef(preds, reals)[0,1] if N > 2 else 0.0
        ts = ic * np.sqrt(N-2) / np.sqrt(max(1-ic**2, 1e-9))
        rows.append({"window":W, "IC":ic, "t-stat":ts, "N":N})
        print(f"{W:>8d}  {ic:>+10.5f}  {ts:>10.3f}  {N:>6d}")
    return pd.DataFrame(rows)

win_sens = window_sensitivity(ret, order)
```

### 11.2 Interpreting the Sensitivity Table

Examine the IC and its $t$-statistic for each window:

- **IC $\approx 0$ for all windows:** Confirms that ARMA has no predictive power
  for the conditional mean at any window. This is the expected result and correctly
  motivates nonlinear models.

- **IC peaks at a particular $W$:** Adopt that window. The peak reflects the
  approximate stationarity horizon — the time scale over which the ARMA structure
  is most stable.

- **IC $t$-statistic $> 2$:** The IC is statistically significant at the 5% level.
  With $N \approx 200$–$500$ walk-forward forecasts, this threshold corresponds
  to $|IC| > 0.10$–$0.14$. Very few linear models achieve this on index returns.

---

## 12. Model Fitting and Coefficient Interpretation

### 12.1 The Coefficient Table

```python
def fit_and_diagnose(series, order):
    fit   = ARIMA(series, order=(order[0],0,order[1]),
                  trend="c").fit(
                      method_kwargs={"warn_convergence":False})
    resid = pd.Series(fit.resid.values).dropna()
    params, bse, pvals = fit.params, fit.bse, fit.pvalues

    print(f"ARMA{order} on {TICKER} log returns\n")
    print(f"{'Parameter':>12s}  {'Estimate':>12s}  {'Std Err':>10s}  "
          f"{'t-stat':>8s}  {'p-value':>8s}")
    for name, est, se, pv in zip(params.index, params, bse, pvals):
        sig = "**" if pv < 0.01 else ("*" if pv < 0.05 else "")
        print(f"{name:>12s}  {est:>+12.6f}  {se:>10.6f}  "
              f"{est/se:>8.3f}  {pv:>8.4f} {sig}")
    print(f"\nLog-L: {fit.llf:.2f}  AIC: {fit.aic:.2f}  BIC: {fit.bic:.2f}")
    return fit, resid

train_end = int(len(ret) * 0.70)
fit, resid = fit_and_diagnose(ret.iloc[:train_end], order)
```

### 12.2 Reading the Coefficient Table

For each estimated coefficient, the output provides:

**Estimate $\hat{\phi}_i$ or $\hat{\theta}_j$:** The point estimate of the
coefficient. For SPY daily log returns, expect all AR and MA coefficients to
satisfy $|\hat{\phi}_i|, |\hat{\theta}_j| < 0.10$. Values close to zero confirm
near-white-noise behaviour.

**Standard Error $\text{se}(\hat{\phi}_i)$:** Derived from the Fisher information
matrix. Proportional to $1/\sqrt{T}$ — larger samples produce tighter estimates.

**$t$-statistic $= \hat{\phi}_i / \text{se}(\hat{\phi}_i)$:** Under $H_0: \phi_i = 0$,
this follows a standard normal asymptotically. Reject at the 5% level if
$|t| > 1.96$. For near-white-noise returns, most coefficients will have $|t| < 2$,
confirming that the estimated structure is statistically fragile.

**Unconditional mean:** Computed as $\hat{\mu} = \hat{c} / (1 - \sum_i \hat{\phi}_i)$.
This should equal approximately the sample mean of the return series.

---

## 13. Residual Diagnostics

### 13.1 The Three-Test Battery

A correctly specified ARMA model has residuals $\hat{\epsilon}_t$ that are
**white noise** and (approximately) **Gaussian**. We test these two properties
with three formal tests, listed with their precise null hypotheses:

**Test 1 — Ljung-Box on $\hat{\epsilon}_t$** (tests mean model adequacy):

$$H_0: \rho(1) = \rho(2) = \cdots = \rho(m) = 0 \quad \text{(residuals are white noise)}$$

$$Q(m) = T(T+2) \sum_{k=1}^{m} \frac{\hat{\rho}_k^2}{T-k} \overset{H_0}{\sim} \chi^2(m - p - q)$$

A $p$-value $> 0.05$ means the ARMA model has captured all *linear* structure.
A $p$-value $< 0.05$ means residual autocorrelation remains — the model is
under-specified (increase $p$ or $q$).

**Test 2 — Ljung-Box on $\hat{\epsilon}_t^2$** (tests variance model adequacy):

$$H_0: \rho(1) = \cdots = \rho(m) = 0 \quad \text{on squared residuals}$$

A significant result ($p < 0.05$) means squared residuals are autocorrelated —
the variance is time-varying (ARCH effects). This test is expected to **fail**
for equity returns, and the failure directly motivates GARCH as the next model.

**Test 3 — Jarque-Bera on $\hat{\epsilon}_t$** (tests distributional adequacy):

$$H_0: \hat{\epsilon}_t \sim \mathcal{N}(0, \sigma^2)$$

Expected to fail decisively, confirming fat tails in the innovation distribution.
This motivates Student-$t$ or skew-$t$ innovations in the GARCH extension.

```python
def residual_tests(resid):
    lb   = acorr_ljungbox(resid,    lags=[5,10,20], return_df=True)
    lb2  = acorr_ljungbox(resid**2, lags=[5,10,20], return_df=True)
    jb_stat, jb_p = stats.jarque_bera(resid)

    print("Ljung-Box on residuals ε̂_t  (expect p > 0.05):")
    for lag in [5,10,20]:
        p = lb.loc[lag,"lb_pvalue"]
        print(f"  lag={lag:2d}: Q={lb.loc[lag,'lb_stat']:.2f}  "
              f"p={p:.4f}  {'✓ white noise' if p>0.05 else '✗ structure remains'}")

    print("\nLjung-Box on ε̂_t² (expect p < 0.05 — ARCH effects):")
    for lag in [5,10,20]:
        p = lb2.loc[lag,"lb_pvalue"]
        print(f"  lag={lag:2d}: Q={lb2.loc[lag,'lb_stat']:.2f}  "
              f"p={p:.4f}  {'✗ ARCH detected → GARCH needed' if p<0.05 else '✓'}")

    print(f"\nJarque-Bera: stat={jb_stat:.1f}  p={jb_p:.2e}  "
          f"skew={resid.skew():.3f}  kurt={resid.kurt():.3f}")
    print("Expected: reject normality (fat tails confirm GARCH-t motivation)")

residual_tests(resid)
```

### 13.2 The Diagnostic Logic

The residual diagnostics collectively constitute a **specification test** for the
ARMA model. The pattern of results tells a precise scientific story:

- **Ljung-Box on $\hat{\epsilon}_t$ passes:** The ARMA model has extracted all
  exploitable linear mean structure. There is no more linear predictability to be
  found. Any remaining predictability must be nonlinear.

- **Ljung-Box on $\hat{\epsilon}_t^2$ fails:** The second moment of returns is
  predictable even when the first moment is not. This is the GARCH phenomenon.
  It is not a failure of the ARMA model — it is a finding about the structure of
  equity return variance.

- **Jarque-Bera fails:** The innovation distribution has heavy tails. A GARCH
  model with Student-$t$ innovations — rather than Gaussian — is required for
  correct inference and risk measurement.

---

## 14. Impulse Response Function

### 14.1 Definition and Computation

The **Impulse Response Function (IRF)** $\{\psi_k\}_{k=0}^{\infty}$ characterises
the dynamic effect of a unit shock $\epsilon_t = 1$ on future values of $r_{t+k}$.
It is the MA$(\infty)$ representation coefficient at lag $k$:

$$r_t = \mu + \sum_{k=0}^{\infty} \psi_k\, \epsilon_{t-k}, \quad \psi_0 = 1$$

The IRF coefficients are computed via the recursive formula:

$$\psi_0 = 1$$

$$\psi_k = \sum_{i=1}^{p} \hat{\phi}_i\, \psi_{k-i} + \hat{\theta}_k \qquad (k \geq 1, \text{ with } \hat{\theta}_k = 0 \text{ for } k > q)$$

For a stationary ARMA, $\psi_k \to 0$ exponentially fast as $k \to \infty$.
The speed of decay measures how long a market shock persists in the return series.

```python
def compute_irf(order, fit, periods=30):
    """
    Compute IRF from fitted ARMA coefficients.
    phi: AR coefficients [φ_1, ..., φ_p]
    theta: MA coefficients [θ_1, ..., θ_q]
    """
    p_names = [n for n in fit.params.index if n.startswith("ar.")]
    q_names = [n for n in fit.params.index if n.startswith("ma.")]
    phi   = fit.params[p_names].values if p_names else np.zeros(0)
    theta = fit.params[q_names].values if q_names else np.zeros(0)
    p, q  = len(phi), len(theta)

    psi = np.zeros(periods)
    for k in range(periods):
        val  = (theta[k] if k < q else 0.0)
        val += (1.0      if k == 0 else 0.0)
        for i in range(1, min(p, k) + 1):
            val += phi[i-1] * psi[k-i]
        psi[k] = val
    return psi

irf = compute_irf(order, fit)

# Plot
fig, ax = plt.subplots(figsize=(10, 4))
colors  = ["#d62728" if v < 0 else "#1f77b4" for v in irf]
ax.bar(range(len(irf)), irf, color=colors, alpha=0.8, width=0.7)
ax.axhline(0, color="black", lw=0.6)
ax.set_xlabel("Lag $k$ (days)")
ax.set_ylabel("$\\psi_k$ — effect of unit shock on $r_{t+k}$")
ax.set_title(f"Impulse Response Function — ARMA{order}")
ax.grid(True, alpha=0.25, axis="y")
plt.tight_layout()
plt.show()
```

### 14.2 Interpreting the IRF

The IRF translates the abstract ARMA coefficients into an economically interpretable
narrative about **how market shocks propagate through prices**:

- **$\psi_0 = 1$:** By construction. A shock today has a full unit effect on
  today's return.

- **$\psi_1 > 0$:** The shock has a positive effect the next day — consistent
  with under-reaction or momentum.

- **$\psi_1 < 0$:** The shock reverses the next day — consistent with over-reaction
  and bid-ask bounce.

- **$\psi_k \approx 0$ for $k \geq 2$:** The shock has negligible effect beyond
  one day. For a well-functioning liquid market (SPY), this is the expected result
  — shocks are incorporated into prices within one trading day.

The **half-life of the impulse response** is the smallest $k$ such that
$|\psi_k| < 0.5$. For SPY, this will typically be $k = 1$ or $k = 2$ days,
confirming rapid price discovery.

---

## 15. Walk-Forward Evaluation

### 15.1 Why Standard Cross-Validation is Invalid

Standard $k$-fold cross-validation randomly assigns observations to training and
validation folds. For time series, this allows **future information to leak into
the training set** — an observation at time $t+5$ used for training contains
information not available at time $t$. The resulting performance estimates are
optimistically biased and scientifically invalid.

The correct procedure is **walk-forward validation** (also called time-series
cross-validation or rolling origin evaluation):

```
WALK-FORWARD PROTOCOL:

t = W_train:
  Train on [0, W_train)
  Forecast horizon h
  Record predicted vs realised
  Advance by W_retrain
  Repeat until end of series

Timeline:
|─────────────────────────────────────────────────────────────|
| TRAIN [t-W, t) | EMBARGO (h days) | EVALUATE [t, t+h) |
```

The **embargo** of $h$ days between the training window end and the evaluation
window start is mandatory. Without the embargo, the training and evaluation return
windows can overlap — for a 21-day horizon, adjacent observations share 20 days
of return, creating near-perfect spurious correlation.

### 15.2 Direct Multi-Horizon Forecasting

For each horizon $h \in \{1, 5, 10, 21\}$, the ARMA model is applied directly
as a $h$-step forecaster:

$$\hat{R}_{t:t+h} = \hat{\mathbb{E}}\!\left[\sum_{k=1}^h r_{t+k} \;\middle|\; \mathcal{F}_t\right] \approx \sum_{k=1}^h \hat{r}_{t+k|t}$$

The cumulative $h$-period log return is the target, and the sum of $h$ one-step
forecasts from the fitted model is the prediction.

```python
def walk_forward(log_ret, order, W=W_TRAIN,
                 horizons=HORIZONS, step=W_RETRAIN):
    results = {}
    for h in horizons:
        records = []
        T = len(log_ret)
        t = W
        while t + h <= T:
            train = log_ret.iloc[t-W:t]
            try:
                fit   = ARIMA(train, order=(order[0],0,order[1]),
                              trend="c").fit(
                                  method_kwargs={"warn_convergence":False})
                pred  = fit.forecast(steps=h).sum()
            except:
                pred  = train.mean() * h
            records.append({
                "date"     : log_ret.index[t],
                "arma_pred": pred,
                "rw_pred"  : train.mean() * h,   # Random walk with drift
                "zero_pred": 0.0,                 # Pure EMH benchmark
                "realized" : log_ret.iloc[t:t+h].sum()
            })
            t += step
        results[h] = pd.DataFrame(records).set_index("date")
    return results

results = walk_forward(ret, order)
```

---

## 16. Performance Scorecard

### 16.1 The Metric Suite

Each metric in the scorecard measures a distinct dimension of predictive quality.
No single metric is sufficient in isolation.

**Information Coefficient (IC):**

$$\text{IC} = \text{Corr}(\hat{r}_{t+h}, r_{t+h})$$

The IC measures the linear association between predictions and outcomes. It is
the most common evaluation metric in quantitative finance. Interpretation thresholds:

| $|\text{IC}|$ | Interpretation |
|:-------------:|:---------------|
| $< 0.02$ | No meaningful predictive power |
| $0.02$–$0.05$ | Weak signal (marginal) |
| $0.05$–$0.10$ | Moderate signal (actionable) |
| $> 0.10$ | Strong signal (rare for linear models on equities) |

The $t$-statistic of the IC tests $H_0: \text{IC} = 0$:

$$t_{\text{IC}} = \text{IC} \cdot \frac{\sqrt{N-2}}{\sqrt{1 - \text{IC}^2}} \overset{H_0}{\sim} t(N-2)$$

**Theil's U statistic:**

$$U = \sqrt{\frac{\text{MSE}_{\text{model}}}{\text{MSE}_{\text{RW}}}}$$

$U < 1$ means the model beats the random walk benchmark in mean squared error.
$U = 1$ means it matches the benchmark. $U > 1$ means the model is worse than
simply using the historical mean drift as a forecast.

**Long-Short Sharpe Ratio:**

$$\text{SR}_{\text{L/S}} = \frac{\mathbb{E}[\text{sign}(\hat{r}_{t+h}) \cdot r_{t+h}]}{\text{Std}[\text{sign}(\hat{r}_{t+h}) \cdot r_{t+h}]} \cdot \sqrt{f}$$

where $f$ is the annualisation factor ($f = 252/h$). This is the Sharpe ratio of
a strategy that goes long if $\hat{r}_{t+h} > 0$ and short otherwise. It does
**not** account for transaction costs — a positive SR here does not imply a
profitable strategy after costs.

```python
def scorecard(results):
    ann = {1:np.sqrt(252), 5:np.sqrt(52), 10:np.sqrt(26), 21:np.sqrt(12)}
    print(f"{'h':>4s}  {'Model':>9s}  {'IC':>8s}  {'IC t':>7s}  "
          f"{'HitRate':>8s}  {'TheilU':>8s}  {'SR L/S':>8s}  {'N':>5s}")
    for h, df in results.items():
        real = df["realized"].values
        N    = len(real)
        for col in ["arma_pred","rw_pred","zero_pred"]:
            pred = df[col].values
            ic   = np.corrcoef(pred, real)[0,1] if pred.std()>1e-10 else 0.
            ts   = ic*np.sqrt(N-2)/np.sqrt(max(1-ic**2,1e-9))
            hit  = np.mean(np.sign(pred)==np.sign(real))
            theilu = np.sqrt(np.mean((pred-real)**2) /
                             np.mean((df["rw_pred"].values-real)**2))
            ret_ls = np.sign(pred)*real
            sr   = ret_ls.mean()/ret_ls.std()*ann[h] if ret_ls.std()>1e-10 else 0.
            sig  = "*" if abs(ts) > 2 else ""
            print(f"{h:>4d}  {col[:9]:>9s}  {ic:>+8.4f}  "
                  f"{ts:>7.3f}{sig:1s} {hit:>8.3f}  "
                  f"{theilu:>8.4f}  {sr:>8.4f}  {N:>5d}")
        print()

scorecard(results)
```

### 16.2 The Base Rate Trap

When examining the Hit Rate column, compare it against the **base rate** — the
directional accuracy achieved by always predicting the modal direction (always
"up" for equities):

$$\text{Base Rate} = P(r_{t+h} > 0) \approx \begin{cases} 0.53 & h = 1 \text{ day} \\ 0.58 & h = 5 \text{ days} \\ 0.62 & h = 21 \text{ days} \end{cases}$$

A model with Hit Rate $= 0.60$ at a 21-day horizon is not demonstrating 60%
directional accuracy — it is demonstrating *approximately zero* improvement over
the trivial "always long" strategy. The correct comparison is:

$$\text{Adjusted Hit Rate} = \text{Hit Rate} - \text{Base Rate}$$

Only when $\text{Adjusted Hit Rate} > 0$ does the model add directional value
beyond the unconditional equity premium.

---

## 17. Conclusions and Motivation for Nonlinear Extensions

### 17.1 What the ARMA Analysis Establishes

Upon completing this notebook, the following empirical facts are established with
formal statistical evidence:

1. **$\log P_t \sim I(1)$:** Confirmed by ADF (fail to reject unit root) and KPSS
   (reject stationarity). The random walk is the appropriate null model for prices.

2. **$r_t \sim I(0)$:** Confirmed by ADF (strongly reject unit root). ARMA is a
   valid model class for log returns.

3. **Near-zero linear autocorrelation of $r_t$:** ACF and PACF of log returns show
   no significant structure beyond Bartlett confidence bands. BIC selects a small
   ARMA model with economically negligible coefficients.

4. **Significant nonlinear autocorrelation of $r_t^2$:** ACF of squared returns
   shows strong, persistent autocorrelation at lags 1–20+. The ARCH test on ARMA
   residuals rejects the null of no conditional heteroskedasticity decisively.

5. **Fat-tailed innovation distribution:** Jarque-Bera rejects Gaussianity for
   ARMA residuals. Empirical tail quantiles substantially exceed Gaussian predictions.

6. **Theil's $U \geq 1$ for ARMA:** The ARMA model does not improve on the random
   walk with drift benchmark in MSE terms at any tested horizon.

### 17.2 The Scientific Value of These Results

These results are not a failure — they are a **precise characterisation of where
linear models fail and why**. They establish four research directions:

| Finding | Implication | Next Model |
|:--------|:------------|:-----------|
| ACF($r_t$) $\approx 0$ | No linear mean structure | Nonlinear features (GBM) |
| ACF($r_t^2$) $\gg 0$ | Variance is predictable | GARCH |
| Fat-tailed residuals | Gaussian is misspecified | GARCH-$t$, EVT |
| Theil $U \geq 1$ | RW is unbeaten in mean | Cross-sectional signals |

The path of the research program is now motivated by the data rather than
imposed by convention. The GARCH model is not introduced because it is fashionable —
it is introduced because the Ljung-Box test on $\hat{\epsilon}_t^2$ rejects with
$p \ll 0.001$. The nonlinear GBM model is not introduced because ARMA is old —
it is introduced because ARMA's IC is statistically indistinguishable from zero.

### 17.3 Next Steps

```
Chapter 2: GARCH Variance Model
    Motivation: ARCH test on ε̂_t² (this chapter)
    Model:      ARMA(p,q)-GJR-GARCH(1,1)-t
    New target: σ̂²_{t+h} (conditional variance forecast)

Chapter 3: Fractional Differentiation
    Motivation: Long memory in ACF(r_t²), Hurst H > 0.5
    Tool:       Minimum-d fractional differencing
    Purpose:    Preserve memory structure in features

Chapter 4: Cross-Sectional Factor Model
    Motivation: IC ≈ 0 for univariate ARMA (this chapter)
    Universe:   S&P 500 constituents (N ≈ 400)
    Signal:     Cross-sectional momentum, value, quality

Chapter 5: Gradient Boosting Ensemble
    Motivation: Nonlinear feature interactions
    Features:   All of Chapters 2–4 combined
    Validation: Deflated Sharpe Ratio (Bailey-López de Prado)
```

> **A note on intellectual honesty.** The finding that ARMA fails to beat a
> random walk on S&P 500 daily returns is not a deficiency of this thesis —
> it is its first result. A research program that begins by rigorously establishing
> what *does not work*, and why, before proposing what might work is following the
> scientific method. The threshold for claiming a result is real must be high:
> statistical significance alone is insufficient; economic significance after
> transaction costs and out-of-sample validation are the standards this thesis
> holds itself to.

---

*End of Chapter 1 — Linear Time Series Baseline*

---

> **References**
>
> - Box, G.E.P. & Jenkins, G.M. (1976). *Time Series Analysis: Forecasting and Control.* Holden-Day.
> - Dickey, D.A. & Fuller, W.A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *JASA*, 74(366), 427–431.
> - Kwiatkowski, D. et al. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. *Journal of Econometrics*, 54(1–3), 159–178.
> - Lo, A.W. & MacKinlay, A.C. (1988). Stock market prices do not follow random walks. *Review of Financial Studies*, 1(1), 41–66.
> - López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
> - Mandelbrot, B. (1963). The variation of certain speculative prices. *Journal of Business*, 36(4), 394–419.
> - Wold, H. (1938). *A Study in the Analysis of Stationary Time Series.* Uppsala: Almqvist & Wiksell.
