# Hyperparameter Optimisation for the Three-Factor GBDT Predictor

## `hparam_optimization.py` — Technical Documentation

---

## 1. Motivation and Scope

The main notebook (`TreeBasedPredictor_v2.ipynb`) fixes the GBDT architecture and feature engineering pipeline, but uses **hand-chosen** feature-engineering hyperparameters:

| Parameter | Fixed value | Meaning |
|-----------|-------------|---------|
| `forward_horizon` | 5 days | How far ahead to predict |
| `mom_short` | 1 day | Short momentum lookback |
| `mom_medium` | 5 days | Medium momentum lookback |
| `mom_long` | 21 days | Long momentum lookback |
| `vol_window` | 21 days | Rolling variance window |
| `ema_span` | 21 days | EMA z-score span |

These values are reasonable priors but not necessarily optimal for this dataset, universe, or time period. This script treats them as **search variables** and finds the configuration that maximises a chosen validation metric using Bayesian Optimisation.

> **Important separation of concerns.** The GBDT architecture hyperparameters (`n_estimators`, `learning_rate`, `max_depth`, etc.) are **not** optimised here. Jointly optimising model architecture and feature engineering creates a search space so large that even Bayesian Optimisation overfits. We fix the model at its regularised defaults and search only over the feature space.

---

## 2. Mathematical Framework

### 2.1 The Optimisation Problem

Let $\boldsymbol{\theta} \in \Theta$ be the vector of feature-engineering hyperparameters:

$$\boldsymbol{\theta} = (h, k_s, k_m, k_l, w_v, w_z)$$

where $h$ is the prediction horizon, $k_s < k_m < k_l$ are the three momentum windows, $w_v$ is the volatility lookback, and $w_z$ is the EMA span.

We seek:

$$\boldsymbol{\theta}^* = \arg\max_{\boldsymbol{\theta} \in \Theta} \;\text{ICIR}_{\text{OOS}}(\boldsymbol{\theta})$$

where $\text{ICIR}_{\text{OOS}}$ is computed on held-out data under the walk-forward scheme. This is a **black-box optimisation problem**: the objective has no known gradient and each evaluation is expensive (one full walk-forward cross-validation run).

### 2.2 Momentum Feature Construction

For each prediction horizon $h$ and lookback $k$, the momentum feature is the cumulative log-return with a one-day skip:

$$f_{t,i}^{\text{mom},k} = \log P_{t-1,i} - \log P_{t-k,i}$$

The one-day skip avoids microstructure reversal contamination. The three momentum windows $k_s < k_m < k_l$ capture short, medium, and long-term price continuation effects respectively. The constraint $k_s < k_m < k_l$ is enforced explicitly — configurations that violate it return a large penalty score so the GP surrogate learns to avoid that region.

### 2.3 Target Variable

$$y_{t,i} = \log \frac{P_{t+h,i}}{P_{t,i}} = \sum_{s=1}^{h} r_{t+s,i}$$

The target is the continuous $h$-day forward log-return. After cross-sectional ranking of features, the GBDT learns to predict the expected ranking of assets by their forward returns.

---

## 3. Bayesian Optimisation

### 3.1 Why Bayesian Optimisation Over Grid/Random Search?

| Method | Evaluations needed | Handles correlations | Uses past evaluations |
|--------|-------------------|---------------------|-----------------------|
| Grid search | Exponential in dimensions | No | No |
| Random search | $O(1/\varepsilon^2)$ | No | No |
| **Bayesian Optimisation** | **$O(\text{poly}(d))$** | **Yes** | **Yes** |

Each walk-forward evaluation takes several minutes. With 6 parameters and reasonable grid resolution (5 values each), grid search would require $5^6 = 15{,}625$ evaluations — computationally infeasible. Bayesian Optimisation achieves near-optimal results in 40–80 evaluations by using a probabilistic surrogate model to decide where to evaluate next.

### 3.2 Gaussian Process Surrogate

Bayesian Optimisation maintains a Gaussian Process (GP) posterior over the objective function:

$$f(\boldsymbol{\theta}) \sim \mathcal{GP}\left(\mu(\boldsymbol{\theta}),\, k(\boldsymbol{\theta}, \boldsymbol{\theta}')\right)$$

After $n$ evaluations $\{(\boldsymbol{\theta}_i, y_i)\}_{i=1}^n$, the GP posterior is updated analytically. The posterior mean $\mu_n(\boldsymbol{\theta})$ is the surrogate's best guess for the objective at any untested point, and the posterior variance $\sigma_n^2(\boldsymbol{\theta})$ quantifies uncertainty.

**Kernel choice: Matérn 5/2**

$$k_{5/2}(\mathbf{x}, \mathbf{x}') = \sigma^2\left(1 + \frac{\sqrt{5}r}{\ell} + \frac{5r^2}{3\ell^2}\right) \exp\!\left(-\frac{\sqrt{5}r}{\ell}\right), \quad r = \|\mathbf{x} - \mathbf{x}'\|$$

The Matérn 5/2 kernel is twice continuously differentiable — smoother than Matérn 3/2 (once differentiable) but rougher than the squared-exponential (infinitely differentiable). For integer-valued hyperparameter surfaces that may have step-like behaviour (the signal is qualitatively different at $h=5$ vs $h=6$), the squared-exponential kernel over-smooths and misses important transitions. Matérn 5/2 is the standard choice for hyperparameter optimisation.

### 3.3 Acquisition Function: Expected Improvement

The next evaluation point is chosen by maximising the **Expected Improvement (EI)**:

$$\text{EI}(\boldsymbol{\theta}) = \mathbb{E}\!\left[\max\!\left(f(\boldsymbol{\theta}) - f^+,\, 0\right)\right]$$

where $f^+ = \max_i y_i$ is the current best observation. Under the GP posterior, this has a closed form:

$$\text{EI}(\boldsymbol{\theta}) = (\mu_n(\boldsymbol{\theta}) - f^+)\,\Phi(Z) + \sigma_n(\boldsymbol{\theta})\,\phi(Z), \quad Z = \frac{\mu_n(\boldsymbol{\theta}) - f^+}{\sigma_n(\boldsymbol{\theta})}$$

where $\Phi$ and $\phi$ are the standard normal CDF and PDF respectively.

EI is preferred over UCB (Upper Confidence Bound) for financial applications because:
- EI naturally balances exploration (high $\sigma_n$) and exploitation (high $\mu_n$)
- UCB requires tuning a $\kappa$ parameter that controls this trade-off — one more hyperparameter to set
- EI is bounded below at zero, matching the intuition that only improvements matter

### 3.4 Initial Random Exploration

The first `n_initial_points = 10` evaluations use **Latin Hypercube Sampling (LHS)** to ensure the search space is well-covered before the GP takes over. LHS partitions each dimension into $n$ equal intervals and samples one point from each row/column combination — much better space-filling than pure random sampling, which tends to cluster.

---

## 4. Walk-Forward Validation with Embargo

### 4.1 The Validation Scheme

The validation scheme matches the main notebook exactly, parameterised by the current `forward_horizon`:

$$\mathcal{T}^{\text{train}}_s = \{t : t \leq t_s - h\}, \qquad \mathcal{T}^{\text{test}}_s = \{t : t_s < t \leq t_s + \Delta\}$$

The embargo gap equals `forward_horizon` to prevent the labels for the last $h$ training observations from overlapping with the test window.

### 4.2 Why Not $k$-Fold Cross-Validation?

Standard $k$-fold CV randomly shuffles data, mixing past and future observations in each fold. For financial time series, this is a form of **temporal leakage**: the model sees future prices during training. Walk-forward CV preserves causal ordering at the cost of fewer effective folds, but this trade-off is non-negotiable for financial applications.

### 4.3 Avoiding Look-Ahead in the Optimisation Itself

A subtle form of look-ahead would occur if the optimiser were allowed to select hyperparameters by evaluating the **entire** available time series. The Bayesian optimiser would effectively be fitting hyperparameters to the full in-sample data distribution.

This is mitigated here by keeping `min_train_days = 504` (two years) and `step_days = 63` (one quarter) fixed: the same splits are used for every trial, so the optimiser is always comparing OOS performance on the same held-out windows. The hyperparameter selection is based only on the OOS portion.

---

## 5. Metric Comparative Analysis

### 5.1 Pearson Correlation

$$\rho^{\text{P}}_t = \frac{\sum_i (\hat{y}_{t,i} - \bar{\hat{y}}_t)(y_{t,i} - \bar{y}_t)}{\sqrt{\sum_i (\hat{y}_{t,i} - \bar{\hat{y}}_t)^2} \cdot \sqrt{\sum_i (y_{t,i} - \bar{y}_t)^2}}$$

**Problems in financial time series:**

1. **Heavy tails.** Financial returns have kurtosis far above 3. A single day with a large market move can dominate the correlation estimate for an entire year. Pearson is not robust to this.

2. **Scale sensitivity.** The magnitude of $\hat{y}$ depends on the model's calibration. Two models with identical ranking ability but different output scales get different Pearson correlations.

3. **No regime penalty.** A model that is highly predictive in calm markets and anti-predictive in crises has a Pearson-averaged metric that looks reasonable but is practically useless.

**Verdict: Not appropriate as a primary selection metric.**

### 5.2 IC — Mean Cross-Sectional Spearman Rank Correlation

$$\text{IC}_t = \rho^S(\hat{y}_{t,\cdot},\, y_{t,\cdot}) = 1 - \frac{6 \sum_i d_i^2}{N_t(N_t^2 - 1)}$$

where $d_i = \text{rank}(\hat{y}_{t,i}) - \text{rank}(y_{t,i})$.

**Advantages over Pearson:**

1. **Outlier robustness.** Rank transformation bounds the influence of any single extreme observation.

2. **Scale invariance.** $\text{IC}_t$ is identical whether the model predicts $\hat{y}$ or $c\hat{y}$ for any constant $c > 0$.

3. **Directly interpretable.** $\text{IC}_t > 0$ means the model ranks assets correctly more often than chance on day $t$.

4. **Consistent with portfolio construction.** A long-short strategy formed by ranking assets by $\hat{y}$ extracts alpha in proportion to $\text{IC}$ (Grinold-Kahn law).

**Limitation:** Mean IC ignores the *variance* of IC over time. A model with $\text{IC} \in \{-0.10, +0.20\}$ on alternating days has mean IC $= 0.05$ — the same as a model with $\text{IC} = 0.05$ every day. But the former requires position sizing that accommodates large sign reversals, making it far less practical.

**Verdict: Valid but incomplete.**

### 5.3 ICIR — IC Information Ratio

$$\text{ICIR} = \frac{\bar{\text{IC}}}{\hat{\sigma}_{\text{IC}}} \cdot \sqrt{252}$$

This is the IC Sharpe ratio — signal strength normalised by signal consistency.

**The Grinold-Kahn connection.** The Fundamental Law of Active Management states:

$$\text{IR} \approx \text{IC} \cdot \sqrt{N_{\text{bets}}}$$

where $\text{IR}$ is the portfolio information ratio and $N_{\text{bets}}$ is the number of independent bets per year. Maximising $\text{ICIR}$ is equivalent to maximising $\text{IC} / \sigma_{\text{IC}}$, which — for a fixed $N_{\text{bets}}$ — directly maximises the portfolio IR.

**Regime robustness.** A model with high ICIR delivers consistent signal strength across market regimes. A model with high mean IC but low ICIR is regime-dependent — it works brilliantly in some periods and fails in others. For practical portfolio management, consistency is more valuable than peak performance.

**Newey-West correction.** Due to the $h$-day forward return window, consecutive IC observations share $h - 1 = 4$ overlapping data points. The naive standard error $\hat{\sigma}_\text{IC} / \sqrt{T}$ is anti-conservative. The NW-corrected ICIR should be used for significance testing:

$$\text{ICIR}_{\text{NW}} = \frac{\bar{\text{IC}}}{\sqrt{\hat{\sigma}^2_{\text{NW}} / T}}$$

where $\hat{\sigma}^2_{\text{NW}} = \hat{\gamma}_0 + 2\sum_{j=1}^{q}(1 - j/(q+1))\hat{\gamma}_j$ with $q \geq h - 1$.

**Verdict: Primary selection metric. ICIR > IC > Pearson.**

| Metric | Outlier robust | Scale invariant | Penalises inconsistency | Portfolio-theoretic basis |
|--------|---------------|-----------------|------------------------|--------------------------|
| Pearson | ✗ | ✗ | ✗ | Weak |
| IC | ✓ | ✓ | ✗ | Strong |
| **ICIR** | **✓** | **✓** | **✓** | **Strongest (Grinold-Kahn)** |

---

## 6. Output Files

| File | Contents |
|------|----------|
| `bayesian_optimisation_results.png` | Six-panel optimisation diagnostic |
| `metric_comparison.png` | Three-panel metric analysis |
| `optimisation_history.csv` | All trials: params + all three metrics |

### Figure Description

**Figure 1 — Bayesian Optimisation Results (6 panels)**

- **A — Convergence curve.** Best ICIR found so far vs trial index. Should show rapid improvement in early random trials then slower improvement as the GP refines. A plateau indicates convergence; a still-falling curve indicates more trials are needed.

- **B — Horizon × Medium Momentum surface.** Each point is one trial, coloured by ICIR. Green clusters reveal which regions of this 2D projection are most promising. Useful for identifying whether the optimal horizon is short (momentum decays quickly) or long.

- **C — Parameter sensitivity.** Absolute Spearman correlation between each parameter and ICIR across all trials. High sensitivity (|ρ| > 0.3) means the parameter is an important lever; low sensitivity means it can be fixed without much loss. Saves future computation.

- **D — ICIR distribution.** Histogram of all evaluated ICIR values. Shows the typical achievable range and how often random configurations produce positive ICIR.

- **E — Daily IC of best config.** Rolling 63-day IC. Allows visual identification of regimes where the optimal configuration works vs fails — informative for thesis narrative.

- **F — IC autocorrelogram.** Tests for overlap-induced autocorrelation at lag $h$. The bar at lag $h$ should be significantly positive (grey confidence bands) if the forward return windows overlap. This validates the Newey-West bandwidth choice.

**Figure 2 — Metric Comparison (3 panels)**

- **A — Scatter plot.** Predicted vs realised returns. A cigar-shaped cloud tilted toward the 45° line is a good sign. Outliers visible here are the heavy-tail events that Pearson would be dominated by.

- **B — Rolling metrics over time.** Normalised rolling IC and rolling Pearson plotted together. When they diverge, outlier events are driving the difference — confirming IC's advantage.

- **C — Stability heatmap.** IC, ICIR, and Pearson broken down by time quartile. A good model should show positive IC and ICIR across all four quartiles. Large variation across quartiles signals regime-dependence.

---

## 7. Usage

```bash
# Install dependencies
pip install lightgbm scikit-optimize pandas numpy scipy matplotlib seaborn yfinance

# Run with defaults (40 trials, ICIR metric)
python hparam_optimization.py

# The script will:
# 1. Download 10 S&P 500 constituents from Yahoo Finance (2010–2024)
# 2. Run 40 Bayesian Optimisation trials (~2–4 hours depending on hardware)
# 3. Print best configuration and metric comparison table
# 4. Save two figures and a CSV history file
```

### Adjusting the Number of Trials

The trade-off between computation time and solution quality:

| `n_calls` | Approx runtime | Expected quality |
|-----------|---------------|-----------------|
| 20 | ~1 hour | Good initial estimate |
| 40 | ~2–3 hours | Recommended (default) |
| 80 | ~5–6 hours | Near-optimal |
| 120 | ~8–9 hours | Diminishing returns |

### Extending the Search Space

To add LightGBM parameters to the search (not recommended until feature parameters are fixed):

```python
# Add to SEARCH_SPACE
Integer(100, 500, name="n_estimators"),
Integer(3,    6,  name="max_depth"),
Integer(8,   31,  name="num_leaves"),
```

And update the `objective` function signature accordingly. Be aware that this doubles the number of trials needed to achieve the same coverage.
