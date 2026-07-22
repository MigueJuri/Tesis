We must immediately discard the layman’s notion of pairs trading as merely buying a "loser" and shorting a "winner" based on superficial price correlations. Linear correlation (Pearson's $\rho$) measures dependence between stationary variables, yet financial price series are notoriously non-stationary. If you rely on correlation, you will be swiftly punished by spurious regressions and structural breaks.

Instead, we must treat the market as a complex adaptive system and financial time series as physical observables. We construct a synthetic, mean-reverting portfolio (the spread) whose dynamics can be explicitly modeled using the mathematics of non-equilibrium statistical mechanics.

Here is the rigorous formulation and step-by-step implementation for a practitioner constrained to daily or hourly data.

---

### 1. Physics-Based Formalism: The Ornstein-Uhlenbeck Process and Fokker-Planck Equation

We treat the log-prices of two assets, $X_t = \ln(P^{(1)}_t)$ and $Y_t = \ln(P^{(2)}_t)$, as random walks. Our goal is to find a cointegrating vector $[1, -\beta]$ such that the linear combination (the spread) $Z_t$ is stationary ($I(0)$):


$$Z_t = Y_t - \beta X_t$$

We model the dynamics of this synthetic observable $Z_t$ as an Ornstein-Uhlenbeck (OU) process, which is the continuous-time analogue of the discrete autoregressive AR(1) model. It represents a particle undergoing Brownian motion under the influence of a harmonic potential well (a spring restoring it to equilibrium).

$$dZ_t = \theta (\mu - Z_t)dt + \sigma dW_t$$

Where:

* $\theta > 0$ is the rate of mean reversion (the stiffness of the spring).
* $\mu$ is the long-term equilibrium mean.
* $\sigma$ is the volatility of the stochastic shocks.
* $dW_t$ is a standard Wiener process.

To understand the macroscopic behavior of this system, we do not look at single paths. We look at the evolution of its probability density function $p(z,t)$. In physics, the time evolution of the probability density of an OU process is governed by the **Fokker-Planck Equation**:

$$\frac{\partial p(z,t)}{\partial t} = \frac{\partial}{\partial z} \Big[ \theta(z - \mu) p(z,t) \Big] + \frac{1}{2}\sigma^2 \frac{\partial^2 p(z,t)}{\partial z^2}$$

Because the system is mean-reverting, it eventually reaches a thermodynamic equilibrium where $\frac{\partial p}{\partial t} = 0$. Solving the stationary Fokker-Planck equation yields the invariant distribution:

$$p_{st}(z) = \sqrt{\frac{\theta}{\pi \sigma^2}} \exp\left( - \frac{\theta}{\sigma^2} (z-\mu)^2 \right)$$

This proves that the equilibrium distribution of a cointegrated spread is a Gaussian with mean $\mu$ and variance $\frac{\sigma^2}{2\theta}$. This exact mathematical constraint is what justifies the use of a "Z-score" to define entry and exit thresholds.

---

### 2. Step-by-Step Implementation

**Step 1: Universe Selection via Information Theory**
Do not use linear correlation to find pairs. Instead, use **Mutual Information (MI)** to detect both linear and non-linear dependence between the fractional differentials of the price series.


$$I(X;Y) = \sum_{y \in Y} \sum_{x \in X} p(x,y) \log \left( \frac{p(x,y)}{p(x)p(y)} \right)$$


Filter your universe by selecting pairs with the highest mutual information.

**Step 2: Cointegration and Fractional Differentiation**
Verify stationarity. Instead of just relying on the Engle-Granger or Johansen tests—which force you into a binary $I(1)$ or $I(0)$ decision—apply **Fractional Differentiation** (as detailed by Marcos López de Prado). You differentiate the series just enough ($0 < d < 1$) to achieve stationarity (p-value of ADF test < 0.05) while retaining the maximum amount of theoretical memory.

**Step 3: Estimate OU Parameters (Maximum Likelihood)**
Once you have your spread $Z_t$, estimate the OU parameters ($\theta, \mu, \sigma$) using Maximum Likelihood Estimation (MLE) or an OLS regression on the discrete time equation:


$$Z_{t} - Z_{t-1} = a + b Z_{t-1} + \epsilon_t$$


Where $\theta = -\ln(1+b)/\Delta t$, $\mu = -a/b$, and $\sigma = \text{std}(\epsilon) / \sqrt{\Delta t}$.

Crucially, calculate the **half-life** of mean reversion:


$$t_{1/2} = \frac{\ln(2)}{\theta}$$


If $t_{1/2}$ is shorter than your data resolution (e.g., less than an hour), it is HFT noise and you cannot trade it due to latency. If $t_{1/2}$ is 60 days, your capital will be tied up for too long. Target pairs with a half-life between 1 to 10 days.

**Step 4: Execution Mechanics**
Define a dimensionless Z-score:


$$z\text{-score}_t = \frac{Z_t - \mu}{\sqrt{\sigma^2 / 2\theta}}$$

* Enter Long (Buy $Y$, Short $X$): $z\text{-score} < -2.0$
* Enter Short (Short $Y$, Buy $X$): $z\text{-score} > 2.0$
* Exit (Close both positions): $z\text{-score}$ crosses $0$.

---

### 3. The 'Alpha' Hypothesis

**We extract value here because of economic gravitation and structural latency in market microstructure.** Two assets tightly bound by sector or supply-chain fundamentals (e.g., KO and PEP) will drift apart due to idiosyncratic noise or unsynchronized institutional order flow. By treating this divergence as a perturbed physical system bound by a harmonic oscillator (the OU process), we capture the premium generated when the system inevitably relaxes back to its thermodynamic equilibrium. We are harvesting the noise.

---

### 4. 'Red Team' Critique (The Pre-Mortem)

If you deploy this tomorrow, you will likely lose money. Here is why:

* **The Stationarity Mirage (Structural Breaks):** Cointegration is a backward-looking metric. A structural break (e.g., one company announces an acquisition, or regulatory change) fundamentally destroys the harmonic potential well. The spring breaks, the OU process turns into a Brownian motion with drift, and your $z\text{-score}$ diverges to infinity. Your model will average down into ruin.
* **Transaction Cost Friction:** You are paying the bid-ask spread on *four* separate legs (enter $Y$, enter $X$, exit $Y$, exit $X$). If your half-life is too short, the expected value of the spread convergence is completely eaten by the bid-ask bounce.
* **Cost of Borrow and Short Locates:** You must short one of the assets. In real capital deployment, hard-to-borrow (HTB) stocks carry annualized borrow fees that accumulate daily. Your strategy assumes $\mu$ is constant, but borrow costs create a constant negative drift on your expected value.
* **Look-Ahead Bias in Hedge Ratio ($\beta$):** If you compute $\beta$ over your entire dataset and then generate $Z_t$, you have leaked future information into the past. $\beta$ must be calculated dynamically on a rolling window, which means $\mu$ and $\theta$ are also non-stationary.

---

**Language Correction:**
Your prompt stated: "Explain to me, step by step, how to do **pair** trading."
The industry standard terminology is "**pairs** trading" (plural), as the strategy involves the simultaneous trading of a pair of assets, and scales across multiple pairs.