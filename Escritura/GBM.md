## How the Price is Modeled Under Geometric Brownian Motion (GBM)

In continuous quantitative finance, the price of an asset or a trading portfolio is frequently modeled using a stochastic process known as **Geometric Brownian Motion (GBM)**.

The continuous-time change in the stock price ($S_t$) is mathematically represented by the following Stochastic Differential Equation (SDE):

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

Where:

* **$\mu$** represents the constant drift parameter of the asset.


* **$\sigma$** represents the constant diffusion or volatility parameter.


* **$dW_t$** represents a standard Wiener process, which introduces random, independent market innovations over time.



By applying **Ito's Lemma** to solve this stochastic differential equation, the actual path of the stock price over a time horizon $T$ is integrated as:

$$S_T = S_0 \cdot e^{(\mu - \frac{\sigma^2}{2})T}$$

This equation demonstrates that the long-term compounded growth rate of the portfolio is governed by the exponent $\mu - \frac{\sigma^2}{2}$. Furthermore, under this formulation, the relative change of the asset ($\frac{dS_t}{S_t}$) behaves exactly like a standard arithmetic Brownian motion.

---

## Why the Price is Modeled This Way

Financial modelers choose Geometric Brownian Motion over alternative stochastic frameworks (such as standard arithmetic Brownian motion) due to several critical properties:

* **Prevents Unrealistic Negative Prices:** In a standard arithmetic Brownian motion, the random variable can fluctuate into negative territory. Because asset prices are essentially positive, simulating a random price innovation that results in a negative asset value does not make financial sense.


* **The Log-Price Advantage:** GBM resolves the negativity issue by defining the asset price as an exponential function: $Y(t) = \exp[y(t)]$, where $y(t)$ is an arithmetic Brownian motion. While price itself cannot drop below zero, the logarithm of a price ($p = \log(P)$) can perfectly accept negative values. This mathematical setup ensures that simulated asset prices remain strictly positive.


* **Lognormal Distribution of Returns:** Because the underlying log-prices behave normally, the absolute asset price changes within a given time interval are accurately described using a lognormal distribution.



---

## Who Modeled the Price This Way?

The paradigm of modeling asset prices through mathematical randomness was established and refined by several historical and modern figures:

* **Louis Bachelier (1900):** The foundational roots of modeling market dynamics via random steps began with Bachelier's pioneering thesis in 1900, which first explored random price behavior.


* **Classical Option Pricing Creators (Black, Scholes, Merton):** The explicit application of Geometric Brownian Motion became the foundational cornerstone and a basic assumption of classical option pricing theory, widely known as Black-Scholes Theory (BST). Ito's Lemma applied to a GBM asset path serves as the mathematical pillar used to derive the classical Black-Scholes equation.


* **Modern Quantitative Authors:** GBM remains heavily utilized by modern continuous finance practitioners and researchers. It serves as a base assumption for portfolio optimization in major quantitative trading literature, including books by Ernie Chan (*Quantitative Trading*) and the research works of Marcos López de Prado.