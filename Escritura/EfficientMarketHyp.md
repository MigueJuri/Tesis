The **Efficient Market Hypothesis (EMH)** is a foundational framework in financial economics and quantitative finance. Pioneered formally by Nobel laureate Eugene Fama in his seminal 1970 paper, the core premise of EMH is simple: **asset prices fully and instantaneously reflect all available information.**

Because prices always incorporate current knowledge, any new price movement must be driven solely by the arrival of *new* information. By definition, new information (news) is unexpected and unpredictable. Therefore, under a perfectly efficient market, price changes must follow a **random walk**, making it impossible for an investor to consistently achieve "alpha" (returns above the market benchmark) on a risk-adjusted basis.

---

### The Three Forms of Market Efficiency

Market efficiency is not an all-or-nothing proposition. Fama structured the hypothesis into three distinct levels based on what constitutes the definition of "available information." These levels are typically visualized as nested subsets of information:

#### 1. Weak-Form Efficiency

* **The Information Set:** All historical trading information, including past prices, price trends, and trading volumes.
* **The Implication:** Past price patterns cannot be used to predict future prices. Therefore, **technical analysis** (charting, moving averages, momentum indicators) cannot consistently generate superior risk-adjusted returns.
* **Mathematical view:** If a market is weak-form efficient, the current price is a sufficient statistic for predicting tomorrow's price, and price changes are uncorrelated over time: $\text{Cov}(\Delta S_t, \Delta S_{t-k}) = 0$.

#### 2. Semi-Strong-Form Efficiency

* **The Information Set:** All *publicly available* information. This includes not only past trading data but also company financial statements, earnings announcements, macroeconomic data, public management guidance, and geopolitical news.
* **The Implication:** Prices adjust so rapidly to public announcements that no one can profit by analyzing this data after it is released. Consequently, **fundamental analysis** (evaluating balance sheets, P/E ratios, discounted cash flows) is rendered incapable of identifying undervalued or overvalued stocks consistently.

#### 3. Strong-Form Efficiency

* **The Information Set:** *All* information, both public and private (corporate insider information).
* **The Implication:** The market is so highly attuned that even corporate insiders possessing non-public data (e.g., a CEO knowing about an unannounced merger) cannot achieve an anomalous trading edge because that value is already implicitly factored into the asset's equilibrium price dynamics. In reality, most economists agree markets are *not* strong-form efficient, which is why insider trading laws exist.

---

### The Mathematical Backbone: Martingales and Random Walks

In mathematical finance, EMH translates directly into the concept of a **martingale process**. If a market is efficient with respect to an information filtration ($\mathcal{F}_t$) up to time $t$, the expected value of tomorrow’s price, given everything we know today, is exactly today’s price (adjusted for the risk-free rate or risk premium):

$$E[S_{t+1} \mid \mathcal{F}_t] = S_t$$

Because the expected price change is zero ($E[S_{t+1} - S_t \mid \mathcal{F}_t] = 0$), the actual price change $\epsilon_{t+1} = S_{t+1} - S_t$ behaves as an unpredictable white noise innovation. This is why models like **Geometric Brownian Motion (GBM)** use a random Wiener process ($dW_t$) to dictate continuous price adjustments—it models the random, unforecastable stream of news hits.

---

### The Arbitrage Engine: Why Markets Become Efficient

The reason markets approach efficiency is driven by the rational self-interest of competitive market participants:

1. **Information Processing:** Thousands of hedge funds, quantitative trading algorithms, and institutional analysts constantly scan the globe for mispriced assets.
2. **Instant Arbitrage:** If a company's stock is fundamentally worth $\$100$ but trades at $\$98$ due to a temporary imbalance, traders immediately buy the stock to capture the $\$2$ edge. This collective buying pressure instantly drives the price up to $\$100$.
3. **Fleeting Opportunities:** The paradox of EMH (known as the *Grossman-Stiglitz Paradox*) states that if a market were perfectly efficient, no one would spend money researching information, which would then make the market inefficient. Therefore, markets exist in an **equilibrium of near-efficiency**, where anomalies exist but are fleeting, microscopic, and expensive to capture.

---

### The Modern Quantitative & Machine Learning View

While traditional finance textbooks treat EMH as a rigid rule, modern continuous-time quantitative finance and algorithmic asset management (such as the frameworks explored by figures like Marcos López de Prado and Ernie Chan) look at EMH through a more nuanced lens:

* **Risk Premiums vs. Inefficiencies:** Quantitative models often do not look for "market mistakes" (pure inefficiencies). Instead, they look for complex, non-linear mathematical risk factors. If an algorithm generates a steady return, it is often compensation for absorbing a specific, structural market risk that other participants want to hedge away.
* **Statistical Inefficiencies:** High-frequency trading (HFT) and alternative data (such as order book imbalances or satellite imagery) exploit microscopic structural lag times where information takes a few milliseconds or hours to propagate through the market microstructure. Quants look at efficiency as a *spectrum* dictated by transaction costs and latency rather than a absolute state.