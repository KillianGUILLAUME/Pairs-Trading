# Pairs Trading — Quantitative Research Framework

A systematic pairs trading engine on cryptocurrency markets, built around statistical arbitrage theory, Kalman filtering, and machine learning meta-labeling. The long-term research direction extends toward generative stress-testing via Causal Optimal Transport and path signature methods.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Mathematical Foundations](#2-mathematical-foundations)
   - [2.1 Pair Formation](#21-pair-formation)
   - [2.2 Spread Construction and the Hedge Ratio](#22-spread-construction-and-the-hedge-ratio)
   - [2.3 Mean-Reversion Dynamics: Ornstein-Uhlenbeck Process](#23-mean-reversion-dynamics-ornstein-uhlenbeck-process)
   - [2.4 Cointegration Tests](#24-cointegration-tests)
   - [2.5 Hurst Exponent](#25-hurst-exponent)
   - [2.6 Kalman Filter: Dynamic Hedge Ratio](#26-kalman-filter-dynamic-hedge-ratio)
   - [2.7 Trading Signal: Z-Score](#27-trading-signal-z-score)
   - [2.8 Trade Entry and Exit Rules](#28-trade-entry-and-exit-rules)
   - [2.9 Position Sizing: Fractional Kelly Criterion](#29-position-sizing-fractional-kelly-criterion)
3. [System Architecture](#3-system-architecture)
4. [Machine Learning Layer (Meta-Labeling)](#4-machine-learning-layer-meta-labeling)
   - [4.1 Triple Barrier Labeling](#41-triple-barrier-labeling)
   - [4.2 Feature Engineering](#42-feature-engineering)
   - [4.3 Model and Validation](#43-model-and-validation)
5. [Research Direction: Causal Generative Modeling](#5-research-direction-causal-generative-modeling)
   - [5.1 Motivation](#51-motivation)
   - [5.2 Causal Optimal Transport and Path Signatures](#52-causal-optimal-transport-and-path-signatures)
   - [5.3 Autoregressive Causal Architecture](#53-autoregressive-causal-architecture)
   - [5.4 Martingale Regularization](#54-martingale-regularization)
6. [Repository Structure](#6-repository-structure)
7. [Configuration](#7-configuration)
8. [Roadmap](#8-roadmap)

---

## 1. Overview

This project implements a full quantitative pipeline for statistical arbitrage on cryptocurrency spot markets (Binance). The universe consists of the top 50 most liquid USDT pairs, filtered to approximately 23 assets. Candles are at the 1-hour timeframe, with a training window from January 2022 to January 2024.

The strategy exploits mean-reverting spreads between pairs of cointegrated assets. The signal generation is based on a Kalman filter with a dynamic hedge ratio, and a dual z-score confirmation rule. A machine learning meta-labeling layer is built on top to filter low-quality entries. A generative stress-testing module based on Conditional Flow Matching and Diffusion Schrödinger Bridges is under development.

---

## 2. Mathematical Foundations

### 2.1 Pair Formation

The first step reduces the combinatorial space $\binom{N}{2}$ of candidate pairs from the universe of $N$ assets. Two assets are considered as a candidate pair if:

1. They belong to the same cluster in a hierarchical clustering of log-returns.
2. Their log-return correlation satisfies $|\rho_{AB}| \geq \rho_{\min}$.

The distance metric used for clustering is derived from the correlation matrix:

$$d(A, B) = \sqrt{\frac{1 - \rho_{AB}}{2}}$$

This metric satisfies the triangle inequality and maps correlation directly to Euclidean-like distances in $[0, 1]$. Ward linkage is used to build the dendrogram, and clusters are cut at a fixed number $K = 5$.

Only pairs within the same cluster and above the correlation threshold are forwarded to cointegration testing.

### 2.2 Spread Construction and the Hedge Ratio

Given two log-price series $\log P_A(t)$ and $\log P_B(t)$, the spread is:

$$S_t = \log P_A(t) - \beta_t \log P_B(t) - \alpha_t$$

where $\beta_t$ is the hedge ratio (initially estimated via OLS, then tracked dynamically by a Kalman filter) and $\alpha_t$ is the intercept. By construction, if $A$ and $B$ are cointegrated, $S_t$ is stationary.

The initial static estimate is obtained by OLS:

$$\hat{\beta} = \frac{\sum_{t} \log P_B(t) \cdot \log P_A(t)}{\sum_{t} (\log P_B(t))^2}$$

### 2.3 Mean-Reversion Dynamics: Ornstein-Uhlenbeck Process

A cointegrated spread is modeled as an Ornstein-Uhlenbeck (OU) process:

$$dS_t = \kappa (\mu - S_t)\, dt + \sigma\, dW_t$$

where $\kappa > 0$ is the mean-reversion speed, $\mu$ is the long-run mean, $\sigma$ is the diffusion coefficient, and $W_t$ is a standard Brownian motion. The key implication is that the spread reverts to $\mu$ with a characteristic time scale called the **half-life**:

$$\tau_{1/2} = \frac{\ln 2}{\kappa}$$

The half-life is estimated by fitting a discrete-time AR(1) model on $\Delta S_t$:

$$\Delta S_t = \beta_0 + \beta_1 S_{t-1} + \varepsilon_t$$

Then:

$$\hat{\kappa} = -\beta_1, \quad \tau_{1/2} = \frac{-\ln 2}{\beta_1}$$

A valid pair must satisfy $1 \leq \tau_{1/2} \leq 500$ (in hours). Pairs with infinite half-life (i.e., $\beta_1 \geq 0$) are rejected as they exhibit no mean-reversion.

### 2.4 Cointegration Tests

Two complementary tests are run on every candidate pair.

**Engle-Granger (1987):** The test regresses $\log P_A$ on $\log P_B$ and runs an Augmented Dickey-Fuller (ADF) test on the residuals. The null hypothesis is the absence of cointegration. A pair is accepted at significance level $\alpha = 0.05$ if:

$$p\text{-value} < 0.05$$

**Johansen (1988):** A Vector Error Correction Model (VECM) is estimated. The trace statistic tests the null hypothesis that there are at most $r$ cointegrating vectors. For a bivariate system, we test $H_0: r = 0$ using:

$$\Lambda_{\text{trace}}(r) = -T \sum_{i=r+1}^{p} \ln(1 - \hat{\lambda}_i)$$

Cointegration is validated when $\Lambda_{\text{trace}} > \text{CV}_{95\%}$, where $\text{CV}_{95\%}$ is the Johansen critical value at the 5% level.

A pair passes the formation filter if it is accepted by at least one of the two tests.

### 2.5 Hurst Exponent

The Hurst exponent $H$ characterizes the memory properties of a time series and is estimated via the variogram method:

$$\tau(\ell) = \sqrt{\text{Var}(S_{t+\ell} - S_t)}$$

A log-log regression gives:

$$\ln \tau(\ell) \approx H \cdot \ln \ell + C$$

The interpretation is:

- $H < 0.5$: the series is mean-reverting (anti-persistent). This is the regime we target.
- $H = 0.5$: random walk, no exploitable structure.
- $H > 0.5$: trending, persistent dynamics.

A pair is valid only if $H < 0.5$ at formation time.

### 2.6 Kalman Filter: Dynamic Hedge Ratio

The hedge ratio is non-stationary in practice: structural breaks, liquidity shifts, and funding rate changes cause $\beta$ to drift over time. A Kalman filter tracks $\theta_t = [\beta_t, \alpha_t]^\top$ as a latent state.

The state-space model is:

$$\theta_t = \theta_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q)$$
$$\log P_B(t) = F_t^\top \theta_t + v_t, \quad v_t \sim \mathcal{N}(0, R)$$

where $F_t = [\log P_A(t),\ 1]^\top$ and:

$$Q = \begin{pmatrix} \delta_\beta & 0 \\ 0 & \delta_\alpha \end{pmatrix}$$

The prediction and update steps are:

**Prediction:**
$$\hat{\theta}_{t|t-1} = \hat{\theta}_{t-1|t-1}$$
$$P_{t|t-1} = P_{t-1|t-1} + Q$$

**Innovation and Kalman Gain:**
$$e_t = \log P_B(t) - F_t^\top \hat{\theta}_{t|t-1}$$
$$S_t = F_t^\top P_{t|t-1} F_t + R$$
$$K_t = \frac{P_{t|t-1} F_t}{S_t}$$

**Update:**
$$\hat{\theta}_{t|t} = \hat{\theta}_{t|t-1} + K_t e_t$$
$$P_{t|t} = (I - K_t F_t^\top) P_{t|t-1}$$

The spread at time $t$ is the Kalman innovation $e_t = \log P_B(t) - \hat{\beta}_t \log P_A(t) - \hat{\alpha}_t$, which is by construction centered near zero.

The hyperparameters $\delta_\beta$, $\delta_\alpha$, and $R$ are calibrated automatically from the residual variance of an initial OLS fit:

$$R_{\text{opt}} = 3 \cdot \text{Var}(\Delta \hat{\varepsilon}), \quad \delta_\beta = 10^{-3} R_{\text{opt}}, \quad \delta_\alpha = 10^{-2} R_{\text{opt}}$$

### 2.7 Trading Signal: Z-Score

Two complementary z-scores are computed on the spread series $e_t$.

**Rolling z-score** over a window $W$ (default: 168 bars = 1 week):

$$z_t^{\text{roll}} = \frac{e_t - \mu_W(e)}{\sigma_W(e)}$$

**Kalman z-score** (innovation normalized by its own variance):

$$z_t^{\text{Kalman}} = \frac{e_t}{\sqrt{S_t}}$$

The Kalman z-score is model-consistent: its theoretical distribution under the null (no mispricing) is $\mathcal{N}(0, 1)$. It adapts instantly to changes in spread volatility, making it robust to regime shifts.

### 2.8 Trade Entry and Exit Rules

A trade is triggered only when both z-scores confirm the signal simultaneously, and the spread is not in a high-volatility regime.

Let $\theta_{\text{entry}} = 2.0$ and $\theta_{\text{exit}} = 0.5$ be the entry and exit thresholds, and $z_{\max} = 4.0$ a cap to avoid entering during flash crashes.

**Long spread** (buy $A$, sell $B$):

$$z_t^{\text{roll}} < -\theta_{\text{entry}} \quad \text{and} \quad z_t^{\text{Kalman}} < -\theta_{\text{entry}} \quad \text{and} \quad |z_t^{\text{roll}}| \leq z_{\max}$$

**Short spread** (sell $A$, buy $B$):

$$z_t^{\text{roll}} > \theta_{\text{entry}} \quad \text{and} \quad z_t^{\text{Kalman}} > \theta_{\text{entry}} \quad \text{and} \quad |z_t^{\text{roll}}| \leq z_{\max}$$

**Exit** (mean-reversion achieved or spread diverged):

$$|z_t^{\text{roll}}| < \theta_{\text{exit}} \quad \text{or} \quad |z_t^{\text{Kalman}}| < \theta_{\text{exit}} \quad \text{or} \quad |z_t^{\text{roll}}| > z_{\max}$$

A maximum holding period is also enforced, set dynamically to $2 \times \tau_{1/2}$ (capped at a minimum of 20 bars), after which the position is forcibly closed.

**Volatility regime filter:** Trading is suspended when the current rolling spread volatility exceeds twice the long-term median volatility:

$$\hat{\sigma}_W(e) > 2 \cdot \text{median}_{4W}\!\left(\hat{\sigma}_W(e)\right)$$

### 2.9 Position Sizing: Fractional Kelly Criterion

The Kelly criterion gives the theoretically optimal fraction of capital to allocate to maximize the long-run growth rate of wealth. Given a win probability $p$ and a win/loss ratio $b = \bar{r}_{\text{win}} / \bar{r}_{\text{loss}}$:

$$f^* = \frac{p \cdot b - (1 - p)}{b}$$

In practice, full Kelly is too aggressive. The fractional Kelly fraction used here is:

$$f = \frac{f^*}{4}$$

capped at a maximum leverage of 2. The win probability $p$ is the output of the ML meta-labeling model described in the next section.

---

## 3. System Architecture

```
config/
  config.yaml          — Data source, universe, timeframes
  universe.json        — Tradeable instruments

research/
  pairs/
    formation.py       — Clustering, cointegration tests, pair scoring
  signals/
    signal_generator.py — Kalman filter, z-scores, path signature features
  labels/
    triple_barrier.py  — Triple barrier labeling (ML target)
  models/
    ml_model.py        — Meta-labeling classifier (XGBoost / Random Forest)
  sizing/
    kelly_fractionnaire.py — Fractional Kelly position sizing
  backtest/
    engine.py          — Vectorized P&L, performance metrics
    walk_forward.py    — Walk-forward validation
    simulation.py      — Monte Carlo simulation
    stress_test.py     — Stress testing engine

features/
  features_eng.py      — Feature extraction at entry points

neural_nets/
  signature_vf.py      — Causal velocity field conditioned by path signatures
  losses.py            — CFM and Bridge Matching losses
  data_loader.py       — Data pipeline for neural net training
  interpolants.py      — Stochastic interpolants
```

---

## 4. Machine Learning Layer (Meta-Labeling)

The Kalman-based signal identifies candidate entry points. The ML layer acts as a secondary filter: it predicts the probability that a given entry will be profitable, and only confirms trades above a confidence threshold.

### 4.1 Triple Barrier Labeling

Each entry point $t_0$ with direction $d \in \{+1, -1\}$ is labeled using three barriers defined on the z-score path:

- **Take-profit barrier:** $|z_t| \leq z_{\text{upper}} = 0.5$ (mean-reversion achieved).
- **Stop-loss barrier:** $|z_t| \geq z_{\text{lower}} = 4.0$ (spread diverged).
- **Time stop:** $t - t_0 > 2 \cdot \tau_{1/2}$ (trade expired).

The label is $+1$ if the take-profit barrier is hit first, $-1$ if the stop-loss is hit first, and $0$ for a time stop. For the binary classification model, labels $\{-1, 0\}$ are merged into $0$ (non-profitable), and the target is:

$$y = \mathbf{1}[\text{take-profit hit first}]$$

### 4.2 Feature Engineering

Features are computed at each entry point $t_0$, using only information strictly prior to $t_0$ to prevent any data leakage.

| Feature | Description |
|---|---|
| Hurst exponent | Computed on $S_{t_0 - 300 : t_0}$. Quantifies mean-reversion strength. |
| ADF p-value | Stationarity test on $S_{t_0 - 300 : t_0}$ with automatic lag selection (AIC). |
| RSI differential | $\text{RSI}_{14}(A) - \text{RSI}_{14}(B)$. Captures relative momentum imbalance. |
| Rolling volatility 24h | $\hat{\sigma}_{24}(S)$. Short-term spread dispersion. |
| Rolling volatility 72h | $\hat{\sigma}_{72}(S)$. Medium-term spread dispersion. |
| Volatility ratio | $\hat{\sigma}_{24} / \hat{\sigma}_{72}$. Elevated ratio signals a volatility spike at entry. |
| Beta momentum | $\hat{\beta}_{t_0} - \hat{\beta}_{t_0 - 24}$. Rate of change of the Kalman hedge ratio. |
| Z-score at entry | $z_{t_0}^{\text{roll}}$. Did we enter aggressively or conservatively? |

### 4.3 Model and Validation

The classifier is XGBoost or Random Forest, optimized on **precision** rather than accuracy, to minimize false positives (toxic entries) at the cost of missing some good trades.

Since the dataset is a single time series, standard cross-validation is inapplicable due to temporal autocorrelation and label overlap across trades. Validation uses a **Purged Walk-Forward scheme** (after López de Prado, 2018):

- The training set only contains bars that do not overlap in time with any trade in the test set.
- An embargo period is enforced between the last training bar and the first test bar.

---

## 5. Research Direction: Causal Generative Modeling

### 5.1 Motivation

Classical backtesting is limited to a single historical path. The distribution of realized paths is highly non-representative of the true data-generating process, especially in the tails. Stress-testing the strategy on thousands of realistic synthetic scenarios would more faithfully estimate out-of-sample risk.

The challenge is that standard generative models (GANs, normalizing flows) treat time series as static objects and ignore the arrow of time and the causal structure of information. A model that generates the future using future information would be financially meaningless.

### 5.2 Causal Optimal Transport and Path Signatures

The framework is grounded in Causal Optimal Transport (Backhoff-Veraguas et al., 2020). The adapted Wasserstein distance between two stochastic processes $\mu$ and $\nu$ is:

$$\mathcal{AW}_2(\mu, \nu) = \inf_{\pi \in \text{ATC}(\mu, \nu)} \int \|x - y\|^2\, d\pi(x, y)$$

where $\text{ATC}(\mu, \nu)$ is the set of adapted (causal) transport plans — couplings $\pi$ where the distribution of $y_t$ given $x_{0:t}$ depends only on the filtration generated by $x_{0:t}$, not on future realizations.

To condition a neural network on the history $X_{t-w:t}$ without passing the full trajectory, we use the **path signature** (Chen, 1957):

$$\text{Sig}_{s,t}(X) = \left(1, \int_s^t dX^i_u, \int_s^t \int_s^u dX^i_r\, dX^j_u, \ldots \right)_{i,j,\ldots}$$

The truncated signature of order $d$ for a $k$-dimensional path has dimension $\sum_{n=1}^{d} k^n$. It satisfies the **universality theorem**: the linear span of signature features is dense in the space of continuous functions on path space. In other words, the path signature is sufficient to approximate any causal functional of the trajectory.

In our implementation, the path is augmented with a time channel $\tilde{X}_t = (t, S_t, z_t)$ (3-dimensional), giving a truncated signature of dimension $\sum_{n=1}^{3} 3^n = 39$ at depth 3.

### 5.3 Autoregressive Causal Architecture

A naive application of Conditional Flow Matching to trajectory generation would train a velocity field with the target $v^* = X_1 - X_0$, where $X_1$ is the endpoint of the generated path. This target is **anticipative**: the learned drift $b_\theta$ implicitly encodes information about where the trajectory ends, violating the causal filtration. The consequence for stress-testing is that the generated scenarios would exhibit artificially strong mean-reversion (the drift "aims" at $X_1$), under-represent tail events, and produce optimistic risk estimates.

The solution is an **autoregressive formulation** that separates two distinct time variables:

- **Physical time $t$**: the index along the time series ($t = 1, 2, \ldots, T$).
- **Diffusion time $\tau \in [0, 1]$**: the parameter of the transport from noise to the next increment.

Let $r_t = \log P_{t+1} - \log P_t = \log(P_{t+1}/P_t)$ denote the log-return of an asset at time $t$. The data pipeline (see `data_loader.py`) computes standardized bivariate log-returns $\tilde{r}_t \in \mathbb{R}^2$ for the two assets of a pair. The neural network operates entirely in this log-return space.

At each physical time step $t$, the model solves a **local** transport problem: map a noise sample $\Delta_0 \sim \mathcal{N}(0, I)$ to the next standardized log-return $\Delta_1 = \tilde{r}_t$, conditioned on the signature of the past window $[\tilde{r}_{t-w}, \ldots, \tilde{r}_{t-1}]$:

$$v_\theta\!\left(\tau,\, \Delta_\tau,\, \text{Sig}_{t-w,t}\right) \approx \Delta_1 - \Delta_0$$

where $\Delta_\tau = (1 - \tau)\Delta_0 + \tau \Delta_1$ is the linear interpolant along the diffusion axis.

The **Conditional Flow Matching loss** is:

$$\mathcal{L}(\theta) = \mathbb{E}_{t,\, \tau,\, \Delta_0,\, \Delta_1}\!\left[\left\|v_\theta(\tau, \Delta_\tau, \text{Sig}_{t-w,t}) - (\Delta_1 - \Delta_0)\right\|^2\right]$$

with $\tau \sim \mathcal{U}[0,1]$ and $\Delta_0 \sim \mathcal{N}(0, I)$. Training is simulation-free: no ODE/SDE integration is required.

The **Bridge Matching variant** (DSBM) samples $\Delta_\tau$ from a Brownian bridge between $\Delta_0$ and $\Delta_1$:

$$\Delta_\tau = (1-\tau)\Delta_0 + \tau \Delta_1 + \sigma\sqrt{\tau(1-\tau)}\, Z, \quad Z \sim \mathcal{N}(0, I)$$

with the analytic drift target $u^*(\tau, \Delta_\tau | \Delta_1) = (\Delta_1 - \Delta_\tau)/(1-\tau)$. The parameter $\sigma > 0$ regularizes training and captures the intrinsic uncertainty of the market.

**Inference (trajectory generation):** At each physical time step $t$, the generation procedure is:

1. Compute $\text{Sig}_{t-w,t}(\tilde{X})$ from the current window (only past data).
2. Sample $\Delta_0 \sim \mathcal{N}(0, I)$.
3. Integrate the ODE $\frac{d\Delta_\tau}{d\tau} = v_\theta(\tau, \Delta_\tau, \text{Sig}_{t-w,t})$ from $\tau = 0$ to $\tau = 1$ (Euler, $N=50$ steps).
4. Set $\hat{X}_{t+1} = X_t + \hat{\Delta}_1$.
5. Advance the sliding window by one step and repeat.

This procedure is **intrinsically causal**: at every physical step, the velocity field has access only to $\mathcal{F}_t$ (via the signature), and the generated increment $\hat{\Delta}_1$ is stochastic (through $\Delta_0$). No future information leaks into the dynamics.

The generated paths can then be used to stress-test the full Kalman + ML pipeline on thousands of realistic, causally consistent scenarios that extend into the tails of the distribution.

### 5.4 Martingale Regularization

The autoregressive formulation guarantees causality, but does not constrain the **conditional drift** of generated paths. Without regularization, the velocity field may introduce systematic biases, or collapse to a deterministic mapping (mode collapse), producing overconfident scenarios.

The regularization strategy operates at three levels:

**Training: Entropic regularization.** The `EntropicCFMLoss` augments the base CFM loss with a hinge term on the variance of the velocity field predictions across a batch:

$$\mathcal{L} = \mathcal{L}_{\text{CFM}} + \lambda_H \cdot \max\!\left(0,\, \varepsilon_{\min} - \text{Var}_{B}[v_\theta]\right)$$

The term is inactive when the prediction variance is above $\varepsilon_{\min}$ (healthy diversity). It activates only when the model begins to collapse, pushing the predictions apart. This is analogous to the entropic regularization in the Schrodinger Bridge literature, which selects the coupling with maximal entropy among all couplings matching the marginal constraints.

**Inference: RK4 solver.** The ODE integration from $\tau = 0$ to $\tau = 1$ uses a Runge-Kutta 4th order scheme:

$$k_1 = v_\theta(\Delta_{\tau_i}, \tau_i)$$
$$k_2 = v_\theta(\Delta_{\tau_i} + \tfrac{h}{2} k_1,\, \tau_i + \tfrac{h}{2})$$
$$k_3 = v_\theta(\Delta_{\tau_i} + \tfrac{h}{2} k_2,\, \tau_i + \tfrac{h}{2})$$
$$k_4 = v_\theta(\Delta_{\tau_i} + h\, k_3,\, \tau_i + h)$$
$$\Delta_{\tau_{i+1}} = \Delta_{\tau_i} + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

RK4 achieves $\mathcal{O}(h^4)$ local truncation error versus $\mathcal{O}(h)$ for Euler. This eliminates numerical discretization bias that would otherwise accumulate over long trajectories and corrupt the martingale property.

**Inference: PCFM martingale projection.** Following the Physics-Constrained Flow Matching paradigm, the martingale constraint is enforced **exactly at inference** by projecting onto the martingale manifold, without retraining:

1. For a given past context $\text{Sig}_{t-w,t}$, generate $K$ independent increments $\hat{\Delta}_1^{(k)}$ via the trained velocity field.
2. Estimate the conditional drift: $\hat{\mu} = \frac{1}{K} \sum_{k=1}^{K} \hat{\Delta}_1^{(k)}$.
3. Project: $\hat{\Delta}_1^{\text{proj}} = \hat{\Delta}_1 - \hat{\mu} + \mu_{\text{target}}$.

For a **pure martingale**, set $\mu_{\text{target}} = 0$. For a mean-reverting spread (OU process), $\mu_{\text{target}}$ can be set to the empirical conditional drift estimated from the training data.

The projection is an additive shift that preserves the variance and higher moments of the generated distribution while correcting the first moment to machine precision ($\sim 10^{-6}$). Unlike a training-time penalty, the constraint is enforced exactly and can be changed without retraining.

**Directions for further research:**

- **LightSBB-M** (Alouadi and Henry-Labordere, 2026): combines the Schrodinger Bridge (drift control) with Bass transport (volatility control) for joint calibration to market smiles.
- **Vectorial MOT**: multi-asset extension of the martingale constraint. For $N > 30$ assets, block-sparse optimization is required.

---

## 6. Repository Structure

```
pairs_trading/
├── config/
│   ├── config.yaml         # Data pipeline configuration
│   └── universe.json       # Tradeable asset universe
├── data/
│   └── storage/            # Parquet files (gitignored)
├── features/
│   └── features_eng.py     # ML feature extractor
├── neural_nets/
│   ├── signature_vf.py     # Causal velocity field (PyTorch)
│   ├── losses.py           # CFM and bridge matching losses
│   ├── data_loader.py      # Neural net data pipeline
│   └── interpolants.py     # Stochastic interpolants
├── notebooks/              # Jupyter exploration (gitignored)
├── research/
│   ├── backtest/
│   │   ├── engine.py       # Core P&L engine
│   │   ├── walk_forward.py # Walk-forward backtesting
│   │   ├── simulation.py   # Monte Carlo simulation
│   │   └── stress_test.py  # Stress testing
│   ├── labels/
│   │   └── triple_barrier.py
│   ├── models/
│   │   └── ml_model.py
│   ├── pairs/
│   │   └── formation.py
│   ├── signals/
│   │   └── signal_generator.py
│   └── sizing/
│       └── kelly_fractionnaire.py
└── requirements.txt
```

---

## 7. Configuration

The main configuration is in `config/config.yaml`:

```yaml
data:
  source: "binance"
  timeframes: ["1h", "4h", "1d"]
  history:
    start_date: "2022-01-01"
    end_date:   "2024-01-01"
```

The asset universe is defined in `config/universe.json`. It currently includes 23 USDT pairs with minimum 24-hour volume of 10M USDT: BTC, ETH, BNB, SOL, XRP, ADA, AVAX, DOGE, UNI, LINK, NEAR, SUI, LTC, ZEC, ENJ, FET, TRX, PEPE, DEGO, COS, PAXG, EUR, USDC.

Key backtest parameters:

| Parameter | Value | Description |
|---|---|---|
| `initial_capital` | 100,000 USDT | Starting portfolio value |
| `position_size` | 10% | Fraction of capital per leg |
| `fee_rate` | 4 bps | Binance maker/taker fee |
| `slippage_bps` | 1 bps | Market impact estimate |
| `entry_threshold` | 2.0 | Z-score entry level |
| `exit_threshold` | 0.5 | Z-score exit level |
| `stop_loss_z` | 4.0 | Z-score hard stop |

---

## 8. Roadmap

### Phase 1 — Statistical Arbitrage Core (complete)

- [x] Data pipeline (Binance OHLCV, Polars/Parquet)
- [x] Pair formation: hierarchical clustering, Engle-Granger, Johansen, half-life, Hurst filter
- [x] Signal generation: Kalman filter with auto-calibration, dual z-score, volatility regime filter
- [x] Vectorized backtest engine with realistic cost model (fees + slippage)
- [x] Performance metrics: Sharpe, Sortino, Calmar, max drawdown, profit factor
- [x] Walk-forward validation framework
- [x] Stress testing and Monte Carlo simulation modules

### Phase 2 — Machine Learning Meta-Labeling (in progress)

- [x] Triple barrier labeling on z-score paths
- [x] Feature engineering at entry points (Hurst, ADF p-value, RSI differential, volatility ratios, beta momentum)
- [x] Path signature features on sliding windows (iisignature, depth 3)
- [ ] XGBoost / Random Forest meta-labeling classifier
- [ ] Purged walk-forward cross-validation (López de Prado CPCV)
- [ ] Integration of Kelly sizing with ML confidence output
- [ ] End-to-end backtest: Kalman signal filtered by ML classifier

### Phase 3 -- Causal Generative Stress-Testing (research)

- [x] Causal Velocity Field conditioned by truncated path signature (signatory)
- [x] Autoregressive CFM loss and Bridge Matching loss (DSBM) with causal semantics
- [x] Autoregressive inference: `generate_step()` (RK4) and `generate_trajectory()` (multi-step sliding window)
- [x] Stochastic interpolants: linear, stochastic, and Brownian bridge
- [x] Entropic regularization (`EntropicCFMLoss`) to prevent mode collapse
- [x] RK4 solver (4th order Runge-Kutta) for discretization-free ODE integration
- [x] PCFM martingale projection (`martingale_project()`) at inference time
- [ ] Training pipeline: source measure (correlated OU process) to empirical data measure
- [ ] Synthetic scenario generation: thousands of realistic spread paths
- [ ] Stress-testing the full strategy (Phase 1 + Phase 2) on generated scenarios
- [ ] Out-of-sample risk estimation on tail distributions
- [ ] LightSBB-M integration: joint drift + volatility calibration (Alouadi and Henry-Labordere, 2026)
- [ ] Vectorial MOT extension for multi-asset pairs ($N > 30$)
- [ ] Research report: *Generative AI for Pairs Trading: Robust Backtesting via Causal Optimal Transport and Sig-Diffusions*

---

## References

- Engle, R.F. and Granger, C.W.J. (1987). Co-integration and error correction: representation, estimation, and testing. *Econometrica*, 55(2), 251–276.
- Johansen, S. (1988). Statistical analysis of cointegration vectors. *Journal of Economic Dynamics and Control*, 12(2–3), 231–254.
- Kalman, R.E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35–45.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Backhoff-Veraguas, J., Bartl, D., Beiglböck, M., and Eder, M. (2020). Adapted Wasserstein distances and stability in mathematical finance. *Finance and Stochastics*, 24, 601–632.
- Chen, K.T. (1957). Integration of paths, geometric invariants and a generalized Baker-Hausdorff formula. *Annals of Mathematics*, 65(1), 163–178.
- Tong, A., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Fatras, K., Wolf, G., and Bengio, Y. (2024). Improving and generalizing flow-matching for marginal inference in latent variable models. *arXiv:2302.00482*.
- Shi, Y., De Bortoli, V., Campbell, A., and Doucet, A. (2024). Diffusion Schrödinger Bridge Matching. *NeurIPS 2023*.
- Guyon, J. (2024). Dispersion-constrained martingale Schrödinger problems and the exact joint S&P 500/VIX smile calibration puzzle. *Finance and Stochastics*, 28, 497–547.
- Lyons, T. (1998). Differential equations driven by rough signals. *Revista Matemática Iberoamericana*, 14(2), 215–310.
