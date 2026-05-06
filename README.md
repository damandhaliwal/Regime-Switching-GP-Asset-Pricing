# Regime-Switching Gaussian Process Asset Pricing (RSGP)

A probabilistic machine-learning framework for cross-sectional equity return prediction that unifies three statistical components: **non-parametric Gaussian Process pricing functions**, a **K-state Hidden Markov Model** over market regimes, and **macro-covariate-driven non-homogeneous transitions** between regimes. The model delivers posterior distributions over expected returns, interpretable ARD lengthscales, and regime-conditional portfolio alphas.

---

## Table of Contents

- [Research Motivation](#research-motivation)
- [Model](#model)
  - [Generative Story](#generative-story)
  - [Novelty](#novelty)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Pipeline](#data-pipeline)
  - [Required Raw Files](#required-raw-files)
  - [Stage 1: Convert Raw → Parquet](#stage-1-convert-raw--parquet)
  - [Stage 2: Build the Model-Ready Panel](#stage-2-build-the-model-ready-panel)
- [Validation Tests](#validation-tests)
- [Running the Real-Data Experiment](#running-the-real-data-experiment)
- [Generating Tables and Figures](#generating-tables-and-figures)
- [Evaluation Protocol](#evaluation-protocol)
  - [Out-of-Sample R²](#out-of-sample-r)
  - [Portfolio Construction](#portfolio-construction)
  - [Baselines](#baselines)
- [Key Design Decisions](#key-design-decisions)
- [Outputs](#outputs)
- [References](#references)

---

## Research Motivation

Two persistent gaps exist in cross-sectional asset pricing:

1. **Functional form.** Fama-MacBeth (1973) assumes linearity. While machine-learning models (Gu, Kelly & Xiu 2020) outperform linear benchmarks, they fit a single time-invariant pricing function.
2. **Non-stationarity.** The characteristics → returns mapping shifts with the macroeconomic environment (momentum crashes, factor premia cycles, liquidity crises). Standard ML ignores this regime dependence.

The RSGP framework addresses both gaps. By placing a GP over each regime's pricing function, we obtain **posterior distributions over expected returns** — not point estimates. This enables:

- Quantified uncertainty on stock-level predictions (directly usable in Bayesian portfolio optimization).
- Identification of **which firm characteristics are priced** in each regime via regime-dependent ARD lengthscales.
- A first-class measure of **regime uncertainty** via the posterior $p(k_t \mid \text{data})$.

---

## Model

### Generative Story

$$r_{n,t} = f_{k_t}(x_{n,t}) + \varepsilon_{n,t}, \qquad f_k \sim \mathcal{GP}(0, \kappa_k), \qquad \varepsilon_{n,t} \sim \mathcal{N}(0, \sigma_k^2)$$

| Symbol | Meaning |
|---|---|
| $r_{n,t}$ | Excess return of stock $n$ in month $t$ |
| $x_{n,t} \in \mathbb{R}^D$ | Firm characteristics (rank-transformed to $[-0.5, 0.5]$) |
| $k_t \in \{0, \ldots, K-1\}$ | Latent market regime at time $t$ |
| $f_k$ | Regime-$k$ pricing function, drawn from a GP with ARD kernel $\kappa_k$ |
| $\sigma_k^2$ | Regime-$k$ observation noise variance |

**Regime dynamics.** $k_t$ follows a non-homogeneous Markov chain:

$$P(k_t = j \mid k_{t-1} = i, z_t) = \text{softmax}_j \bigl( W[i,j,:] \cdot z_t + b[i,j] \bigr)$$

where $z_t \in \mathbb{R}^M$ contains lagged macro covariates (VIX level and change, credit spread, term spread).

**Inference.** Two-stage EM (Li & Ma 2023 strategy):

- **E-step.** Log-space forward-backward algorithm using GP marginal log-likelihoods $\log \mathcal{N}(r_t \mid 0, K_k + \sigma_k^2 I)$ as emissions. Returns posterior responsibilities $\gamma_{t,k} = p(k_t = k \mid \mathbf{r}, \mathbf{X}, \mathbf{Z})$ and pairwise posteriors $\xi_{t,j,k}$.
- **M-step.** Weighted GP hyperparameter optimization (L-BFGS-B with random restarts via JAX JIT) and multinomial logistic regression on transition weights $\xi_{t,j,k}$.

**Prediction.** Out-of-sample return predictions use the full predictive posterior, mixing over regimes:

$$\hat{r}_{n,t} = \sum_k p(k_t = k \mid \text{history}) \cdot \mathbb{E}[f_k(x_{n,t}) \mid \text{history}]$$

### Novelty

| Contribution | Description |
|---|---|
| Macro-driven transitions | Transition probabilities are a function of observable economic conditions (Filardo 1994), not constant. This is absent from both parent papers (Filipović & Pasricha 2022; Li & Ma 2023). |
| Regime-dependent ARD | Separate lengthscale vectors per regime reveal *which* characteristics are priced in each market environment — a direct empirical finding. |
| Regime-conditional alpha | Long-short portfolio performance is computed separately per Viterbi-inferred regime, connecting methodology to economically meaningful return attribution. |

---

## Repository Structure

```
code/
├── models/
│   ├── gp.py              # GP with SE-ARD / Matérn-5/2-ARD kernel; marginal log-likelihood; posterior prediction
│   ├── gp_jax.py          # JAX JIT backend for L-BFGS-B hyperparameter optimization
│   ├── hmm.py             # Log-space forward-backward, Viterbi, posterior smoother
│   └── transitions.py     # Non-homogeneous logistic transitions (Filardo 1994); L-BFGS-B fit
│
├── inference/
│   ├── em.py              # EM training loop, EMConfig, EMResult, predict_next_month
│   └── init.py            # K-means-on-volatility deterministic initialization
│
├── rsgp/
│   ├── characteristics.py # Monthly panel: Compustat lag, momentum, idiosyncratic vol → char_panel_raw.parquet
│   ├── preprocess.py      # Universe filter, excess returns, delisting, rank transform → panel.pickle
│   ├── macro.py           # FRED macro covariates → Z.parquet
│   └── data.py            # load_aligned_panel() — merge panel.pickle + Z.parquet into AlignedPanel
│
├── eval/
│   ├── metrics.py         # OOS R², regime-conditional R², Sharpe, max drawdown, CAPM/FF5 alpha
│   ├── portfolios.py      # Decile long-short portfolio construction
│   └── baselines.py       # Fama-MacBeth, single-regime GP, random forest (rolling window)
│
├── tests/
│   ├── toy_gp.py          # Validation gate 1: ARD lengthscale + noise recovery
│   ├── toy_hmm.py         # Validation gate 2: Viterbi regime accuracy > 0.90
│   ├── toy_transitions.py # Validation gate 3: logistic coefficient recovery
│   ├── toy_integrated.py  # Validation gate 4: full pipeline on synthetic data
│   ├── test_characteristics.py
│   └── test_idio_vol.py
│
├── scripts/
│   ├── convert_raw_to_parquet.py  # Stage 1: CSV/SAS → typed parquets
│   ├── build_panel.py             # Stage 2: characteristics + preprocess pipeline
│   ├── verify_parquet.py          # Sanity-check parquet contents
│   ├── run_real.py                # Rolling OOS experiment (parallelized across months)
│   ├── make_tables.py             # Table 1, Table 2, Figures 4–5
│   └── make_toy_figures.py        # Diagnostic figures for toy tests
│
├── environment.yml        # Conda environment (Python 3.11, PyTorch, JAX, polars, scikit-learn)
│
output/
├── figures/               # PNG figures (regime probabilities, ARD lengthscales, alpha comparison, …)
└── tables/                # CSV + LaTeX tables (OOS R², portfolio statistics)

paper/                     # LaTeX source and bibliography
Proposal/                  # Original project proposal (LaTeX)
```

> **macOS note.** The data module lives under `rsgp/` (not `data/`) because macOS has a case-insensitive filesystem and the artifact directory is named `Data/`.

---

## Installation

```bash
conda env create -f code/environment.yml
conda activate rsgp
```

The environment includes Python 3.11, NumPy, SciPy, Polars ≥ 0.20, pandas ≥ 2.0, PyArrow ≥ 14, Matplotlib, scikit-learn, and JAX 0.4.26 (CPU). JAX is used as a JIT/autodiff backend for GP hyperparameter optimization; the CPU path is the supported configuration.

All scripts assume they are run from the `code/` directory:

```bash
cd code/
```

---

## Data Pipeline

### Required Raw Files

Place the following files under `Data/raw/` (relative to `code/`):

| File | Source | Description |
|---|---|---|
| `ccm_monthly.sas7bdat` or `.csv` | CRSP/Compustat (WRDS) | Monthly stock data merged with Compustat annual fundamentals |
| `crsp_daily.csv` | CRSP (WRDS) | Daily CRSP returns (used for idiosyncratic volatility) |
| `ff_factors_daily.csv` | Kenneth French Data Library | Daily Fama-French factors |
| `ff_factors_monthly.csv` | Kenneth French Data Library | Monthly FF factors (mkt, SMB, HML, RMW, CMA, rf) |
| `sp500_constituents.csv` | Compustat (WRDS) Security Universe | Point-in-time S&P 500 membership (permno, start, ending) |
| `sp500_index.csv` | CRSP | Daily S&P 500 total return index (`sprtrn`) |
| `delisting.csv` | CRSP | Delisting returns and codes |
| `fred_macro.parquet` | FRED (pre-fetched) | VIX, BAA–AAA credit spread, term spread, TED spread |

### Stage 1: Convert Raw → Parquet

```bash
python scripts/convert_raw_to_parquet.py
```

Outputs typed parquet files to `Data/parquet/`:
`ccm_merged.parquet`, `crsp_daily.parquet`, `ff_factors_daily.parquet`, `ff_factors.parquet`, `sp500_constituents.parquet`, `sp500_index.parquet`, `delisting.parquet`.

Use `scripts/verify_parquet.py` to sanity-check column names and row counts.

### Stage 2: Build the Model-Ready Panel

```bash
python scripts/build_panel.py
```

This runs two sub-stages in sequence:

**Stage 2a — Characteristics** (`rsgp/characteristics.py`):
Computes 10 firm characteristics from the merged panel (2000-01 to 2024-12):

| Characteristic | Description |
|---|---|
| `log_mktcap` | Log market capitalization |
| `bm` | Book-to-market ratio |
| `ep` | Earnings-to-price |
| `mom_12_1` | 12-1 month momentum |
| `mom_1` | 1-month short-term reversal |
| `roe` | Return on equity |
| `roa` | Return on assets |
| `asset_growth` | Year-over-year asset growth |
| `turnover` | Share turnover |
| `idio_vol` | Idiosyncratic volatility (3-month rolling residual from daily FF3 regression) |

Output: `Data/parquet/char_panel_raw.parquet`.

**Stage 2b — Preprocessing** (`rsgp/preprocess.py`):
1. S&P 500 point-in-time universe filter (removes survivorship bias).
2. Excess returns (subtracts 1-month T-bill rate from `ff_factors.parquet`).
3. Delisting return composition (CRSP `DLRET` if available; else −30% for performance-related delistings; 0 otherwise).
4. Cross-sectional rank transform to $[-0.5, 0.5]$ per month (Gu/Kelly/Xiu convention); missing values imputed with 0 (cross-sectional median of the transformed scale).
5. Drop months with fewer than 30 stocks.

Output: `Data/parquet/panel.pickle` — ragged lists `{dates, returns, X, permnos}` with T ≈ 300 months and mean $N_t \approx 470$ stocks.

**Macro covariates** (`rsgp/macro.py`):
```bash
python -m rsgp.macro   # reads Data/parquet/fred_macro.parquet → Data/parquet/Z.parquet
```

Produces the transition covariate matrix $Z$ (T × M) with columns `vix_lag1`, `vix_chg_lag1`, `credit_spread_lag1`, `term_spread_lag1`. All covariates are lagged one month to avoid look-ahead.

---

## Validation Tests

All four gates must pass before running on real data. From `code/`:

```bash
# Gate 1: GP + ARD recovery (~2 min)
# Pass: median lengthscale rel. error < 20% per dimension; median σ_n rel. error < 10%
PYTHONPATH=. python -m tests.toy_gp

# Gate 2: HMM regime recovery (~1 min)
# Pass: Viterbi accuracy > 0.90 after label alignment
PYTHONPATH=. python -m tests.toy_hmm

# Gate 3: logistic transition coefficient recovery (~1 min)
# Pass: coefficient RMSE < 0.1; correct sign on every coefficient
PYTHONPATH=. python -m tests.toy_transitions

# Gate 4: full integrated pipeline on synthetic data (~15 min)
# 100 months × 100 stocks, D=5, K=2
# Pass: regime recovery > 0.85; ARD lengthscales per regime within 30%; transition RMSE < 0.15
PYTHONPATH=. python -m tests.toy_integrated
```

---

## Running the Real-Data Experiment

```bash
cd code/

# SE-ARD kernel, 2010-01 to 2024-12 OOS, 8 parallel workers
python scripts/run_real.py \
    --oos-start 2010-01 \
    --kernel se_ard \
    --workers 8 \
    --output-dir ../output/real

# Matérn-5/2 ARD kernel
python scripts/run_real.py \
    --oos-start 2010-01 \
    --kernel matern52_ard \
    --workers 8 \
    --output-dir ../output/real_matern
```

**What happens:** For each OOS month $t$ from `--oos-start` onward, a worker:
1. Runs a cold EM fit on all months up to $t-1$ (expanding window).
2. Predicts returns for month $t$ using `predict_next_month`, which computes a regime-probability-weighted mixture of per-regime GP posteriors.
3. Writes prediction rows, regime rows, and parameter snapshots.

**Key CLI options:**

| Flag | Default | Description |
|---|---|---|
| `--oos-start` | `2010-01` | First OOS month |
| `--oos-end` | (last month) | Last OOS month (inclusive) |
| `--kernel` | `se_ard` | GP kernel: `se_ard` or `matern52_ard` |
| `--workers` | `8` | Number of parallel worker processes |
| `--max-iter` | `8` | Maximum EM iterations per OOS month |
| `--max-monthly-points` | `100` | Max stocks subsampled per month for GP fitting |
| `--max-prediction-points` | `600` | Max training points used in `predict_next_month` |
| `--max-months-per-regime-fit` | `25` | Cap on months included per regime in M-step |

**Outputs** (in `--output-dir`):

| File | Description |
|---|---|
| `predictions.parquet` | Per-stock predictions: `date`, `permno`, `prediction`, `prediction_var`, `actual_return` |
| `regimes.parquet` | Per-month regime diagnostics: Viterbi state, next-regime probabilities, EM convergence |
| `params.pkl` | Full parameter snapshots (GP hyperparams, transition weights) for each OOS month |
| `config.json` | Script arguments and EMConfig for reproducibility |

---

## Generating Tables and Figures

```bash
cd code/
python scripts/make_tables.py
```

Reads `output/real/predictions.parquet` and `output/real/regimes.parquet`, baseline results from `output/baselines/predictions.parquet` (generated separately by `eval/baselines.py`), and Fama-French factors from `Data/parquet/ff_factors.parquet`. Writes:

| Output | Location | Contents |
|---|---|---|
| Table 1 | `output/tables/table_1_prediction_performance.{csv,tex}` | OOS cross-sectional R² — overall and per-regime — for RSGP (SE-ARD), RSGP (Matérn), single-regime GP, Fama-MacBeth, Random Forest |
| Table 2 | `output/tables/table_2_portfolio_performance.{csv,tex}` | Long-short decile portfolio: annualized return, volatility, Sharpe, max drawdown, CAPM alpha, FF5 alpha |
| Figure 4 | `output/figures/regime_lengthscales.png` | Regime-dependent ARD lengthscale bar chart (one bar per regime per characteristic) |
| Figure 5 | `output/figures/regime_probabilities.png` | Posterior regime probability time series with NBER recession shading |

Toy diagnostic figures (produced by `scripts/make_toy_figures.py`):

| Figure | Description |
|---|---|
| `toy_ard_recovery.png` | Recovered vs. true ARD lengthscales across draws |
| `toy_regime_recovery.png` | Viterbi accuracy and gamma time series for the integrated test |
| `transition_vix_sensitivity.png` | Transition probabilities as a function of VIX (synthetic) |

---

## Evaluation Protocol

### Out-of-Sample R²

Following Gu, Kelly & Xiu (2020), the pooled cross-sectional OOS R² with no intercept:

$$R^2_{\text{OOS}} = 1 - \frac{\sum_{t,n} (r_{n,t} - \hat{r}_{n,t})^2}{\sum_{t,n} r_{n,t}^2}$$

Reported overall and per-regime (conditional on the Viterbi-inferred regime at each prediction month).

### Portfolio Construction

- Sort stocks each month by predicted $\hat{r}_{n,t+1}$ into deciles.
- Long the top decile, short the bottom decile, equal-weighted within each leg.
- Monthly rebalance; no transaction costs (primary); 10 bps one-way as robustness.
- Report: annualized mean return, annualized volatility, annualized Sharpe ratio, max drawdown, CAPM alpha, Fama-French 5-factor alpha.
- **Regime-conditional:** compute portfolio statistics separately for months assigned to each regime by Viterbi decoding.

### Baselines

All baselines use the same rolling-window protocol (initial training 2000–2009, then expanding).

| Baseline | Implementation |
|---|---|
| Fama-MacBeth | Monthly cross-sectional OLS on characteristics; coefficients averaged over all training months |
| Single-Regime GP | SE-ARD GP fit on a pooled subsample of training data (800 points); ablates the HMM component |
| Random Forest | 500 trees, scikit-learn `RandomForestRegressor`, subsampled to 10 000 training rows; benchmark from Gu/Kelly/Xiu (2020) |

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Forward-backward precision | Log-space with `logsumexp` | Prevents underflow past month ~20 |
| Cholesky stability | Jitter $10^{-6} \cdot I$ with up to $10^3\times$ fallback | Ensures PSD kernel matrices |
| Logistic clip | Linear predictors clipped to $[-30, 30]$ | Prevents NaN gradients from macro spikes (e.g., March 2020 VIX) |
| GP optimizer | L-BFGS-B with 3–5 random restarts via JAX JIT | LBFGS converges reliably; JAX JIT provides autodiff gradients and batched Cholesky |
| Regime initialization | K-means on rolling 3-month S&P 500 volatility | Deterministic, avoids random-init local optima |
| Label switching | Regimes relabeled by ascending GP noise variance after every fit | Ensures regime 0 = low-volatility / calm and regime 1 = high-volatility / crisis |
| Ragged panel | Lists of per-month arrays; no padding | Padding silently corrupts GP marginal likelihoods (varying $N_t$) |
| Subsampling | Up to 100 stocks/month for fitting; up to 600 for prediction | Controls $O(N^3)$ GP cost; documented in all outputs |

---

## Outputs

Pre-computed outputs are committed to the `output/` directory:

```
output/
├── figures/
│   ├── regime_probabilities.png      # Posterior regime probs over 2010-2024
│   ├── regime_lengthscales.png       # ARD lengthscales per regime per characteristic
│   ├── alpha_comparison.png          # CAPM / FF5 alpha across models
│   ├── kernel_sensitivity.png        # SE-ARD vs. Matérn-5/2 performance comparison
│   ├── transition_vix_sensitivity.png
│   ├── toy_ard_recovery.png
│   └── toy_regime_recovery.png
└── tables/
    ├── table_1_prediction_performance.csv
    ├── table_1_prediction_performance.tex
    ├── table_2_portfolio_performance.csv
    └── table_2_portfolio_performance.tex
```

---

## References

- **Gu, S., Kelly, B., & Xiu, D. (2020).** Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2273. *(Canonical ML cross-sectional paper; source of OOS R² formula, rank transform, RF/NN baselines.)*
- **Filipović, D., & Pasricha, G. (2022).** Gaussian process cross-sectional pricing (single regime). *(First parent paper.)*
- **Li, M., & Ma, T. (2023).** Hidden Markov models with GP emissions; two-stage EM. *(Second parent paper; source of inference strategy.)*
- **Filardo, A. J. (1994).** Business-cycle phases and their transitional dynamics. *Journal of Business & Economic Statistics*, 12(3), 299–308. *(Non-homogeneous Markov switching with macro covariates.)*
- **Guidolin, M., & Timmermann, A. (2008).** International asset allocation under regime switching, skew, and kurtosis preferences. *Review of Financial Studies*, 21(2), 889–935. *(Regime-switching asset pricing with macro drivers.)*
- **Park, C., & Lee, J. (2023).** HMM-GPSM: HMM-governed GP for structural monitoring. *(Closest methodological precedent; univariate time-series, homogeneous transitions.)*
- **Fama, E. F., & MacBeth, J. D. (1973).** Risk, return, and equilibrium: Empirical tests. *Journal of Political Economy*, 81(3), 607–636.
