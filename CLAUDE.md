# CLAUDE.md

Reference guide for the regime-switching Gaussian process project on cross-sectional equity returns. Keep this up to date — future sessions start here.

---

## Current status

*(update this line every working session)*

- E-step: not yet implemented
- M-step: not yet implemented
- Toy tests: not yet written
- Real data: Stage 1 (raw → parquet) complete. Stage 2 (characteristic panel + preprocessing) complete on full CCM universe — `panel.pickle` has T=300, D=10, mean N_t≈7100, X ∈ [−0.5, 0.5]. S&P 500 point-in-time filter and macro Z still pending (both dormant behind file-existence switches).
- Python module for Stage 2 lives under `rsgp/` rather than `data/` — macOS case-insensitive FS collides with the `Data/` artifact dir.

---

## The research question

Which stock characteristics matter for returns, in which market environment?

Two gaps in existing cross-sectional asset pricing:
1. **Functional form.** Fama-MacBeth assumes linear. Gu, Kelly & Xiu (2020) show ML beats linear, but still fits one time-invariant function.
2. **Non-stationarity.** The characteristics → returns map shifts with the macro regime (momentum crashes, factor premia cycles). Standard ML ignores this.

---

## Model

Three pieces stitched into one generative story:

$$r_{n,t} = f_{k_t}(x_{n,t}) + \varepsilon_{n,t}, \quad f_k \sim \mathcal{GP}(0, \kappa_k)$$

- **GP with ARD kernel** — non-parametric characteristics → returns mapping with calibrated uncertainty. ARD lengthscales tell you which characteristics matter.
- **K-state HMM** — the pricing function itself switches across latent regimes; different market environments get fundamentally different $f_k$.
- **Non-homogeneous transitions** — regime-switch probabilities are a logistic function of observable macro covariates (VIX, credit spread, term spread), not constant.

**Inference.** Two-stage EM (Li & Ma 2023). E-step = forward-backward with GP marginal likelihood $\mathcal{N}(r_t \mid 0, K_k + \sigma_k^2 I)$ as emission. M-step = GP hyperparameters + logistic transition coefficients, each weighted by posterior regime responsibilities.

---

## Novelty framing (read this before writing anything for the report)

Instructor scored novelty 7/10 and flagged that the project looks like "Filipović & Pasricha + Li & Ma." The statistical machinery — HMM governing regime-dependent GPs — already exists (Park & Lee HMM-GPSM, Song et al. HM-GPFR, Marino & Cicirello). **The novelty is in three specific things that need to be foregrounded, not buried:**

1. **Macro-driven (non-homogeneous) transitions.** Not in either parent paper. Regimes don't switch randomly, they switch *because of* observable economic conditions. This is the genuine methodological contribution.
2. **Regime-dependent ARD lengthscales.** Which characteristics matter more in crisis vs. calm — falls out of the framework for free, needs to be highlighted as an empirical finding.
3. **Regime-conditional long-short alpha.** Shows that regime-aware nonparametric pricing translates into economically meaningful portfolio returns. Moves the project from "methodological combination" to "new empirical finding."

First two are already in the model. Third needs the eval pipeline to explicitly produce it.

---

## Economic motivation (also read before writing)

Instructor pushed back hard on "unknown functional form" as the motivation — it doesn't distinguish GP from RF or any reduced-form model. The actual argument for a probabilistic approach in cross-sectional pricing:

- **Posterior distributions over expected returns**, not point estimates. Lets you distinguish "this characteristic predicts returns" from "we're uncertain about the relationship."
- **Feeds portfolio construction.** Uncertainty on $\hat{r}_{n,t}$ is directly usable in mean-variance and Bayesian portfolio optimization; a random forest gives you nothing comparable.
- **Regime uncertainty is first-class.** The posterior $p(k_t \mid \text{data})$ tells you how confident the model is about the current regime — economically meaningful, and not available from frequentist alternatives.

Don't write the framing as "ML is better." Write it as "uncertainty quantification and regime identification are the economic quantities we care about, and the probabilistic framework delivers them natively."

---

## Shape and notation conventions

Pin these once; every file respects them.

- `T` — number of months
- `N_t` — number of stocks in month `t` (ragged; S&P constituents change)
- `D` — number of firm characteristics
- `K` — number of regimes (start K=2)
- `M` — number of macro covariates driving transitions

| Object | Shape | Notes |
|---|---|---|
| `returns` | list of length T, each `(N_t,)` | monthly excess returns |
| `X` | list of length T, each `(N_t, D)` | standardized characteristics |
| `Z` | `(T, M)` | macro covariates for transitions |
| `gamma` | `(T, K)` | posterior regime responsibilities |
| `xi` | `(T-1, K, K)` | pairwise posterior for transitions |
| `lengthscales` | `(K, D)` | one ARD vector per regime |
| `sigma_noise` | `(K,)` | observation noise per regime |
| `transition_W` | `(K, K, M)` | logistic coefficients on macro |
| `transition_b` | `(K, K)` | logistic intercepts |

Ragged handling: lists of per-month tensors, **not** padded. Padding introduces silent bugs in the GP marginal likelihood.

---

## Code layout

```
/models
  gp.py                 # GP emission with ARD, marginal log-likelihood
  hmm.py                # forward-backward (log-space), Viterbi
  transitions.py        # logistic non-homogeneous transition matrix
/inference
  em.py                 # E-step + M-step orchestrator, ELBO tracking
  init.py               # K-means-on-volatility regime initialization
/rsgp                   # named rsgp/ not data/ — macOS case-insensitive FS vs Data/
  characteristics.py    # monthly panel, Compustat lag, momentum, idio vol → char_panel_raw.parquet
  preprocess.py         # universe filter, excess returns, delisting, rank transform → panel.pickle
  macro.py              # FRED → Z.parquet (dormant until fred_macro.parquet arrives)
/eval
  metrics.py            # out-of-sample R² (Gu/Kelly/Xiu formulation)
  portfolios.py         # decile long-short, Sharpe, alpha
  baselines.py          # Fama-MacBeth, single-regime GP, RF
/tests
  toy_gp.py             # GP + ARD recovery
  toy_hmm.py            # HMM regime recovery with fixed emissions
  toy_transitions.py    # logistic covariate recovery
  toy_integrated.py     # full pipeline on 100 months × 100 stocks synthetic
/scripts
  run_real.py           # S&P 500 fit, 2000–2024
  make_tables.py        # Table 1, Table 2, Figure 4
```

---

## How to run

```bash
# Environment
conda env create -f environment.yml     # Python 3.11, PyTorch 2.3, scipy, pandas
conda activate rsgp

# Toy tests (should all pass before real data)
python -m tests.toy_gp              # ~2 min
python -m tests.toy_hmm             # ~1 min
python -m tests.toy_transitions     # ~1 min
python -m tests.toy_integrated      # ~15 min

# Real data
python scripts/run_real.py --k 2 --kernel se_ard --start 2000 --end 2024
python scripts/run_real.py --k 2 --kernel matern52_ard --start 2000 --end 2024

# Eval tables
python scripts/make_tables.py
```

---

## Validation gates (concrete pass criteria)

**Do not run on real data until all four pass.**

1. **GP + ARD recovery** (`toy_gp.py`). Generate `y = f(x) + ε` with known ARD lengthscales, `D=5`, `N=200`, `σ=0.1`. Pass: recovered lengthscales within 20% relative error; recovered `σ` within 10%.
2. **HMM recovery** (`toy_hmm.py`). Simulate 2-regime HMM with well-separated Gaussian emissions. Pass: Viterbi classification accuracy > 0.90 after label alignment.
3. **Transition recovery** (`toy_transitions.py`). Simulate logistic transitions driven by known `Z`. Pass: coefficient RMSE < 0.1, correct sign on every coefficient.
4. **Integrated pipeline** (`toy_integrated.py`). 100 months × 100 stocks, `D=5`, `K=2`, known regime sequence. Pass: regime recovery > 0.85, ARD lengthscales per regime within 30%, transition coefficients within RMSE 0.15.

---

## Numerical stability (things that will silently break)

- **Forward-backward in log-space.** Underflows by month 20 otherwise. Use `logsumexp`.
- **GP jitter.** Add `1e-6 * I` to `K_k + σ²I` before Cholesky. Increase to `1e-4` if still failing.
- **Logistic transition clipping.** Clip linear predictor to `[-30, 30]` before sigmoid, or macro spikes (March 2020 VIX) produce NaN gradients.
- **M-step optimizer.** Use LBFGS with 3–5 random restarts for GP hyperparameters. Adam is too slow and too noisy at this problem size.
- **Cholesky fallback.** Wrap every Cholesky in try/except; on failure, retry with 10× jitter before giving up.

---

## EM gotchas

- **Label switching.** Regime 0 in one run is regime 1 in another. After every fit, relabel regimes by average `|r|` (ascending) or average VIX responsibility, so plots compare across seeds.
- **Initialization.** EM is non-convex here. Init regimes by K-means on rolling 3-month volatility of the market return. Never use random init — convergence to junk local optima is common.
- **Convergence.** Relative change in ELBO < `1e-5` for 3 consecutive iterations, hard cap at 100 iterations.
- **ELBO must be monotone.** If it decreases, there's a bug in the M-step weighting or the E-step normalization. Assert this in every test.

---

## Data specification

(Instructor flagged vagueness here — be precise.)

- **Universe.** S&P 500 constituents, point-in-time (to avoid survivorship bias). Handle entries/exits by including stocks only during months they were in the index.
- **Frequency.** Monthly returns, rebalanced last trading day of the month.
- **Window.** 2000-01 to 2024-12 (300 months). Pre-2000 constituent data is thinner and the macro regime is arguably different.
- **Characteristics.** Target ~20 firm characteristics spanning Gu/Kelly/Xiu's canonical categories: size (market cap, log market cap), value (B/M, E/P, CF/P), momentum (12-1, 6-1, short-reversal), profitability (ROE, ROA, gross profitability), investment (asset growth, investment/assets), liquidity (turnover, bid-ask), risk (beta, idiosyncratic vol, max daily return). Final list locked before first real-data run.
- **Macro covariates for transitions.** VIX (level + monthly change), BAA–AAA credit spread, term spread (10Y–3M), TED spread. Lagged by one month to avoid look-ahead.
- **Sources.** CRSP via WRDS for returns, Compustat for fundamentals, FRED for macro, CBOE for VIX.

**Preprocessing decisions (lock these and don't change mid-project):**
- Returns: excess over 1-month T-bill.
- Characteristics: cross-sectionally ranked within each month, mapped to `[-0.5, 0.5]` (Gu/Kelly/Xiu convention). This is winsorization + standardization in one step and handles outliers.
- Missing characteristics: cross-sectional median imputation within month.
- Delisting returns: CRSP delisting return when available, otherwise -30% for performance-related delistings, 0 otherwise.

---

## Evaluation protocol

**Training pipeline.** Rolling window with expanding training data. Initial training: 2000–2009 (120 months). Predict 2010-01. Retrain including 2010-01, predict 2010-02. Continue through 2024-12. Total: 180 out-of-sample months.

**Goal statement.** Both explanation (regime-dependent ARD tells us which characteristics are priced when) and prediction (OOS R² and portfolio performance). Be explicit about this in the report.

**Cross-sectional R² (Gu/Kelly/Xiu formulation, no intercept):**

$$R^2_{\text{OOS}} = 1 - \frac{\sum_{t,n} (r_{n,t} - \hat{r}_{n,t})^2}{\sum_{t,n} r_{n,t}^2}$$

Pooled across all OOS months. Report overall, plus per-regime (conditional on Viterbi-inferred regime at prediction time).

**Portfolio construction.**
- Sort stocks each month by predicted $\hat{r}_{n,t+1}$ into deciles.
- Long top decile, short bottom decile, equal-weighted.
- Monthly rebalance, no transaction costs in the primary table (include 10 bps one-way as a robustness check).
- Report: annualized mean return, annualized volatility, annualized Sharpe ratio (with 1-month T-bill as rf), max drawdown.

**Alpha.** CAPM alpha and Fama-French 5-factor alpha on the long-short return series. This is the instructor's explicit suggestion — it's the "so what" of the paper.

**Regime-conditional alpha.** Compute long-short Sharpe and FF5 alpha separately in each Viterbi-inferred regime. This is where the novelty payoff shows up: the model predicts that regime 1 and regime 2 have different characteristic pricing, and the portfolio results should confirm it.

---

## Baselines

All baselines use the same universe, characteristics, and rolling-window protocol.

1. **Fama-MacBeth.** Month-by-month cross-sectional OLS of returns on characteristics. Standard benchmark from the empirical asset pricing literature.
2. **Single-regime GP with SE-ARD kernel.** Clean ablation of the regime-switching component. If the full model doesn't beat this, the HMM adds nothing.
3. **Random forest.** 500 trees, max depth tuned on a validation slice. Justified as a baseline by Gu, Kelly & Xiu (2020), who identify tree-based ensembles as the strongest non-neural ML benchmark for cross-sectional return prediction. **Cite this explicitly** in the report — instructor asked for the citation.
4. **(Optional) Neural network.** 3-layer MLP, also from Gu/Kelly/Xiu. Include if time allows; otherwise flag as future work.

Baseline hyperparameters frozen before the primary comparison runs.

---

## Deliverables

- **Table 1.** Out-of-sample cross-sectional R² — overall and per-regime — for RSGP (SE-ARD), RSGP (Matérn 5/2-ARD), single-regime GP, Fama-MacBeth, Random Forest. RSGP should win overall, and the per-regime numbers should show the regime-conditioning matters.
- **Table 2.** Long-short decile portfolio statistics — annualized return, volatility, Sharpe, max drawdown, CAPM alpha, FF5 alpha. Same model set as Table 1, plus regime-conditional RSGP rows.
- **Figure 4.** Regime-dependent ARD lengthscale bar chart. Grouped bars, one group per characteristic, two bars per group (one per regime). The interpretability payoff — this is the figure that makes the economic story land.
- **Figure 5 (bonus).** Posterior regime probability time series with NBER recession shading, to show the learned regimes correspond to recognizable macro episodes.

---

## References

- **Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning."** Canonical ML-for-cross-section paper. Source of the characteristic list, the cross-sectional R² formula, the rank-based characteristic transformation, and the RF/NN baselines.
- **Filipović & Pasricha (2022).** GP applied to cross-sectional pricing, single regime. One of the two parent papers.
- **Li & Ma (2023).** HMM with GP emissions, two-stage EM. Source of the inference strategy. Second parent paper.
- **Filardo (1994).** Original non-homogeneous Markov switching with macro covariates. Justifies the logistic transition parametrization.
- **Guidolin & Timmermann (2008).** Regime-switching asset pricing with macro drivers — economic motivation for regime-dependent pricing.
- **Park & Lee, HMM-GPSM.** Closest existing method methodologically, but univariate time-series, homogeneous transitions.
- **Song et al., HM-GPFR.** HMM-GP for functional regression. Different domain, similar machinery.
- **Marino & Cicirello.** HMM-GP for structural health monitoring. Engineering domain, confirms the basic statistical machinery is not novel on its own.

---

## Open questions

- How to pick K? Start with K=2 (crisis/calm), check BIC at K=3. Higher K probably not identifiable with 300 months.
- Should ARD lengthscales share a prior across regimes? A hierarchical prior would regularize and could improve small-regime estimation. Not in v1, flag for future work.
- O(N³) GP cost at N=500. Mitigation: randomly subsample 200 stocks per month for training; predict on all. Document the subsampling and show robustness at N=300.
- How to handle the boundary between regimes in OOS prediction? Use soft prediction $\hat{r}_{n,t} = \sum_k p(k_t \mid \text{data}_{<t}) \hat{r}_{n,t}^{(k)}$, not hard Viterbi assignment. This uses the full posterior and is strictly better when the regime is uncertain.
