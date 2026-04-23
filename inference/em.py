"""EM training loop for regime-switching GP emissions with macro transitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.special import logsumexp

from inference.init import RegimeInitialization, initialize_regimes
from models.gp import GPHyperparams, fit_gp_hyperparameters, marginal_log_likelihood, predict
from models.hmm import posteriors, viterbi
from models.transitions import fit_logistic_weighted, log_transition_matrix
from rsgp.data import AlignedPanel, load_aligned_panel

DEFAULT_MACRO_COLS = ["vix_lag1", "vix_chg_lag1", "credit_spread_lag1", "term_spread_lag1"]


@dataclass
class EMConfig:
    K: int = 2
    max_iter: int = 100
    tol: float = 1e-5
    patience: int = 3
    gp_restarts: int = 4
    gp_seed: int = 0
    transition_restarts: int = 2
    transition_seed: int = 0
    transition_l2: float = 1e-4
    min_regime_mass: float = 1.0
    max_prediction_points: int = 800


@dataclass
class EMResult:
    dates: list
    gamma: np.ndarray
    xi: np.ndarray
    pi: np.ndarray
    gp_params: list[GPHyperparams]
    transition_W: np.ndarray
    transition_b: np.ndarray
    log_likelihood_history: list[float]
    viterbi_path: np.ndarray
    init_states: np.ndarray
    init_volatility: np.ndarray
    log_emit: np.ndarray
    z_columns: list[str] = field(default_factory=list)
    converged: bool = False


def _compute_gp_emission_log_likelihoods(
    X: Sequence[np.ndarray],
    returns: Sequence[np.ndarray],
    gp_params: Sequence[GPHyperparams],
) -> np.ndarray:
    T = len(X)
    K = len(gp_params)
    log_emit = np.empty((T, K), dtype=np.float64)
    for t, (X_t, y_t) in enumerate(zip(X, returns)):
        for k, hp in enumerate(gp_params):
            log_emit[t, k] = marginal_log_likelihood(X_t, y_t, hp.lengthscales, hp.signal_var, hp.noise_var)
    return log_emit


def _hard_xi_from_states(states: np.ndarray, k: int) -> np.ndarray:
    xi = np.zeros((len(states) - 1, k, k), dtype=np.float64)
    for t in range(len(states) - 1):
        xi[t, states[t], states[t + 1]] = 1.0
    return xi


def _transition_matrix_for_covariate(W: np.ndarray, b: np.ndarray, z: np.ndarray) -> np.ndarray:
    logits = np.einsum("jkm,m->jk", W, z) + b
    logits[:, 0] = 0.0
    logits -= logsumexp(logits, axis=1, keepdims=True)
    return np.exp(logits)


def _regime_order(gamma: np.ndarray, volatility: np.ndarray) -> np.ndarray:
    scores = []
    for k in range(gamma.shape[1]):
        weight = gamma[:, k].sum()
        if weight <= 0:
            scores.append(np.inf)
        else:
            scores.append(float(np.dot(gamma[:, k], volatility) / weight))
    return np.argsort(np.array(scores))


def _permute_xi(xi: np.ndarray, order: np.ndarray) -> np.ndarray:
    return xi[:, order][:, :, order]


def _permute_model(
    order: np.ndarray,
    gamma: np.ndarray,
    xi: np.ndarray,
    pi: np.ndarray,
    gp_params: Sequence[GPHyperparams],
    W: np.ndarray,
    b: np.ndarray,
    log_emit: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[GPHyperparams], np.ndarray, np.ndarray, np.ndarray | None]:
    perm_gamma = gamma[:, order]
    perm_xi = _permute_xi(xi, order)
    perm_pi = pi[order]
    perm_gp = [gp_params[idx] for idx in order]
    perm_W = W[order][:, order, :]
    perm_b = b[order][:, order]
    perm_log_emit = None if log_emit is None else log_emit[:, order]
    return perm_gamma, perm_xi, perm_pi, perm_gp, perm_W, perm_b, perm_log_emit


def _fit_gp_block(
    X: Sequence[np.ndarray],
    returns: Sequence[np.ndarray],
    gamma: np.ndarray,
    config: EMConfig,
    init_params: Sequence[GPHyperparams] | None = None,
) -> list[GPHyperparams]:
    gp_params: list[GPHyperparams] = []
    D = X[0].shape[1]
    for k in range(config.K):
        weights = np.asarray(gamma[:, k], dtype=np.float64)
        if weights.sum() < config.min_regime_mass:
            if init_params is None:
                gp_params.append(GPHyperparams(lengthscales=np.ones(D), signal_var=1.0, noise_var=0.1))
            else:
                gp_params.append(init_params[k])
            continue
        hp, _ = fit_gp_hyperparameters(
            data=list(zip(X, returns)),
            weights=weights,
            D=D,
            n_restarts=config.gp_restarts,
            init=None if init_params is None else init_params[k],
            seed=config.gp_seed + 17 * k,
        )
        gp_params.append(hp)
    return gp_params


def _fit_transition_block(
    Z: np.ndarray,
    xi: np.ndarray,
    config: EMConfig,
) -> tuple[np.ndarray, np.ndarray]:
    return fit_logistic_weighted(
        Z,
        xi,
        K=config.K,
        M=Z.shape[1],
        l2=config.transition_l2,
        n_restarts=config.transition_restarts,
        seed=config.transition_seed,
    )


def fit_em(
    dates: Sequence,
    returns: Sequence[np.ndarray],
    X: Sequence[np.ndarray],
    Z: np.ndarray,
    config: EMConfig | None = None,
    init: RegimeInitialization | None = None,
    z_columns: Sequence[str] | None = None,
) -> EMResult:
    config = config or EMConfig()
    dates = list(dates)
    returns = list(returns)
    X = list(X)
    Z = np.asarray(Z, dtype=np.float64)

    if not (len(dates) == len(returns) == len(X) == Z.shape[0]):
        raise ValueError("dates, returns, X, and Z must have the same length")
    if config.K < 1:
        raise ValueError("K must be >= 1")

    init = init or initialize_regimes(dates, k=config.K)
    gamma = init.gamma.copy()
    xi = _hard_xi_from_states(init.states, config.K)
    pi = gamma[0].copy()
    pi /= pi.sum()

    gp_params = _fit_gp_block(X, returns, gamma, config)
    W, b = _fit_transition_block(Z, xi, config)

    history: list[float] = []
    stale = 0
    converged = False

    for _ in range(config.max_iter):
        log_emit = _compute_gp_emission_log_likelihoods(X, returns, gp_params)
        gamma, xi, log_lik = posteriors(np.log(np.clip(pi, 1e-12, None)), log_transition_matrix(W, b, Z), log_emit)

        order = _regime_order(gamma, init.volatility)
        gamma, xi, pi, gp_params, W, b, log_emit = _permute_model(order, gamma, xi, pi, gp_params, W, b, log_emit)
        history.append(float(log_lik))
        if len(history) > 1 and history[-1] + 1e-8 < history[-2]:
            raise AssertionError(f"observed log-likelihood decreased: {history[-2]} -> {history[-1]}")

        if len(history) > 1:
            rel = abs(history[-1] - history[-2]) / max(1.0, abs(history[-2]))
            stale = stale + 1 if rel < config.tol else 0
            if stale >= config.patience:
                converged = True
                break

        pi = gamma[0].copy()
        pi /= pi.sum()
        gp_params = _fit_gp_block(X, returns, gamma, config, init_params=gp_params)
        W, b = _fit_transition_block(Z, xi, config)

    log_emit = _compute_gp_emission_log_likelihoods(X, returns, gp_params)
    gamma, xi, log_lik = posteriors(np.log(np.clip(pi, 1e-12, None)), log_transition_matrix(W, b, Z), log_emit)
    order = _regime_order(gamma, init.volatility)
    gamma, xi, pi, gp_params, W, b, log_emit = _permute_model(order, gamma, xi, pi, gp_params, W, b, log_emit)
    if not history or abs(log_lik - history[-1]) > 1e-10:
        if history and log_lik + 1e-8 < history[-1]:
            raise AssertionError(f"final observed log-likelihood decreased: {history[-1]} -> {log_lik}")
        history.append(float(log_lik))

    vpath = viterbi(np.log(np.clip(pi, 1e-12, None)), log_transition_matrix(W, b, Z), log_emit)
    return EMResult(
        dates=dates,
        gamma=gamma,
        xi=xi,
        pi=pi,
        gp_params=gp_params,
        transition_W=W,
        transition_b=b,
        log_likelihood_history=history,
        viterbi_path=vpath,
        init_states=init.states,
        init_volatility=np.asarray(init.volatility, dtype=np.float64),
        log_emit=log_emit,
        z_columns=list(z_columns or []),
        converged=converged,
    )


def fit_em_from_disk(
    config: EMConfig | None = None,
    required_macro_cols: Sequence[str] = DEFAULT_MACRO_COLS,
) -> tuple[EMResult, AlignedPanel]:
    aligned = load_aligned_panel(required_macro_cols=required_macro_cols)
    result = fit_em(
        dates=aligned.dates,
        returns=aligned.returns,
        X=aligned.X,
        Z=aligned.Z,
        config=config,
        z_columns=aligned.z_columns,
    )
    return result, aligned


def _allocate_training_points(
    sizes: np.ndarray,
    month_weights: np.ndarray,
    max_points: int,
) -> np.ndarray:
    mass = np.clip(month_weights, 0.0, None) * sizes
    if mass.sum() <= 0:
        mass = sizes.astype(np.float64)
    raw = max_points * mass / mass.sum()
    counts = np.minimum(np.floor(raw).astype(np.int32), sizes)
    remaining = max_points - int(counts.sum())
    if remaining > 0:
        frac = raw - np.floor(raw)
        order = np.argsort(-frac)
        for idx in order:
            if remaining == 0:
                break
            if counts[idx] < sizes[idx]:
                counts[idx] += 1
                remaining -= 1
    return counts


def _build_regime_training_subset(
    X: Sequence[np.ndarray],
    returns: Sequence[np.ndarray],
    month_weights: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    sizes = np.array([len(y_t) for y_t in returns], dtype=np.int32)
    counts = _allocate_training_points(sizes, month_weights, max_points=max_points)
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for count, X_t, y_t in zip(counts, X, returns):
        if count <= 0:
            continue
        if count >= len(y_t):
            xs.append(X_t)
            ys.append(y_t)
            continue
        idx = np.sort(rng.choice(len(y_t), size=count, replace=False))
        xs.append(X_t[idx])
        ys.append(y_t[idx])
    if not xs:
        xs = [X[-1]]
        ys = [returns[-1]]
    return np.vstack(xs), np.concatenate(ys)


def predict_next_month(
    result: EMResult,
    train_X: Sequence[np.ndarray],
    train_returns: Sequence[np.ndarray],
    test_X: np.ndarray,
    next_z: np.ndarray,
    max_points: int | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    max_points = max_points or EMConfig().max_prediction_points
    next_z = np.asarray(next_z, dtype=np.float64)

    regime_means = []
    regime_vars = []
    training_sizes = []
    for k, hp in enumerate(result.gp_params):
        X_sub, y_sub = _build_regime_training_subset(
            train_X,
            train_returns,
            month_weights=result.gamma[:, k],
            max_points=max_points,
            seed=seed + 101 * k,
        )
        mean_k, var_k = predict(X_sub, y_sub, test_X, hp)
        regime_means.append(mean_k)
        regime_vars.append(var_k)
        training_sizes.append(len(y_sub))

    regime_means_arr = np.vstack(regime_means)
    regime_vars_arr = np.vstack(regime_vars)
    P_next = _transition_matrix_for_covariate(result.transition_W, result.transition_b, next_z)
    next_regime_probs = result.gamma[-1] @ P_next
    pred_mean = next_regime_probs @ regime_means_arr
    pred_var = next_regime_probs @ (regime_vars_arr + regime_means_arr ** 2) - pred_mean ** 2

    return {
        "mean": pred_mean.astype(np.float32),
        "var": np.maximum(pred_var, 1e-8).astype(np.float32),
        "next_regime_probs": next_regime_probs.astype(np.float32),
        "regime_means": regime_means_arr.astype(np.float32),
        "regime_vars": regime_vars_arr.astype(np.float32),
        "training_sizes": np.array(training_sizes, dtype=np.int32),
    }
