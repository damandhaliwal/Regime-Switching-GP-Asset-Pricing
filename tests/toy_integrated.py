"""Validation gate 4: integrated synthetic pipeline.

Run: PYTHONPATH=. python -m tests.toy_integrated
"""
from __future__ import annotations

from itertools import permutations

import numpy as np

from inference.em import EMConfig, fit_em
from inference.init import initialize_from_volatility_proxy
from models.gp import GPHyperparams, se_ard_kernel
from models.transitions import log_transition_matrix


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _sample_panel(
    rng: np.random.Generator,
    X: np.ndarray,
    hp: GPHyperparams,
) -> np.ndarray:
    K = se_ard_kernel(X, X, hp.lengthscales, hp.signal_var)
    L = np.linalg.cholesky(K + 1e-6 * np.eye(X.shape[0]))
    return L @ rng.standard_normal(X.shape[0]) + rng.normal(0.0, np.sqrt(hp.noise_var), size=X.shape[0])


def _align_labels(true_states: np.ndarray, pred_states: np.ndarray, k: int) -> np.ndarray:
    best = np.arange(k)
    best_acc = -1.0
    for perm in permutations(range(k)):
        mapped = np.array([perm[s] for s in pred_states])
        acc = float((mapped == true_states).mean())
        if acc > best_acc:
            best_acc = acc
            best = np.array(perm)
    return best


def test_integrated_pipeline() -> None:
    rng = np.random.default_rng(0)
    T = 100
    N = 100
    D = 5
    K = 2
    M = 2

    hp_true = [
        GPHyperparams(lengthscales=np.array([0.35, 0.7, 1.0, 1.6, 2.2]), signal_var=1.0, noise_var=0.01),
        GPHyperparams(lengthscales=np.array([1.2, 0.45, 0.8, 2.1, 0.55]), signal_var=1.0, noise_var=0.01),
    ]
    W_true = np.zeros((K, K, M))
    b_true = np.zeros((K, K))
    W_true[0, 1, :] = np.array([1.8, -0.4])
    W_true[1, 1, :] = np.array([-1.0, 0.0])
    b_true[0, 1] = -1.4
    b_true[1, 1] = 2.2
    pi_true = np.array([0.85, 0.15])

    Z = rng.normal(0.0, 1.0, size=(T, M))
    states = np.empty(T, dtype=np.int32)
    states[0] = rng.choice(K, p=pi_true)
    for t in range(1, T):
        logits = W_true[states[t - 1]] @ Z[t] + b_true[states[t - 1]]
        logits[0] = 0.0
        states[t] = rng.choice(K, p=_softmax(logits))

    X = []
    returns = []
    for t in range(T):
        X_t = rng.normal(0.0, 1.5, size=(N, D))
        y_t = _sample_panel(rng, X_t, hp_true[states[t]])
        X.append(X_t.astype(np.float32))
        returns.append(y_t.astype(np.float32))

    vol_proxy = 0.12 + 0.18 * states + rng.normal(0.0, 0.01, size=T)
    init = initialize_from_volatility_proxy(dates=list(range(T)), volatility=vol_proxy, k=K)
    result = fit_em(
        dates=list(range(T)),
        returns=returns,
        X=X,
        Z=Z,
        config=EMConfig(
            K=K,
            max_iter=25,
            tol=1e-4,
            patience=3,
            gp_restarts=2,
            transition_restarts=4,
            transition_l2=1e-6,
            gp_seed=0,
            transition_seed=0,
            min_regime_mass=5.0,
            max_prediction_points=400,
        ),
        init=init,
        z_columns=["z1", "z2"],
    )

    ll_hist = np.array(result.log_likelihood_history)
    assert np.all(np.diff(ll_hist) >= -1e-8), f"log-likelihood not monotone: {ll_hist}"

    order = _align_labels(states, result.viterbi_path, k=K)
    aligned_path = np.array([order[s] for s in result.viterbi_path])
    acc = float((aligned_path == states).mean())

    gp_rel = []
    for est_k, true_k in enumerate(order):
        rel = np.abs(result.gp_params[est_k].lengthscales - hp_true[true_k].lengthscales) / hp_true[true_k].lengthscales
        gp_rel.append(rel)
        print(
            f"  regime {est_k}: true_ls={np.round(hp_true[true_k].lengthscales, 3)} "
            f"est_ls={np.round(result.gp_params[est_k].lengthscales, 3)} rel={np.round(rel, 3)}"
        )
    gp_rel = np.vstack(gp_rel)
    gp_median = np.median(gp_rel, axis=1)

    W_perm = result.transition_W[order][:, order, :]
    b_perm = result.transition_b[order][:, order]
    P_true = np.exp(log_transition_matrix(W_true, b_true, Z))
    P_est = np.exp(log_transition_matrix(W_perm, b_perm, Z))
    trans_prob_rmse = float(np.sqrt(np.mean((P_est - P_true) ** 2)))

    print(f"  log-likelihood history: {np.round(ll_hist, 2)}")
    print(f"  Viterbi accuracy       : {acc:.3f}")
    print(f"  GP median rel err      : {np.round(gp_median, 3)}")
    print(f"  Transition Prob RMSE   : {trans_prob_rmse:.3f}")

    assert acc > 0.85, f"regime recovery {acc:.3f} below 0.85"
    assert (gp_median < 0.30).all(), f"GP median rel err {gp_median} exceeds 0.30"
    assert trans_prob_rmse < 0.10, f"transition probability RMSE {trans_prob_rmse:.3f} exceeds 0.10"


def main() -> None:
    print("toy_integrated:")
    test_integrated_pipeline()
    print("PASS")


if __name__ == "__main__":
    main()
