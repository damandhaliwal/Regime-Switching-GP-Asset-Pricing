"""Validation gate 3: logistic transition coefficient recovery.

Simulate a 2-state non-homogeneous HMM whose transition probabilities are
logistic in a known macro covariate Z. Pass true-state xi soft counts (one-hot
on the realized transitions) into fit_logistic_weighted and verify the recovered
coefficients are close to the truth.

Pass: coefficient RMSE < 0.1, correct sign on every coefficient.

Run: PYTHONPATH=. python -m tests.toy_transitions
"""
from __future__ import annotations

import numpy as np

from models.transitions import fit_logistic_weighted, log_transition_matrix


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def test_transition_recovery() -> None:
    rng = np.random.default_rng(0)
    T = 2000
    K = 2
    M = 2

    # Ground truth: reference class k=0 has zero params. Free class k=1 gets (W, b).
    W_true = np.zeros((K, K, M))
    b_true = np.zeros((K, K))
    W_true[0, 1, :] = np.array([1.2, -0.8])   # from state 0 -> to state 1, driven by z
    W_true[1, 1, :] = np.array([-1.5, 0.5])   # persistence term
    b_true[0, 1] = -1.0  # baseline prob of leaving state 0 small
    b_true[1, 1] = 2.0   # baseline prob of staying in state 1 large

    Z = rng.normal(0.0, 1.0, size=(T, M))

    # Simulate the state chain under the true logistic transitions.
    states = np.empty(T, dtype=np.int32)
    states[0] = 0
    for t in range(1, T):
        j = states[t - 1]
        logits = W_true[j] @ Z[t] + b_true[j]
        logits[0] = 0.0  # reference
        p = _softmax(logits)
        states[t] = rng.choice(K, p=p)

    # Build xi as one-hot on realized transitions (T-1, K, K).
    xi = np.zeros((T - 1, K, K))
    for t in range(T - 1):
        xi[t, states[t], states[t + 1]] = 1.0

    W_hat, b_hat = fit_logistic_weighted(Z, xi, K=K, M=M, l2=1e-6, n_restarts=2)

    # Compare only the free parameters (col k=0 is structural zero).
    diff_W = (W_hat[:, 1:, :] - W_true[:, 1:, :]).reshape(-1)
    diff_b = (b_hat[:, 1:] - b_true[:, 1:]).reshape(-1)
    diffs = np.concatenate([diff_W, diff_b])
    rmse = float(np.sqrt((diffs ** 2).mean()))

    signs_ok = (np.sign(W_hat[:, 1:, :]) == np.sign(W_true[:, 1:, :])).all() and \
               (np.sign(b_hat[:, 1:]) == np.sign(b_true[:, 1:])).all()

    print(f"  W_true (j=0, k=1): {W_true[0, 1]}  recovered: {np.round(W_hat[0, 1], 3)}")
    print(f"  W_true (j=1, k=1): {W_true[1, 1]}  recovered: {np.round(W_hat[1, 1], 3)}")
    print(f"  b_true[:, 1]     : {b_true[:, 1]}  recovered: {np.round(b_hat[:, 1], 3)}")
    print(f"  RMSE across free coefficients: {rmse:.4f}")

    # Sanity: transition matrix normalizes to 1.
    log_P = log_transition_matrix(W_hat, b_hat, Z)
    assert np.allclose(np.exp(log_P).sum(axis=2), 1.0, atol=1e-8), "rows of P don't sum to 1"

    assert rmse < 0.10, f"RMSE {rmse:.4f} exceeds 0.10"
    assert signs_ok, "coefficient sign recovery failed"


def main() -> None:
    print("toy_transitions:")
    test_transition_recovery()
    print("PASS")


if __name__ == "__main__":
    main()
