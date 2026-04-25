"""Validation gate 2: HMM regime recovery with fixed Gaussian emissions.

Simulate a 2-state HMM with well-separated Gaussian emissions and a homogeneous
transition matrix, pass the true parameters into the forward-backward + Viterbi
routines, and check that Viterbi recovers the regime sequence.

Pass: accuracy > 0.90 after label alignment.

Run: PYTHONPATH=. python -m tests.toy_hmm
"""
from __future__ import annotations

import numpy as np

from models.hmm import posteriors, viterbi


def _simulate_hmm(rng: np.random.Generator, T: int, pi: np.ndarray,
                  A: np.ndarray, mus: np.ndarray, sigmas: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    K = pi.shape[0]
    states = np.empty(T, dtype=np.int32)
    obs = np.empty(T)
    states[0] = rng.choice(K, p=pi)
    obs[0] = rng.normal(mus[states[0]], sigmas[states[0]])
    for t in range(1, T):
        states[t] = rng.choice(K, p=A[states[t - 1]])
        obs[t] = rng.normal(mus[states[t]], sigmas[states[t]])
    return states, obs


def _gaussian_log_emit(obs: np.ndarray, mus: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    T = obs.shape[0]
    K = mus.shape[0]
    diff = obs[:, None] - mus[None, :]
    var = sigmas[None, :] ** 2
    return -0.5 * np.log(2 * np.pi * var) - 0.5 * diff * diff / var


def _align(true_states: np.ndarray, pred: np.ndarray, K: int) -> np.ndarray:
    """Greedy label permutation maximizing accuracy (good enough for K=2, 3)."""
    from itertools import permutations
    best_perm, best_acc = tuple(range(K)), -1.0
    for perm in permutations(range(K)):
        mapped = np.array([perm[s] for s in pred])
        acc = (mapped == true_states).mean()
        if acc > best_acc:
            best_acc = acc
            best_perm = perm
    return np.array([best_perm[s] for s in pred])


def test_hmm_recovery() -> None:
    rng = np.random.default_rng(0)
    T = 300
    K = 2
    pi = np.array([0.5, 0.5])
    A = np.array([[0.95, 0.05],
                  [0.10, 0.90]])
    mus = np.array([0.0, 2.0])
    sigmas = np.array([0.5, 0.5])

    true_states, obs = _simulate_hmm(rng, T, pi, A, mus, sigmas)

    log_pi = np.log(pi)
    log_trans = np.broadcast_to(np.log(A), (T - 1, K, K)).copy()
    log_emit = _gaussian_log_emit(obs, mus, sigmas)

    gamma, xi, log_lik = posteriors(log_pi, log_trans, log_emit)
    path = viterbi(log_pi, log_trans, log_emit)
    aligned = _align(true_states, path, K)
    acc = (aligned == true_states).mean()

    # Responsibility sanity checks.
    assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-8), "gamma rows don't sum to 1"
    assert np.allclose(xi.sum(axis=(1, 2)), 1.0, atol=1e-8), "xi slices don't sum to 1"

    print(f"  T={T}, K={K}, Viterbi accuracy={acc:.3f} (log_lik={log_lik:.2f})")
    assert acc > 0.90, f"accuracy {acc} below 0.90 threshold"


def main() -> None:
    print("toy_hmm:")
    test_hmm_recovery()
    print("PASS")


if __name__ == "__main__":
    main()
