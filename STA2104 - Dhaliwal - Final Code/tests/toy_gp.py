"""Validation gate 1: pooled GP + ARD recovery.

Simulate multiple monthly panels from the same GP hyperparameters and fit them
jointly, matching the real regime-level M-step more closely than a single draw.

Pass: median recovered lengthscales within 20% relative error per dimension;
median recovered σ_n within 10%.

Run: PYTHONPATH=. python -m tests.toy_gp
"""
from __future__ import annotations

import numpy as np

from models.gp import GPHyperparams, fit_gp_hyperparameters, se_ard_kernel


def _sample_gp(rng: np.random.Generator, X: np.ndarray,
               lengthscales: np.ndarray, signal_var: float,
               noise_var: float) -> np.ndarray:
    K = se_ard_kernel(X, X, lengthscales, signal_var)
    N = X.shape[0]
    L = np.linalg.cholesky(K + 1e-6 * np.eye(N))
    f = L @ rng.standard_normal(N)
    eps = rng.normal(0.0, np.sqrt(noise_var), size=N)
    return f + eps


def test_ard_recovery() -> None:
    D = 5
    N = 80
    T = 10
    n_draws = 10
    lengthscales_true = np.array([0.35, 0.60, 1.00, 1.50, 2.50])
    signal_var_true = 1.0
    noise_var_true = 0.01  # σ = 0.1

    rel_errs = []
    noise_rel_errs = []
    for seed in range(n_draws):
        base_rng = np.random.default_rng(seed)
        X = base_rng.normal(0.0, 1.5, size=(N, D))
        data: list[tuple[np.ndarray, np.ndarray]] = []
        for t in range(T):
            rng = np.random.default_rng(seed * 100 + t)
            y = _sample_gp(rng, X, lengthscales_true, signal_var_true, noise_var_true)
            data.append((X, y))

        hp, nll = fit_gp_hyperparameters(
            data=data, D=D, n_restarts=4, seed=1,
            init=GPHyperparams(lengthscales=np.ones(D), signal_var=1.0, noise_var=0.1),
        )
        rel_err_ell = np.abs(hp.lengthscales - lengthscales_true) / lengthscales_true
        noise_rel_err = abs(np.sqrt(hp.noise_var) - np.sqrt(noise_var_true)) / np.sqrt(noise_var_true)
        rel_errs.append(rel_err_ell)
        noise_rel_errs.append(noise_rel_err)
        print(
            f"  draw={seed:02d} ls={np.round(hp.lengthscales, 3)} "
            f"noise={np.sqrt(hp.noise_var):.3f} rel={np.round(rel_err_ell, 3)} "
            f"noise_rel={noise_rel_err:.3f} nll={nll:.2f}"
        )

    median_rel = np.median(np.vstack(rel_errs), axis=0)
    median_noise = float(np.median(noise_rel_errs))
    print(f"  lengthscales true     : {lengthscales_true}")
    print(f"  median rel err per dim: {np.round(median_rel, 3)}")
    print(f"  median noise rel err  : {median_noise:.3f}")

    assert (median_rel < 0.20).all(), f"median lengthscale rel err {median_rel} exceeds 0.20"
    assert median_noise < 0.10, f"median noise rel err {median_noise} exceeds 0.10"


def main() -> None:
    print("toy_gp:")
    test_ard_recovery()
    print("PASS")


if __name__ == "__main__":
    main()
