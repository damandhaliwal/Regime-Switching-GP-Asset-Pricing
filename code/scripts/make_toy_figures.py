"""Generate Figures 1 and 2 from the proposal using the integrated toy pipeline.

Figure 1: true vs recovered ARD lengthscales across both regimes.
Figure 2: inferred posterior regime probabilities overlaid on the true regime
sequence.

Mirrors the data-generating process in tests/toy_integrated.py but writes
figures rather than running assertions.
"""
from __future__ import annotations

from itertools import permutations
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.em import EMConfig, fit_em
from inference.init import initialize_from_volatility_proxy
from models.gp import GPHyperparams, se_ard_kernel


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _sample_panel(rng: np.random.Generator, X: np.ndarray, hp: GPHyperparams) -> np.ndarray:
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


def _fig1_ard_recovery(hp_true, hp_est, order, out_path: Path) -> None:
    D = len(hp_true[0].lengthscales)
    x = np.arange(D)
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for est_k, true_k in enumerate(order):
        ax = axes[est_k]
        ax.bar(x - width / 2, hp_true[true_k].lengthscales, width=width, label="True", color="#4C72B0")
        ax.bar(x + width / 2, hp_est[est_k].lengthscales, width=width, label="Estimated", color="#DD8452")
        ax.set_title(f"Regime {est_k}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"x{j + 1}" for j in range(D)])
        ax.set_xlabel("Characteristic")
        if est_k == 0:
            ax.set_ylabel("ARD lengthscale")
        ax.legend()
    fig.suptitle("Toy data: true vs recovered ARD lengthscales")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _fig2_regime_recovery(true_states: np.ndarray, gamma: np.ndarray, order: np.ndarray,
                          viterbi_acc: float, out_path: Path) -> None:
    T = len(true_states)
    t_axis = np.arange(T)
    # Align gamma columns so that column `true_k` corresponds to the true label `true_k`.
    # _align_labels gives a mapping pred -> true, so for column indexing we need the
    # inverse permutation.
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    prob_regime1_true_aligned = gamma[:, inverse[1]]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.step(t_axis, true_states, where="post", color="black", linewidth=1.2, label="True regime")
    ax.plot(t_axis, prob_regime1_true_aligned, color="#DD8452", linewidth=1.6,
            label=r"Inferred $P(s_t = 1 \mid \mathrm{data})$")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(0, T - 1)
    ax.set_xlabel("Month")
    ax.set_ylabel("Regime / probability")
    ax.set_title(f"Toy data: inferred regime probability vs true sequence  "
                 f"(Viterbi accuracy = {viterbi_acc:.2%})")
    ax.legend(loc="center left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    out_dir = ROOT / "output" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

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
            max_months_per_regime_fit=None,
            max_monthly_points=None,
        ),
        init=init,
        z_columns=["z1", "z2"],
    )

    order = _align_labels(states, result.viterbi_path, k=K)
    aligned_path = np.array([order[s] for s in result.viterbi_path])
    viterbi_acc = float((aligned_path == states).mean())

    fig1_path = out_dir / "toy_ard_recovery.png"
    fig2_path = out_dir / "toy_regime_recovery.png"
    _fig1_ard_recovery(hp_true, result.gp_params, order, fig1_path)
    _fig2_regime_recovery(states, result.gamma, order, viterbi_acc, fig2_path)

    print(f"viterbi accuracy: {viterbi_acc:.3f}")
    print(f"wrote {fig1_path}")
    print(f"wrote {fig2_path}")


if __name__ == "__main__":
    main()
