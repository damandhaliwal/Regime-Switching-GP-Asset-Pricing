"""Baseline model implementations on the aligned rolling window."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from models.gp import GPHyperparams, fit_gp_hyperparameters, predict
from rsgp.data import AlignedPanel


def _stack_rows(X: Sequence[np.ndarray], returns: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return np.vstack(X), np.concatenate(returns)


def _sample_training_subset(
    X: Sequence[np.ndarray],
    returns: Sequence[np.ndarray],
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    X_all, y_all = _stack_rows(X, returns)
    if len(y_all) <= max_points:
        return X_all, y_all
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(y_all), size=max_points, replace=False))
    return X_all[idx], y_all[idx]


def fama_macbeth_predict(
    train_X: Sequence[np.ndarray],
    train_returns: Sequence[np.ndarray],
    test_X: np.ndarray,
) -> np.ndarray:
    """Monthly cross-sectional OLS of returns on (1, X), averaged over months."""
    coefs = []
    for X_t, y_t in zip(train_X, train_returns):
        X_aug = np.column_stack([np.ones(X_t.shape[0]), X_t])
        coef, *_ = np.linalg.lstsq(X_aug, y_t, rcond=None)
        coefs.append(coef)
    coef_bar = np.mean(np.vstack(coefs), axis=0)
    alpha = coef_bar[0]
    beta = coef_bar[1:]
    return (alpha + test_X @ beta).astype(np.float32)


def single_regime_gp_predict(
    train_X: Sequence[np.ndarray],
    train_returns: Sequence[np.ndarray],
    test_X: np.ndarray,
    max_points: int = 800,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    hp, _ = fit_gp_hyperparameters(
        data=list(zip(train_X, train_returns)),
        weights=np.ones(len(train_X)),
        D=test_X.shape[1],
        n_restarts=2,
        init=GPHyperparams(lengthscales=np.ones(test_X.shape[1]), signal_var=1.0, noise_var=0.1),
        seed=seed,
    )
    X_sub, y_sub = _sample_training_subset(train_X, train_returns, max_points=max_points, seed=seed)
    mean, var = predict(X_sub, y_sub, test_X, hp)
    return mean.astype(np.float32), {"hp": hp, "train_points": len(y_sub), "var": var.astype(np.float32)}


def random_forest_predict(
    train_X: Sequence[np.ndarray],
    train_returns: Sequence[np.ndarray],
    test_X: np.ndarray,
    random_state: int = 0,
    n_estimators: int = 500,
    max_depth: int | None = None,
    max_train_rows: int = 10000,
) -> np.ndarray:
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError("scikit-learn is required for the random forest baseline") from e

    X_all, y_all = _stack_rows(train_X, train_returns)
    if len(y_all) > max_train_rows:
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(len(y_all), size=max_train_rows, replace=False))
        X_all = X_all[idx]
        y_all = y_all[idx]
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_all, y_all)
    return rf.predict(test_X).astype(np.float32)


def run_baselines_rolling(
    aligned: AlignedPanel,
    oos_start_idx: int,
    oos_end_idx: int | None = None,
    gp_max_points: int = 800,
    rf_random_state: int = 0,
) -> pd.DataFrame:
    rows = []
    oos_end_idx = len(aligned.dates) if oos_end_idx is None else oos_end_idx
    for t in range(oos_start_idx, oos_end_idx):
        train_X = aligned.X[:t]
        train_returns = aligned.returns[:t]
        test_X = aligned.X[t]
        actual = aligned.returns[t]
        permnos = aligned.permnos[t]
        date = pd.to_datetime(aligned.dates[t])

        fm_pred = fama_macbeth_predict(train_X, train_returns, test_X)
        gp_pred, _ = single_regime_gp_predict(
            train_X, train_returns, test_X, max_points=gp_max_points, seed=t
        )
        rf_pred = random_forest_predict(
            train_X, train_returns, test_X, random_state=rf_random_state + t
        )

        for model, pred in (
            ("Fama-MacBeth", fm_pred),
            ("Single-Regime GP", gp_pred),
            ("Random Forest", rf_pred),
        ):
            rows.extend(
                {
                    "date": date,
                    "permno": int(permno),
                    "model": model,
                    "prediction": float(p_hat),
                    "actual_return": float(y),
                }
                for permno, p_hat, y in zip(permnos, pred, actual)
            )
    return pd.DataFrame(rows)
