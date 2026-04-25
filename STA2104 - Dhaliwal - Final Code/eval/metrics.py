"""Evaluation metrics for out-of-sample predictions and portfolios."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

FF_PATH = Path("Data/parquet/ff_factors.parquet")


def pooled_oos_r2(actual: list[np.ndarray], predicted: list[np.ndarray]) -> float:
    y = np.concatenate([np.asarray(x, dtype=np.float64) for x in actual])
    yhat = np.concatenate([np.asarray(x, dtype=np.float64) for x in predicted])
    denom = float(np.sum(y ** 2))
    if denom <= 0:
        return float("nan")
    return 1.0 - float(np.sum((y - yhat) ** 2)) / denom


def regime_conditional_oos_r2(
    actual: list[np.ndarray],
    predicted: list[np.ndarray],
    states: np.ndarray,
) -> dict[int, float]:
    out: dict[int, float] = {}
    for state in np.unique(states):
        idx = np.flatnonzero(states == state)
        out[int(state)] = pooled_oos_r2([actual[i] for i in idx], [predicted[i] for i in idx])
    return out


def sharpe_ratio(returns: np.ndarray, annualize: bool = True) -> float:
    returns = np.asarray(returns, dtype=np.float64)
    if returns.size < 2:
        return float("nan")
    mean = returns.mean()
    std = returns.std(ddof=1)
    if std <= 0:
        return float("nan")
    sharpe = mean / std
    return float(np.sqrt(12.0) * sharpe if annualize else sharpe)


def annualized_mean(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=np.float64)
    return float(12.0 * returns.mean()) if returns.size else float("nan")


def annualized_volatility(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=np.float64)
    return float(np.sqrt(12.0) * returns.std(ddof=1)) if returns.size > 1 else float("nan")


def max_drawdown(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=np.float64)
    if not returns.size:
        return float("nan")
    wealth = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(wealth)
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min())


def _factor_frame(ff_path: Path = FF_PATH) -> pd.DataFrame:
    cols = ["date", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    frame = pl.read_parquet(ff_path).select(cols).to_pandas()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    return frame


def _merge_monthly_factors(portfolio_df: pd.DataFrame, ff_path: Path = FF_PATH) -> pd.DataFrame:
    df = portfolio_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df.merge(_factor_frame(ff_path), on=["year", "month"], how="left", suffixes=("", "_ff"))


def _alpha_from_regression(
    portfolio_df: pd.DataFrame,
    factor_cols: list[str],
    ff_path: Path = FF_PATH,
) -> float:
    merged = _merge_monthly_factors(portfolio_df, ff_path=ff_path).dropna(subset=["long_short", *factor_cols])
    if merged.shape[0] < len(factor_cols) + 3:
        return float("nan")
    y = merged["long_short"].to_numpy(dtype=np.float64)
    X = merged[factor_cols].to_numpy(dtype=np.float64)
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(12.0 * beta[0])


def capm_alpha(portfolio_df: pd.DataFrame, ff_path: Path = FF_PATH) -> float:
    return _alpha_from_regression(portfolio_df, ["mkt_rf"], ff_path=ff_path)


def ff5_alpha(portfolio_df: pd.DataFrame, ff_path: Path = FF_PATH) -> float:
    return _alpha_from_regression(portfolio_df, ["mkt_rf", "smb", "hml", "rmw", "cma"], ff_path=ff_path)
