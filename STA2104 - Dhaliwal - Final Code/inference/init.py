"""Deterministic regime initialization from a rolling volatility proxy."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

INDEX_PATH = Path("Data/parquet/sp500_index.parquet")


@dataclass
class RegimeInitialization:
    dates: list
    volatility: np.ndarray
    states: np.ndarray
    gamma: np.ndarray
    centers: np.ndarray


def _kmeans_1d(values: np.ndarray, k: int, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    if k < 1:
        raise ValueError("k must be >= 1")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("values must be 1-D")
    quantiles = np.linspace(0.0, 1.0, k + 2)[1:-1]
    centers = np.quantile(values, quantiles)
    if k == 1:
        centers = np.array([values.mean()], dtype=np.float64)

    for _ in range(max_iter):
        dist = np.abs(values[:, None] - centers[None, :])
        labels = dist.argmin(axis=1)
        new_centers = centers.copy()
        for idx in range(k):
            members = values[labels == idx]
            if members.size:
                new_centers[idx] = members.mean()
        if np.allclose(new_centers, centers, atol=1e-8):
            break
        centers = new_centers

    order = np.argsort(centers)
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[int(label)] for label in labels], dtype=np.int32)
    return labels, centers[order]


def initialize_from_volatility_proxy(
    dates: Sequence,
    volatility: np.ndarray,
    k: int = 2,
) -> RegimeInitialization:
    volatility = np.asarray(volatility, dtype=np.float64)
    states, centers = _kmeans_1d(volatility, k=k)
    gamma = np.eye(k, dtype=np.float64)[states]
    return RegimeInitialization(
        dates=list(dates),
        volatility=volatility,
        states=states,
        gamma=gamma,
        centers=centers,
    )


def _load_sp500_volatility(dates: Sequence, index_path: Path = INDEX_PATH) -> np.ndarray:
    """Rolling 3-month std of the S&P 500 return series, aligned to `dates`.

    `sp500_index.parquet` stores daily `sprtrn`. Aggregate to monthly arithmetic
    return first, then take the 3-month rolling std; joining the *daily* frame
    to monthly dates fans rows out and corrupts the proxy.
    """
    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} not found")
    date_df = (
        pl.DataFrame({"date": dates})
        .with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("date").dt.year().alias("_year"),
            pl.col("date").dt.month().alias("_month"),
        )
        .sort("date")
    )
    monthly = (
        pl.read_parquet(index_path)
        .select(pl.col("caldt").alias("date"), "sprtrn")
        .with_columns(
            pl.col("date").dt.year().alias("_year"),
            pl.col("date").dt.month().alias("_month"),
            (pl.col("sprtrn").cast(pl.Float64) + 1.0).log().alias("_log1p"),
        )
        .group_by(["_year", "_month"])
        .agg(pl.col("_log1p").sum().alias("_log_ret"))
        .sort(["_year", "_month"])
        .with_columns((pl.col("_log_ret").exp() - 1.0).alias("_monthly_ret"))
        .with_columns(
            pl.col("_monthly_ret")
            .rolling_std(window_size=3, min_samples=3)
            .alias("volatility")
        )
        .select(["_year", "_month", "volatility"])
    )
    aligned = (
        date_df.join(monthly, on=["_year", "_month"], how="left")
        .with_columns(pl.col("volatility").fill_null(strategy="forward").fill_null(strategy="backward"))
        .sort("date")
    )
    if aligned["volatility"].null_count():
        raise ValueError("volatility proxy still contains nulls after fill")
    if aligned.height != len(list(dates)):
        raise ValueError(
            f"volatility proxy length {aligned.height} does not match dates length {len(list(dates))}"
        )
    return aligned["volatility"].to_numpy().astype(np.float64)


def initialize_regimes(
    dates: Sequence,
    k: int = 2,
    index_path: Path = INDEX_PATH,
) -> RegimeInitialization:
    volatility = _load_sp500_volatility(dates, index_path=index_path)
    return initialize_from_volatility_proxy(dates=dates, volatility=volatility, k=k)
