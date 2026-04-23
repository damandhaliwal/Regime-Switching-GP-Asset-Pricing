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
    index_df = (
        pl.read_parquet(index_path)
        .select(pl.col("caldt").alias("date"), "sprtrn")
        .sort("date")
        .with_columns(
            pl.col("sprtrn").rolling_std(window_size=3, min_samples=3).alias("volatility"),
            pl.col("date").dt.year().alias("_year"),
            pl.col("date").dt.month().alias("_month"),
        )
        .select(["_year", "_month", "volatility"])
    )
    aligned = (
        date_df.join(index_df, on=["_year", "_month"], how="left")
        .with_columns(pl.col("volatility").fill_null(strategy="forward").fill_null(strategy="backward"))
        .sort("date")
    )
    if aligned["volatility"].null_count():
        raise ValueError("volatility proxy still contains nulls after fill")
    return aligned["volatility"].to_numpy().astype(np.float64)


def initialize_regimes(
    dates: Sequence,
    k: int = 2,
    index_path: Path = INDEX_PATH,
) -> RegimeInitialization:
    volatility = _load_sp500_volatility(dates, index_path=index_path)
    return initialize_from_volatility_proxy(dates=dates, volatility=volatility, k=k)
