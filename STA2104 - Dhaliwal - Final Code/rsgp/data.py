"""Shared loaders for panel and aligned macro data."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

OUT = Path("Data/parquet")


@dataclass
class AlignedPanel:
    dates: list
    returns: list[np.ndarray]
    X: list[np.ndarray]
    permnos: list[np.ndarray]
    Z: np.ndarray
    z_columns: list[str]


def load_panel(panel_path: Path = OUT / "panel.pickle") -> dict:
    with open(panel_path, "rb") as f:
        return pickle.load(f)


def _panel_date_frame(panel: dict) -> pl.DataFrame:
    return pl.DataFrame({
        "idx": pl.Series(np.arange(len(panel["dates"])), dtype=pl.Int32),
        "date": pl.Series(panel["dates"]).cast(pl.Date),
    })


def load_aligned_panel(
    required_macro_cols: Sequence[str] | None = None,
    panel_path: Path = OUT / "panel.pickle",
    z_path: Path = OUT / "Z.parquet",
) -> AlignedPanel:
    panel = load_panel(panel_path)
    z = pl.read_parquet(z_path).sort("date")
    z_columns = [c for c in z.columns if c != "date"]
    if required_macro_cols is None:
        required_macro_cols = z_columns
    else:
        required_macro_cols = list(required_macro_cols)

    missing = [c for c in required_macro_cols if c not in z_columns]
    if missing:
        raise ValueError(f"missing required macro columns in {z_path}: {missing}")

    joined = (
        _panel_date_frame(panel)
        .join(z.select(["date", *required_macro_cols]), on="date", how="inner")
        .drop_nulls(required_macro_cols)
        .sort("date")
    )

    idx = joined["idx"].to_list()
    dates = joined["date"].to_list()
    returns = [panel["returns"][i] for i in idx]
    X = [panel["X"][i] for i in idx]
    permnos = [panel["permnos"][i] for i in idx]
    Z = joined.select(required_macro_cols).to_numpy().astype(np.float32)
    return AlignedPanel(
        dates=dates,
        returns=returns,
        X=X,
        permnos=permnos,
        Z=Z,
        z_columns=required_macro_cols,
    )
