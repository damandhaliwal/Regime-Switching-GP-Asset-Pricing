"""Stage 2b: char_panel_raw -> model-ready ragged panel.

Steps:
  1. Universe filter (S&P 500 point-in-time; identity passthrough if constituents absent).
  2. Excess returns (subtract monthly rf from ff_factors.parquet).
  3. Delisting return composition (CRSP DLRET if available; else -0.30 for perf-related codes).
  4. Cross-sectional rank transform to [-0.5, 0.5], median-impute missing (Gu/Kelly/Xiu).
  5. Drop months with N_t < 30.
  6. Split into ragged lists: {"dates", "returns", "X", "permnos"} and pickle.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import polars as pl

from rsgp.characteristics import CHAR_COLS

OUT = Path("Data/parquet")
MIN_STOCKS_PER_MONTH = 30

log = logging.getLogger("preprocess")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# --------------------------------------------------------------------------- #
# Steps                                                                       #
# --------------------------------------------------------------------------- #

def apply_universe_filter(df: pl.DataFrame) -> pl.DataFrame:
    path = OUT / "sp500_constituents.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not present; real-data panel build requires point-in-time S&P 500 membership"
        )
    cst = pl.read_parquet(path).select(
        pl.col("permno").cast(pl.Int32).alias("PERMNO"),
        pl.col("start").cast(pl.Date),
        pl.col("ending").cast(pl.Date),
    ).drop_nulls(["PERMNO", "start", "ending"])
    joined = (
        df.join(cst, on="PERMNO", how="inner")
        .filter(pl.col("MthCalDt").is_between(pl.col("start"), pl.col("ending")))
        .select(df.columns)
        .unique(subset=["PERMNO", "MthCalDt"], keep="first", maintain_order=True)
    )
    log.info("universe filter: %d -> %d rows after S&P 500 point-in-time join", len(df), len(joined))
    return joined


def compute_excess_returns(df: pl.DataFrame) -> pl.DataFrame:
    ff = pl.read_parquet(OUT / "ff_factors.parquet").select(["date", "rf"]).rename({"date": "MthCalDt"})
    # ff month_end dates may be exact calendar month-end; CCM MthCalDt is trading day-based.
    # Align by year-month.
    df2 = df.with_columns(pl.col("MthCalDt").dt.month_end().alias("_me"))
    ff2 = ff.with_columns(pl.col("MthCalDt").dt.month_end().alias("_me")).drop("MthCalDt")
    joined = df2.join(ff2, on="_me", how="left").drop("_me")
    n_miss = joined["rf"].null_count()
    if n_miss:
        log.warning("excess returns: %d rows without rf match (left as null)", n_miss)
    return joined.with_columns(
        (pl.col("ret") - pl.col("rf")).alias("excess_ret")
    ).drop("rf")


def apply_delisting(df: pl.DataFrame) -> pl.DataFrame:
    path = OUT / "delisting.parquet"
    if not path.exists():
        log.warning("delisting file absent -- no delisting return composition")
        return df
    dl = pl.read_parquet(path).select(["PERMNO", "DLSTDT", "DLSTCD", "DLRET"])
    dl = dl.with_columns(pl.col("DLSTDT").dt.month_end().alias("_me"))
    df2 = df.with_columns(pl.col("MthCalDt").dt.month_end().alias("_me"))
    joined = df2.join(dl, on=["PERMNO", "_me"], how="left").drop("_me", "DLSTDT")

    # Rule: DLRET if present; else -0.30 for perf-related (400-599); else 0.
    dl_return = (
        pl.when(pl.col("DLRET").is_not_null()).then(pl.col("DLRET"))
        .when((pl.col("DLSTCD") >= 400) & (pl.col("DLSTCD") < 600)).then(-0.30)
        .when(pl.col("DLSTCD").is_not_null()).then(0.0)
        .otherwise(None)
    )
    # Compose: if delisting applies, use (1+ret)*(1+dl) - 1; fall back to ret.
    composed = (
        pl.when(dl_return.is_not_null())
        .then((1.0 + pl.col("excess_ret").fill_null(0.0)) * (1.0 + dl_return) - 1.0)
        .otherwise(pl.col("excess_ret"))
    )
    n_applied = joined.filter(dl_return.is_not_null()).height
    log.info("delisting: composed returns on %d (permno, month) rows", n_applied)
    return joined.with_columns(composed.alias("excess_ret")).drop("DLSTCD", "DLRET")


def rank_transform(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Per-month cross-sectional rank -> [-0.5, 0.5]. Nulls skipped in rank, then filled with 0."""
    ranked = df
    for c in cols:
        valid = pl.col(c).is_not_null() & pl.col(c).is_finite()
        n_valid = valid.sum().over("MthCalDt")
        # ordinal rank among valid values (1..n_valid); null elsewhere.
        r = pl.when(valid).then(pl.col(c)).otherwise(None).rank(method="ordinal").over("MthCalDt")
        transformed = pl.when(n_valid > 1).then((r - 1) / (n_valid - 1) - 0.5).otherwise(0.0)
        ranked = ranked.with_columns(transformed.fill_null(0.0).cast(pl.Float32).alias(c))
    return ranked


def split_to_ragged(df: pl.DataFrame) -> dict:
    df = df.sort(["MthCalDt", "PERMNO"])
    dates: list[np.datetime64] = []
    returns: list[np.ndarray] = []
    X: list[np.ndarray] = []
    permnos: list[np.ndarray] = []
    for (date,), g in df.group_by(["MthCalDt"], maintain_order=True):
        if g.height < MIN_STOCKS_PER_MONTH:
            continue
        dates.append(date)
        returns.append(g["excess_ret"].to_numpy().astype(np.float32))
        X.append(g.select(CHAR_COLS).to_numpy().astype(np.float32))
        permnos.append(g["PERMNO"].to_numpy().astype(np.int32))
    return {"dates": dates, "returns": returns, "X": X, "permnos": permnos}


# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #

def build_and_save(out_path: Path = OUT / "panel.pickle") -> None:
    log.info("starting preprocess")
    cst_path = OUT / "sp500_constituents.parquet"
    if not cst_path.exists():
        raise FileNotFoundError(
            f"{cst_path} not present; run scripts/convert_raw_to_parquet.py after adding Data/raw/sp500_constituents.csv"
        )
    df = pl.read_parquet(OUT / "char_panel_raw.parquet")
    log.info("  raw panel: %d rows, %d months", len(df), df["MthCalDt"].n_unique())

    # Pre-rank null fractions (diagnostic).
    log.info("  pre-rank non-null fractions:")
    for c in CHAR_COLS:
        frac = 1.0 - df[c].null_count() / len(df)
        log.info("    %-13s frac_non_null=%.3f", c, frac)

    df = apply_universe_filter(df)
    df = compute_excess_returns(df)
    df = apply_delisting(df)
    df = rank_transform(df, CHAR_COLS)
    df = df.filter(pl.col("excess_ret").is_not_null() & pl.col("excess_ret").is_finite())

    panel = split_to_ragged(df)
    T = len(panel["dates"])
    N = [r.shape[0] for r in panel["returns"]]
    log.info("  panel: T=%d, mean N_t=%.0f, min=%d, max=%d, D=%d",
             T, np.mean(N) if N else 0, min(N) if N else 0, max(N) if N else 0, len(CHAR_COLS))

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(panel, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(out_path)
    log.info("finished preprocess: wrote %s", out_path)


if __name__ == "__main__":
    build_and_save()
