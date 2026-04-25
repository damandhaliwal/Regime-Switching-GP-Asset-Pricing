"""Stage 2a: build the raw characteristic panel from typed parquets.

Inputs:
  Data/parquet/ccm_merged.parquet         -- CRSP monthly x Compustat annual
  Data/parquet/crsp_daily.parquet         -- CRSP daily (for idio vol)
  Data/parquet/ff_factors_daily.parquet   -- daily FF factors (for idio vol)

Output:
  Data/parquet/char_panel_raw.parquet     -- long (permno, date, ret, <10 chars>)

No rank transform here; raw numeric values only. Cross-sectional ranking +
excess returns + delisting handling live in data/preprocess.py.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

OUT = Path("Data/parquet")

log = logging.getLogger("characteristics")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

START = pl.date(2000, 1, 1)
END = pl.date(2024, 12, 31)

CHAR_COLS = [
    "log_mktcap", "bm", "ep", "mom_12_1", "mom_1",
    "roe", "roa", "asset_growth", "turnover", "idio_vol",
]


# --------------------------------------------------------------------------- #
# Monthly base + Compustat lag                                                #
# --------------------------------------------------------------------------- #

def load_monthly() -> pl.DataFrame:
    cols = [
        "PERMNO", "MthCalDt", "MthRet", "MthRetx", "MthCap", "MthPrc",
        "MthVol", "ShrOut", "gvkey", "datadate",
        "at", "ceq", "ni", "ib",
    ]
    df = (
        pl.scan_parquet(OUT / "ccm_merged.parquet")
        .select(cols)
        .filter(pl.col("MthCalDt").is_between(START, END))
        .filter(pl.col("MthCap").is_not_null() & pl.col("MthRet").is_not_null())
        .collect()
    )
    # One row per (PERMNO, MthCalDt): CCM can have multiple gvkey links per permno-month.
    # Keep the row with the freshest non-null datadate (most recent fundamentals).
    df = (
        df.sort(["PERMNO", "MthCalDt", "datadate"], nulls_last=True)
        .group_by(["PERMNO", "MthCalDt"], maintain_order=True)
        .last()
    )
    return df


def _apply_compustat_lag(df: pl.DataFrame) -> pl.DataFrame:
    """Asof-join the freshest fundamentals known at least 6 months prior.

    Emits *_lag (datadate + 6mo <= MthCalDt) and at_lag_prior (datadate + 18mo).
    """
    fund = (
        pl.scan_parquet(OUT / "ccm_merged.parquet")
        .select(["PERMNO", "datadate", "at", "ceq", "ni", "ib"])
        .filter(pl.col("datadate").is_not_null() & pl.col("at").is_not_null())
        .unique(subset=["PERMNO", "datadate"])
        .collect()
        .sort(["PERMNO", "datadate"])
    )
    # Asof-key: datadate shifted forward 6mo, so values are only matched once 6mo have passed.
    fund_lag = fund.with_columns(
        (pl.col("datadate").dt.offset_by("6mo")).alias("available_dt"),
    ).rename({"at": "at_lag", "ceq": "ceq_lag", "ni": "ni_lag", "ib": "ib_lag"})
    fund_prior = fund.with_columns(
        (pl.col("datadate").dt.offset_by("18mo")).alias("available_dt"),
    ).select(["PERMNO", "available_dt", "at"]).rename({"at": "at_lag_prior"})

    df = df.sort(["PERMNO", "MthCalDt"])

    df = df.join_asof(
        fund_lag.sort(["PERMNO", "available_dt"]).select(
            ["PERMNO", "available_dt", "at_lag", "ceq_lag", "ni_lag", "ib_lag"]
        ),
        left_on="MthCalDt", right_on="available_dt",
        by="PERMNO", strategy="backward",
    )
    df = df.join_asof(
        fund_prior.sort(["PERMNO", "available_dt"]),
        left_on="MthCalDt", right_on="available_dt",
        by="PERMNO", strategy="backward",
    )
    return df


# --------------------------------------------------------------------------- #
# Per-characteristic builders                                                 #
# --------------------------------------------------------------------------- #

def _build_size_value_profitability(df: pl.DataFrame) -> pl.DataFrame:
    # CRSP MthCap is in $ thousands; Compustat at/ceq/ni/ib are in $ millions.
    # Convert MthCap to $M so book/market ratios are dimensionally correct.
    mkt_M = pl.col("MthCap") / 1000.0
    return df.with_columns(
        pl.col("MthCap").log().alias("log_mktcap"),
        (pl.col("ceq_lag") / mkt_M).alias("bm"),
        (pl.col("ib_lag") / mkt_M).alias("ep"),
        (pl.col("ni_lag") / pl.col("ceq_lag")).alias("roe"),
        (pl.col("ni_lag") / pl.col("at_lag")).alias("roa"),
        (pl.col("at_lag") / pl.col("at_lag_prior") - 1.0).alias("asset_growth"),
    )


def _build_momentum(df: pl.DataFrame) -> pl.DataFrame:
    """mom_12_1: prod(1+r)[t-12..t-2] - 1. mom_1: r[t-1]."""
    df = df.sort(["PERMNO", "MthCalDt"])
    log_one_plus = (1.0 + pl.col("MthRet")).fill_null(1.0).log()
    return df.with_columns(
        # 11-period sum of log(1+r) lagged by 2 -> product over months t-12..t-2.
        (
            log_one_plus.shift(2).rolling_sum(window_size=11).over("PERMNO").exp() - 1.0
        ).alias("mom_12_1"),
        pl.col("MthRet").shift(1).over("PERMNO").alias("mom_1"),
    )


def _build_turnover(df: pl.DataFrame) -> pl.DataFrame:
    # ShrOut is in thousands of shares; MthVol is raw shares. Scale to fraction.
    return df.with_columns(
        pl.when(pl.col("ShrOut") > 0)
        .then(pl.col("MthVol") / (pl.col("ShrOut") * 1000.0))
        .otherwise(None)
        .alias("turnover")
    )


# --------------------------------------------------------------------------- #
# Idiosyncratic volatility (daily)                                            #
# --------------------------------------------------------------------------- #

def _idio_vol_for_window(r_excess: np.ndarray, mkt: np.ndarray) -> float:
    """OLS residual stdev of r_excess = a + b*mkt + eps, scaled to monthly (*sqrt(21))."""
    if r_excess.size < 30:
        return np.nan
    X = np.column_stack([np.ones_like(mkt), mkt])
    beta, *_ = np.linalg.lstsq(X, r_excess, rcond=None)
    resid = r_excess - X @ beta
    return float(np.std(resid, ddof=2) * np.sqrt(21.0))


def _build_idio_vol_from_daily() -> pl.DataFrame:
    """60-trading-day residual stdev ending on each month's last trading day.

    Returns a frame (PERMNO, MthCalDt, idio_vol) with MthCalDt = calendar month-end
    matching the CCM convention (so downstream joins line up).
    """
    log.info("  loading daily returns + FF factors for idio_vol")
    daily = (
        pl.scan_parquet(OUT / "crsp_daily.parquet")
        .select(["PERMNO", "date", "RET"])
        .filter(pl.col("date").is_between(pl.date(1999, 1, 1), END))
        .filter(pl.col("RET").is_not_null())
        .collect()
    )
    ff = (
        pl.read_parquet(OUT / "ff_factors_daily.parquet")
        .select(["date", "mkt_rf", "rf"])
    )
    d = daily.join(ff, on="date", how="inner").with_columns(
        (pl.col("RET") - pl.col("rf")).alias("r_ex"),
        pl.col("date").dt.month_end().alias("month_end"),
    ).sort(["PERMNO", "date"])

    log.info("  computing rolling residual stdev over ~%d rows", len(d))

    permnos = d["PERMNO"].to_numpy()
    dates = d["date"].to_numpy()
    r_ex = d["r_ex"].to_numpy().astype(np.float64)
    mkt = d["mkt_rf"].to_numpy().astype(np.float64)
    month_end = d["month_end"].to_numpy()

    # For each (permno, month_end) take the last 60 trading rows ending on or before month_end.
    WIN = 60
    out_permno: list[int] = []
    out_month: list[np.datetime64] = []
    out_iv: list[float] = []

    # Walk per-permno blocks.
    n = len(d)
    start = 0
    while start < n:
        end = start
        while end < n and permnos[end] == permnos[start]:
            end += 1
        pn = permnos[start]
        block_r = r_ex[start:end]
        block_m = mkt[start:end]
        block_me = month_end[start:end]
        # Take last day of each calendar month.
        changes = np.concatenate(([True], block_me[1:] != block_me[:-1]))
        last_idx = np.flatnonzero(np.concatenate((changes[1:], [True])))
        for li in last_idx:
            lo = max(0, li + 1 - WIN)
            if li + 1 - lo < 30:
                continue
            iv = _idio_vol_for_window(block_r[lo:li + 1], block_m[lo:li + 1])
            if np.isfinite(iv):
                out_permno.append(int(pn))
                out_month.append(block_me[li])
                out_iv.append(iv)
        start = end

    log.info("  idio_vol: %d (permno, month) rows", len(out_permno))
    months_np = np.array(out_month, dtype="datetime64[D]")
    return pl.DataFrame({
        "PERMNO": pl.Series(out_permno, dtype=pl.Int32),
        "MthCalDt": pl.Series(months_np.astype("datetime64[ms]")).cast(pl.Date),
        "idio_vol": pl.Series(out_iv, dtype=pl.Float32),
    })


# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #

def _log_char_stats(df: pl.DataFrame) -> None:
    n = len(df)
    for c in CHAR_COLS:
        s = df[c]
        n_null = s.null_count()
        frac_null = n_null / n if n else 0.0
        finite = s.drop_nulls().to_numpy()
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            log.info("    %-13s frac_null=%.3f (all null/non-finite)", c, frac_null)
            continue
        q = np.quantile(finite, [0.01, 0.5, 0.99])
        log.info("    %-13s frac_null=%.3f mean=%.4f p01=%.4f p50=%.4f p99=%.4f",
                 c, frac_null, float(finite.mean()), q[0], q[1], q[2])
        if frac_null > 0.40:
            log.warning("      %s has >40%% nulls — check fundamentals lag", c)


def build_char_panel(out_path: Path = OUT / "char_panel_raw.parquet") -> None:
    log.info("starting char_panel_raw")
    df = load_monthly()
    log.info("  loaded %d monthly rows (%d permnos)", len(df), df["PERMNO"].n_unique())

    df = _apply_compustat_lag(df)
    df = _build_size_value_profitability(df)
    df = _build_momentum(df)
    df = _build_turnover(df)

    iv = _build_idio_vol_from_daily()
    # Join on (PERMNO, year-month) since CCM MthCalDt is last trading day (not calendar month-end).
    df = df.with_columns(pl.col("MthCalDt").dt.month_end().alias("_me"))
    iv2 = iv.with_columns(pl.col("MthCalDt").dt.month_end().alias("_me")).drop("MthCalDt")
    df = df.join(iv2, on=["PERMNO", "_me"], how="left").drop("_me")

    out = df.select(
        pl.col("PERMNO"),
        pl.col("MthCalDt"),
        pl.col("MthRet").alias("ret"),
        *[pl.col(c).cast(pl.Float32) for c in CHAR_COLS],
    ).sort(["MthCalDt", "PERMNO"])

    _log_char_stats(out)

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    out.write_parquet(tmp, compression="snappy")
    tmp.replace(out_path)
    log.info("finished char_panel_raw: wrote %d rows to %s", len(out), out_path)


if __name__ == "__main__":
    build_char_panel()
