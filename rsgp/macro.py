"""Stage 2c: FRED macro parquet -> Z matrix for non-homogeneous transitions.

Dormant until fred_macro.parquet exists. Produces (T, M) macro covariates,
all lagged 1 month (month-t transition prior uses month-(t-1) macro state).
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from rsgp.data import load_panel

OUT = Path("Data/parquet")
MACRO_COLS = ["vix", "vix_chg", "credit_spread", "term_spread"]

log = logging.getLogger("macro")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def build_macro_panel(
    out_path: Path = OUT / "Z.parquet",
    panel_path: Path = OUT / "panel.pickle",
) -> None:
    src = OUT / "fred_macro.parquet"
    if not src.exists():
        log.warning("fred_macro.parquet not found; Z not built")
        return
    if not panel_path.exists():
        log.warning("panel.pickle not found; Z not built")
        return
    panel = load_panel(panel_path)
    panel_dates = (
        pl.DataFrame({"date": panel["dates"]})
        .with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("date").dt.year().alias("_year"),
            pl.col("date").dt.month().alias("_month"),
        )
        .sort("date")
    )
    df = pl.read_parquet(src).sort("date")
    missing = [c for c in MACRO_COLS if c not in df.columns]
    if missing:
        log.warning("macro: columns missing from fred_macro: %s", missing)
    keep = [c for c in MACRO_COLS if c in df.columns]
    lagged = (
        df.with_columns(
            pl.col("date").dt.year().alias("_year"),
            pl.col("date").dt.month().alias("_month"),
            *[pl.col(c).shift(1).alias(f"{c}_lag1") for c in keep],
        )
        .select(["_year", "_month", *[f"{c}_lag1" for c in keep]])
    )
    aligned = (
        panel_dates.join(lagged, on=["_year", "_month"], how="left")
        .select(["date", *[f"{c}_lag1" for c in keep]])
        .sort("date")
    )

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    aligned.write_parquet(tmp, compression="snappy")
    tmp.replace(out_path)
    fully_observed = aligned.drop_nulls().height
    if fully_observed:
        first_full = aligned.drop_nulls().select(pl.col("date").min()).item()
    else:
        first_full = None
    log.info(
        "finished macro: wrote %d panel-aligned rows, M=%d to %s (first fully observed date=%s)",
        len(aligned), len(keep), out_path, first_full,
    )


if __name__ == "__main__":
    build_macro_panel()
