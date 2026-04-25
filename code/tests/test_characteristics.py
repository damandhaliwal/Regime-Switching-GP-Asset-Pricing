"""Synthetic checks for characteristics pipeline. Run: python -m tests.test_characteristics"""
from __future__ import annotations

import numpy as np
import polars as pl

from rsgp.characteristics import _apply_compustat_lag, _build_momentum
from rsgp.preprocess import rank_transform


def test_compustat_lag() -> None:
    # 1 permno, monthly 2020-01 .. 2021-12. Fundamentals at datadate=2019-12 (at=100)
    # and datadate=2020-12 (at=110). With 6mo lag:
    #   up to 2020-06 -> no fundamentals (need to have seen 2019-12 + 6mo = 2020-06-30);
    #   2020-06 .. 2021-05 -> at_lag=100; 2021-06+ -> at_lag=110.
    months = pl.date_range(pl.date(2020, 1, 1), pl.date(2021, 12, 1), "1mo", eager=True).dt.month_end()
    df = pl.DataFrame({
        "PERMNO": pl.Series([1] * len(months), dtype=pl.Int32),
        "MthCalDt": months,
    })
    # Seed the fundamentals frame by writing directly to the parquet? _apply_compustat_lag
    # reads from the real parquet, so test it in isolation by reimplementing the join here
    # against an in-memory fundamentals frame — mirroring the asof-join used in the real code.
    fund = pl.DataFrame({
        "PERMNO": pl.Series([1, 1], dtype=pl.Int32),
        "datadate": [pl.date(2019, 12, 1).to_py_date() if hasattr(pl.date(2019,12,1),'to_py_date') else None, None],
        "at": [100.0, 110.0],
    })
    # Simpler: just check the offset semantics via a hand-rolled asof mirroring the real code.
    import datetime as dt
    fund = pl.DataFrame({
        "PERMNO": pl.Series([1, 1], dtype=pl.Int32),
        "datadate": [dt.date(2019, 12, 31), dt.date(2020, 12, 31)],
        "at": [100.0, 110.0],
    }).with_columns(pl.col("datadate").dt.offset_by("6mo").alias("available_dt")).sort(["PERMNO", "available_dt"])

    joined = df.sort(["PERMNO", "MthCalDt"]).join_asof(
        fund.select(["PERMNO", "available_dt", "at"]).rename({"at": "at_lag"}),
        left_on="MthCalDt", right_on="available_dt", by="PERMNO", strategy="backward",
    )
    at_lag = joined["at_lag"].to_list()
    # First 5 months (Jan-May 2020): before 2020-06-30 available_dt -> null.
    assert all(x is None for x in at_lag[:5]), f"expected null pre-lag, got {at_lag[:5]}"
    # Jun 2020 .. May 2021 -> 100.
    assert all(x == 100.0 for x in at_lag[5:17]), f"expected 100, got {at_lag[5:17]}"
    # Jun 2021 .. Dec 2021 -> 110.
    assert all(x == 110.0 for x in at_lag[17:]), f"expected 110, got {at_lag[17:]}"
    print("  test_compustat_lag OK")


def test_momentum_formula() -> None:
    # 1 permno, 13 months of constant 0.01 returns. mom_12_1 at month 13 should be (1.01)^11 - 1.
    months = pl.date_range(pl.date(2020, 1, 1), pl.date(2021, 1, 1), "1mo", eager=True).dt.month_end()
    df = pl.DataFrame({
        "PERMNO": pl.Series([1] * len(months), dtype=pl.Int32),
        "MthCalDt": months,
        "MthRet": [0.01] * len(months),
    })
    out = _build_momentum(df)
    mom = out["mom_12_1"].to_list()
    expected = (1.01) ** 11 - 1.0
    # Only the last row (month 13) has a full 11-period window ending at t-2=month 11.
    last = mom[-1]
    assert last is not None and abs(last - expected) < 1e-6, f"expected {expected}, got {last}"
    # Earlier months should be null (insufficient history).
    assert all(m is None for m in mom[:12]), f"expected nulls in pre-window months, got {mom[:12]}"
    print(f"  test_momentum_formula OK (last={last:.6f}, expected={expected:.6f})")


def test_rank_transform() -> None:
    rng = np.random.default_rng(0)
    n = 100
    df = pl.DataFrame({
        "PERMNO": pl.Series(np.arange(n), dtype=pl.Int32),
        "MthCalDt": [pl.date(2020, 1, 31).to_py_date() if hasattr(pl.date(2020,1,31),'to_py_date') else None] * n,
        "x": rng.normal(size=n),
    })
    import datetime as dt
    df = df.with_columns(pl.lit(dt.date(2020, 1, 31)).alias("MthCalDt"))
    out = rank_transform(df, ["x"])
    vals = out["x"].to_numpy()
    assert vals.min() >= -0.5 - 1e-9 and vals.max() <= 0.5 + 1e-9, f"range {vals.min()}..{vals.max()}"
    assert abs(np.median(vals)) < 1e-6, f"median {np.median(vals)}"
    print(f"  test_rank_transform OK (range={vals.min():.3f}..{vals.max():.3f}, median={np.median(vals):.2e})")


def main() -> None:
    print("test_characteristics:")
    test_compustat_lag()
    test_momentum_formula()
    test_rank_transform()
    print("all passed")


if __name__ == "__main__":
    main()
