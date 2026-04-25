"""Quick post-conversion sanity pass over Data/parquet/*.

Prints schema + row count for every parquet, sanity-checks FF factor units,
and renders a VIX time series plot if fred_macro is present.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import polars as pl

OUT = Path("Data/parquet")
DIAG = OUT / "_diagnostics"


def _schema_and_rows() -> None:
    for p in sorted(OUT.glob("*.parquet")):
        lf = pl.scan_parquet(p)
        n = lf.select(pl.len()).collect().item()
        schema = lf.collect_schema()
        print(f"\n== {p.name} ({n:,} rows, {len(schema)} cols) ==")
        for name, dtype in list(schema.items())[:20]:
            print(f"  {name:32s} {dtype}")
        if len(schema) > 20:
            print(f"  ... ({len(schema) - 20} more columns)")


def _ccm_vs_constituents() -> None:
    ccm = OUT / "ccm_merged.parquet"
    cst = OUT / "sp500_constituents.parquet"
    if not (ccm.exists() and cst.exists()):
        return
    ccm_permno = (
        pl.scan_parquet(ccm)
        .filter(pl.col("MthCalDt").is_between(pl.date(2000, 1, 1), pl.date(2024, 12, 31)))
        .select(pl.col("PERMNO").unique())
        .collect()
        .to_series()
    )
    cst_permno = pl.scan_parquet(cst).select(pl.col("permno").unique()).collect().to_series()
    shared = set(ccm_permno.to_list()) & set(cst_permno.to_list())
    print(f"\nccm ∩ sp500_constituents (2000-2024): {len(shared)} permnos (expect ~1500-1800)")


def _sp500_constituent_coverage() -> None:
    cst = OUT / "sp500_constituents.parquet"
    char = OUT / "char_panel_raw.parquet"
    panel = OUT / "panel.pickle"
    if not cst.exists():
        return
    cst_df = pl.read_parquet(cst).with_columns(
        pl.col("permno").cast(pl.Int32).alias("PERMNO"),
        pl.col("start").cast(pl.Date),
        pl.col("ending").cast(pl.Date),
    )
    summary = cst_df.select(
        pl.len().alias("n_rows"),
        pl.col("PERMNO").n_unique().alias("n_permno"),
        pl.col("start").min().alias("date_min"),
        pl.col("ending").max().alias("date_max"),
    ).row(0, named=True)
    print(
        "\nsp500_constituents coverage:"
        f" rows={summary['n_rows']:,} permnos={summary['n_permno']:,}"
        f" date={summary['date_min']}..{summary['date_max']}"
    )

    if char.exists():
        filtered = (
            pl.scan_parquet(char)
            .join(cst_df.lazy().select(["PERMNO", "start", "ending"]), on="PERMNO", how="inner")
            .filter(pl.col("MthCalDt").is_between(pl.col("start"), pl.col("ending")))
            .group_by("MthCalDt")
            .agg(pl.col("PERMNO").n_unique().alias("n_permno"))
            .sort("MthCalDt")
            .collect()
        )
        if filtered.height:
            stats = filtered.select(
                pl.col("n_permno").mean().alias("mean_n"),
                pl.col("n_permno").min().alias("min_n"),
                pl.col("n_permno").max().alias("max_n"),
            ).row(0, named=True)
            print(
                "post-filter char panel counts:"
                f" months={filtered.height:,} mean N_t={stats['mean_n']:.1f}"
                f" min={stats['min_n']} max={stats['max_n']}"
            )

    if panel.exists():
        with open(panel, "rb") as f:
            p = pickle.load(f)
        sizes = np.array([len(x) for x in p["returns"]], dtype=np.int32)
        print(
            "panel.pickle counts:"
            f" months={len(sizes):,} mean N_t={sizes.mean():.1f}"
            f" min={sizes.min()} max={sizes.max()}"
        )


def _ff_units() -> None:
    ff = OUT / "ff_factors.parquet"
    if not ff.exists():
        return
    s = pl.read_parquet(ff).select(
        pl.col("mkt_rf").mean().alias("mean"),
        pl.col("mkt_rf").std().alias("std"),
    ).row(0, named=True)
    print(f"\nff_factors.mkt_rf: mean={s['mean']:.5f} std={s['std']:.5f}")
    if abs(s["mean"]) > 0.1:
        print("  WARNING: mean looks too large — units likely still in percent, not decimal")
    else:
        print("  OK: decimal units confirmed")


def _vix_plot() -> None:
    fred = OUT / "fred_macro.parquet"
    if not fred.exists():
        return
    df = pl.read_parquet(fred)
    if "vix" not in df.columns:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DIAG.mkdir(parents=True, exist_ok=True)
    out = DIAG / "vix.png"
    pdf = df.select(["date", "vix"]).drop_nulls().to_pandas()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pdf["date"], pdf["vix"], lw=1)
    ax.set_title("VIX (monthly, end-of-month)")
    ax.set_ylabel("VIX")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"\nwrote {out}")


def main() -> None:
    _schema_and_rows()
    _ccm_vs_constituents()
    _sp500_constituent_coverage()
    _ff_units()
    _vix_plot()


if __name__ == "__main__":
    main()
