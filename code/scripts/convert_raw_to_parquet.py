"""Convert raw WRDS / French / FRED downloads to parquet.

Run once after manual downloads. Idempotent — rerunning overwrites atomically.
Skips cleanly when an expected raw file is absent.

Layer is deliberately dumb: load raw -> fix dtypes -> sanity log -> write parquet.
No merges, no lagging, no winsorization, no characteristic construction. Those
live in data/preprocess.py.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path

import pandas as pd
import polars as pl

RAW = Path("Data/raw")
OUT = Path("Data/parquet")
OUT.mkdir(parents=True, exist_ok=True)
PANEL_START = dt.date(2000, 1, 1)
PANEL_END = dt.date(2024, 12, 31)

RAW_FILES = {
    "ccm_merged":    "eitujyjpb40cbqxj.csv",
    "crsp_daily":    "nm9as3ldy4ps9kjj.csv",
    "delisting":     "vld7qrxmhgwq35gp.csv",
    "sp500_index":   "yr4lsuqjkrbm0bpw.csv",
    "ff3_daily":     "F-F_Research_Data_Factors_daily.csv",
    "ff5_daily":     "F-F_Research_Data_5_Factors_2x3_daily.csv",
    "mom_daily":     "F-F_Momentum_Factor_daily.csv",
    "crsp_monthly":  "crsp_monthly.csv",
    "sp500_consts":  "sp500_constituents.csv",
    "fred_macro":    "fred_macro.csv",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("convert_raw")


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _raw_path(key: str) -> Path:
    p = RAW / RAW_FILES[key]
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def _atomic_sink(lf: pl.LazyFrame, out: Path) -> None:
    tmp = out.with_suffix(out.suffix + ".tmp")
    lf.sink_parquet(tmp, compression="snappy")
    tmp.replace(out)


def _atomic_write(df: pl.DataFrame, out: Path) -> None:
    tmp = out.with_suffix(out.suffix + ".tmp")
    df.write_parquet(tmp, compression="snappy")
    tmp.replace(out)


# Pattern-based dtype coercion for wide CCM file (1045 columns).
_DATE_COLS_CCM = {
    "MthCalDt", "MthPrcDt", "MthPrevDt",
    "SecInfoStartDt", "SecInfoEndDt", "SecurityBegDt", "SecurityEndDt",
    "ULINKDT", "ULINKENDDT", "dldte", "ipodate", "datadate",
    "apdedate", "fdate", "pdate",
}
_INT32_COLS_CCM = {
    "PERMNO", "PERMCO", "SICCD", "NASDIssuno", "YYYYMM", "HdrSICCD",
    "UGVKEY", "APERMNO", "ULINKID", "UPERMNO", "UPERMCO", "USEDFLAG",
    "gvkey", "fyear", "fyr", "fyrc", "cik", "sic",
    "spcindcd", "spcseccd", "stko",
}
_INT64_COLS_CCM = {"ShrOut", "MthFloatShrQty", "MthVol"}
_STRING_COLS_CCM = {
    # CRSP identifier / descriptor fields
    "HdrCUSIP", "CUSIP", "Ticker", "IssuerNm", "USIncFlg", "IssuerType",
    "SecurityType", "SecuritySubType", "ShareType", "ExchangeTier",
    "PrimaryExch", "TradingStatusFlg", "ConditionalType", "ShareClass",
    "SecurityActiveFlg", "TradingSymbol", "NAICS", "HdrPrimaryExch",
    "MthCompFlg", "MthCompSubFlg", "MthPrcFlg", "MthDtFlg", "MthDelFlg",
    "MthPrevPrcFlg", "MthPrevDtFlg", "MthRetFlg", "MthVolFlg",
    "MthFacShrFlg", "MthDisCnt", "MthPrcVolMissCnt",
    "ULINKPRIM", "UIID", "ULINKTYPE",
    # Compustat company / address fields
    "conm", "conml", "add1", "add2", "add3", "add4", "addzip", "busdesc",
    "city", "costat", "county", "dlrsn", "ein", "fax", "fic",
    "ggroup", "gind", "gsector", "gsubind", "idbflag", "incorp", "loc",
    "phone", "prican", "prirow", "priusa", "spcsrc", "state", "weburl",
    # Compustat fiscal / accounting flags
    "tic", "indfmt", "consol", "popsrc", "datafmt", "acctchg", "acctstd",
    "acqmeth", "adrr", "compst", "curcd", "curncd", "currtr", "curuscn",
    "final", "ismod", "ogm", "scf", "src", "stalt", "udpl", "upd",
    "bspr", "ltcm",
}


def _date_parse_expr(col: str) -> pl.Expr:
    """Try both YYYY-MM-DD and YYYYMMDD; coalesce. strict=False -> null on fail."""
    c = pl.col(col).cast(pl.Utf8, strict=False)
    iso = c.str.to_date(format="%Y-%m-%d", strict=False)
    compact = c.str.to_date(format="%Y%m%d", strict=False)
    return pl.coalesce([iso, compact]).alias(col)


def _date_parse_expr_as(col: str, alias: str) -> pl.Expr:
    c = pl.col(col).cast(pl.Utf8, strict=False)
    iso = c.str.to_date(format="%Y-%m-%d", strict=False)
    compact = c.str.to_date(format="%Y%m%d", strict=False)
    return pl.coalesce([iso, compact]).alias(alias)


def _recast_ccm(schema: pl.Schema) -> list[pl.Expr]:
    """Build cast expressions for the 1045-column CCM file.

    The file is read as all-Utf8; here we coerce by column name. Unknown columns
    default to Float32 — the long tail is Compustat accounting numerics.
    """
    exprs: list[pl.Expr] = []
    for name in schema.names():
        if name in _DATE_COLS_CCM:
            exprs.append(_date_parse_expr(name))
        elif name in _INT32_COLS_CCM:
            exprs.append(pl.col(name).cast(pl.Int32, strict=False))
        elif name in _INT64_COLS_CCM:
            exprs.append(pl.col(name).cast(pl.Int64, strict=False))
        elif name in _STRING_COLS_CCM:
            exprs.append(pl.col(name))  # already Utf8
        else:
            exprs.append(pl.col(name).cast(pl.Float32, strict=False))
    return exprs


# --------------------------------------------------------------------------- #
# Converters                                                                  #
# --------------------------------------------------------------------------- #

def convert_ccm_merged() -> None:
    src = _raw_path("ccm_merged")
    dst = OUT / "ccm_merged.parquet"
    log.info("starting ccm_merged (%s)", src.name)

    tmp = dst.with_suffix(".raw.parquet.tmp")
    # infer_schema_length=0 -> read everything as Utf8; we cast deliberately below.
    # This is robust to alpha values in numeric-looking columns (e.g. UIID="00X").
    pl.scan_csv(src, infer_schema_length=0, ignore_errors=False,
                null_values=["", "NA", "NaN"]) \
        .sink_parquet(tmp, compression="snappy")

    lf = pl.scan_parquet(tmp)
    schema = lf.collect_schema()
    _atomic_sink(lf.select(_recast_ccm(schema)), dst)
    tmp.unlink(missing_ok=True)

    lf_final = pl.scan_parquet(dst)
    n_rows = lf_final.select(pl.len()).collect().item()
    stats = lf_final.select(
        pl.col("PERMNO").n_unique().alias("n_permno"),
        pl.col("gvkey").n_unique().alias("n_gvkey"),
        pl.col("MthCalDt").min().alias("date_min"),
        pl.col("MthCalDt").max().alias("date_max"),
        (pl.col("at").is_null().sum() / pl.len()).alias("frac_null_at"),
    ).collect().row(0, named=True)
    log.info("  ccm_merged: rows=%d permno=%d gvkey=%d date=%s..%s frac_null(at)=%.3f",
             n_rows, stats["n_permno"], stats["n_gvkey"],
             stats["date_min"], stats["date_max"], stats["frac_null_at"])
    log.info("finished ccm_merged: wrote %d rows to %s", n_rows, dst)


def convert_crsp_daily() -> None:
    src = _raw_path("crsp_daily")
    dst = OUT / "crsp_daily.parquet"
    log.info("starting crsp_daily (%s)", src.name)

    tmp = dst.with_suffix(".raw.parquet.tmp")
    # All-Utf8 read; cast deliberately in the second pass. CRSP daily has
    # occasional alpha values in nominally-numeric columns (e.g. DLRETX="A").
    pl.scan_csv(src, infer_schema_length=0, ignore_errors=False,
                null_values=["", "NA", "NaN"]) \
        .sink_parquet(tmp, compression="snappy")

    lf = pl.scan_parquet(tmp)
    schema = lf.collect_schema()

    int32_cols = {"PERMNO", "PERMCO", "SHRCD", "EXCHCD", "SICCD", "ISSUNO",
                  "HEXCD", "HSICCD", "HSICMG", "HSICIG", "DLSTCD", "NWPERM",
                  "ACPERM", "ACCOMP", "SHRFLG", "DISTCD", "TRTSCD", "NMSIND",
                  "MMCNT", "NSDINX", "NUMTRD"}
    int64_cols = {"VOL", "SHROUT"}
    date_cols = {"date", "NAMEENDT", "DCLRDT", "DLPDT", "NEXTDT", "PAYDT",
                 "RCRDDT", "SHRENDDT"}
    float_cols = {"DLAMT", "DLRETX", "DLPRC", "DLRET", "DIVAMT", "FACPR",
                  "FACSHR", "BIDLO", "ASKHI", "PRC", "RET", "BID", "ASK",
                  "CFACPR", "CFACSHR", "OPENPRC", "RETX",
                  "vwretd", "vwretx", "ewretd", "ewretx", "sprtrn"}

    exprs: list[pl.Expr] = []
    for name in schema.names():
        if name in date_cols:
            exprs.append(_date_parse_expr(name))
        elif name in int32_cols:
            exprs.append(pl.col(name).cast(pl.Int32, strict=False))
        elif name in int64_cols:
            exprs.append(pl.col(name).cast(pl.Int64, strict=False))
        elif name in float_cols:
            exprs.append(pl.col(name).cast(pl.Float32, strict=False))
        else:
            exprs.append(pl.col(name))  # already Utf8

    _atomic_sink(lf.select(exprs), dst)
    tmp.unlink(missing_ok=True)

    lf_final = pl.scan_parquet(dst)
    n_rows = lf_final.select(pl.len()).collect().item()
    stats = lf_final.select(
        pl.col("PERMNO").n_unique().alias("n_permno"),
        pl.col("date").min().alias("date_min"),
        pl.col("date").max().alias("date_max"),
        (pl.col("RET").is_null().sum() / pl.len()).alias("frac_null_ret"),
        ((pl.col("RET") < -0.5) | (pl.col("RET") > 2.0)).sum().alias("n_ret_out_of_range"),
    ).collect().row(0, named=True)
    log.info("  crsp_daily: rows=%d permno=%d date=%s..%s frac_null(RET)=%.4f ret_out_of_range=%d",
             n_rows, stats["n_permno"], stats["date_min"], stats["date_max"],
             stats["frac_null_ret"], stats["n_ret_out_of_range"])

    tdpy = (
        lf_final
        .with_columns(pl.col("date").dt.year().alias("year"))
        .group_by(["PERMNO", "year"])
        .agg(pl.len().alias("n_days"))
        .select(pl.col("n_days").median().alias("median_tdpy"))
        .collect()
        .item()
    )
    log.info("  crsp_daily: median trading-days-per-permno-year=%.0f (expect ~250)", tdpy)
    log.info("finished crsp_daily: wrote %d rows to %s", n_rows, dst)


def convert_delisting() -> None:
    src = _raw_path("delisting")
    dst = OUT / "delisting.parquet"
    log.info("starting delisting (%s)", src.name)

    df = pl.read_csv(src, null_values=["", "NA", "NaN"])
    int32_cols = [c for c in ["PERMNO", "DLSTCD", "NWPERM", "NWCOMP", "ISSUNO",
                              "HEXCD", "HSICCD", "HSICMG", "HSICIG"] if c in df.columns]
    date_cols = [c for c in ["DLSTDT", "DLPDT", "NEXTDT"] if c in df.columns]
    float_cols = [c for c in ["DLAMT", "DLRET", "DLRETX", "DLPRC"] if c in df.columns]

    df = df.with_columns(
        [pl.col(c).cast(pl.Int32, strict=False) for c in int32_cols]
        + [_date_parse_expr(c) for c in date_cols]
        + [pl.col(c).cast(pl.Float32, strict=False) for c in float_cols]
    )
    _atomic_write(df, dst)

    # DLSTCD buckets: 100 active, 2xx merger, 3xx exchange, 4xx liquidation, 5xx dropped
    buckets = (
        df.select(
            (pl.col("DLSTCD") == 100).sum().alias("code_100"),
            ((pl.col("DLSTCD") >= 200) & (pl.col("DLSTCD") < 300)).sum().alias("code_2xx"),
            ((pl.col("DLSTCD") >= 300) & (pl.col("DLSTCD") < 400)).sum().alias("code_3xx"),
            ((pl.col("DLSTCD") >= 400) & (pl.col("DLSTCD") < 500)).sum().alias("code_4xx"),
            ((pl.col("DLSTCD") >= 500) & (pl.col("DLSTCD") < 600)).sum().alias("code_5xx"),
        ).row(0, named=True)
    )
    log.info("  delisting: rows=%d DLSTCD 100=%d 2xx=%d 3xx=%d 4xx=%d 5xx=%d",
             len(df), buckets["code_100"], buckets["code_2xx"],
             buckets["code_3xx"], buckets["code_4xx"], buckets["code_5xx"])
    log.info("finished delisting: wrote %d rows to %s", len(df), dst)


def convert_sp500_index() -> None:
    src = _raw_path("sp500_index")
    dst = OUT / "sp500_index.parquet"
    log.info("starting sp500_index (%s)", src.name)

    df = pl.read_csv(src, null_values=["", "NA", "NaN"])
    int32_cols = [c for c in ["totcnt", "usdcnt"] if c in df.columns]
    float_cols = [c for c in df.columns if c not in {"caldt", *int32_cols}]

    df = df.with_columns(
        [_date_parse_expr("caldt")]
        + [pl.col(c).cast(pl.Int32, strict=False) for c in int32_cols]
        + [pl.col(c).cast(pl.Float32, strict=False) for c in float_cols]
    )
    _atomic_write(df, dst)

    s = df.select(
        pl.col("caldt").min().alias("date_min"),
        pl.col("caldt").max().alias("date_max"),
        pl.col("sprtrn").mean().alias("mean_sprtrn"),
        pl.col("sprtrn").std().alias("std_sprtrn"),
        pl.col("vwretd").mean().alias("mean_vwretd"),
    ).row(0, named=True)
    log.info("  sp500_index: rows=%d date=%s..%s mean(sprtrn)=%.4f std=%.4f mean(vwretd)=%.4f",
             len(df), s["date_min"], s["date_max"],
             s["mean_sprtrn"], s["std_sprtrn"], s["mean_vwretd"])
    log.info("finished sp500_index: wrote %d rows to %s", len(df), dst)


# --- Fama-French ---

_DATE_TOKEN = re.compile(r"^\s*\d{6,8}\s*$")


def _read_french_csv(path: Path) -> pd.DataFrame:
    """Extract the daily data block from a Ken French CSV.

    File has: multi-line header, blank line, header row whose first token is empty
    (unnamed date column), then data rows whose first token is a 6- or 8-digit date,
    then a blank line + footer.
    """
    with open(path) as f:
        lines = f.readlines()
    start = next(
        i for i, line in enumerate(lines)
        if line.strip() and _DATE_TOKEN.match(line.strip().split(",")[0])
    )
    end = next(
        (i for i in range(start, len(lines))
         if not (lines[i].strip() and _DATE_TOKEN.match(lines[i].strip().split(",")[0]))),
        len(lines),
    )
    header_line = lines[start - 1].rstrip("\n")
    columns = [c.strip() for c in header_line.split(",")]
    if columns and columns[0] == "":
        columns[0] = "date"

    df = pd.read_csv(path, skiprows=start, nrows=end - start, header=None,
                     names=columns, skipinitialspace=True,
                     na_values=["-99.99", "-999"])
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    return df


_FF_RENAME = {"Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml",
              "RMW": "rmw", "CMA": "cma", "RF": "rf", "Mom": "mom"}


def _load_ff_file(key: str) -> pd.DataFrame | None:
    try:
        path = _raw_path(key)
    except FileNotFoundError:
        log.warning("  %s not found, skipping this FF component", RAW_FILES[key])
        return None
    df = _read_french_csv(path)
    df = df.rename(columns={k: v for k, v in _FF_RENAME.items() if k in df.columns})
    # Divide every non-date numeric column by 100 (percent -> decimal).
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    return df


def _build_ff_daily_merged() -> pd.DataFrame:
    ff3 = _load_ff_file("ff3_daily")
    ff5 = _load_ff_file("ff5_daily")
    mom = _load_ff_file("mom_daily")

    if ff3 is None and ff5 is None and mom is None:
        raise FileNotFoundError(RAW / "F-F_*_daily.csv")

    # Merge on date; FF5 wins on shared columns (mkt_rf, smb, hml, rf).
    frames = [x for x in (ff3, ff5, mom) if x is not None]
    daily = frames[0]
    for extra in frames[1:]:
        overlap = [c for c in extra.columns if c in daily.columns and c != "date"]
        new_cols = [c for c in extra.columns if c not in daily.columns]
        daily = daily.merge(extra[["date"] + new_cols], on="date", how="outer")
        if extra is ff5 and overlap:
            daily = daily.merge(extra[["date"] + overlap], on="date", how="left",
                                suffixes=("", "_ff5"))
            for c in overlap:
                daily[c] = daily[f"{c}_ff5"].combine_first(daily[c])
                daily.drop(columns=[f"{c}_ff5"], inplace=True)
    return daily.sort_values("date").reset_index(drop=True)


def convert_ff_factors_daily() -> None:
    dst = OUT / "ff_factors_daily.parquet"
    log.info("starting ff_factors_daily")

    daily = _build_ff_daily_merged()
    factor_cols = [c for c in ["mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"]
                   if c in daily.columns]
    out = daily[["date"] + factor_cols].copy()
    out["date"] = out["date"].dt.date
    for c in factor_cols:
        out[c] = out[c].astype("float32")

    pl_df = pl.from_pandas(out).with_columns(pl.col("date").cast(pl.Date))
    _atomic_write(pl_df, dst)

    summary = pl_df.select(
        pl.col("date").min().alias("date_min"),
        pl.col("date").max().alias("date_max"),
        *[pl.col(c).mean().alias(f"mean_{c}") for c in factor_cols],
        *[pl.col(c).std().alias(f"std_{c}") for c in factor_cols],
    ).row(0, named=True)
    log.info("  ff_factors_daily: rows=%d date=%s..%s", len(pl_df),
             summary["date_min"], summary["date_max"])
    for c in factor_cols:
        log.info("    %s: mean=%.6f std=%.5f", c,
                 summary[f"mean_{c}"], summary[f"std_{c}"])
    log.info("finished ff_factors_daily: wrote %d rows to %s", len(pl_df), dst)


def convert_ff_factors() -> None:
    dst = OUT / "ff_factors.parquet"
    log.info("starting ff_factors")

    daily = _build_ff_daily_merged()

    # Aggregate daily -> monthly via compounding.
    factor_cols = [c for c in ["mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"]
                   if c in daily.columns]
    daily = daily.sort_values("date").reset_index(drop=True)
    month = daily.set_index("date")[factor_cols]
    # Compound: prod(1 + r) - 1, skipping NaNs month-by-month.
    monthly = (1.0 + month).resample("ME").apply(lambda s: s.prod(skipna=False) - 1.0)
    monthly = monthly.reset_index()
    # Snap date to month-end (it already is via resample("M") but make it a plain date).
    monthly["date"] = monthly["date"].dt.date

    for c in factor_cols:
        monthly[c] = monthly[c].astype("float32")

    pl_df = pl.from_pandas(monthly).with_columns(pl.col("date").cast(pl.Date))
    _atomic_write(pl_df, dst)

    summary = pl_df.select(
        pl.col("date").min().alias("date_min"),
        pl.col("date").max().alias("date_max"),
        *[pl.col(c).mean().alias(f"mean_{c}") for c in factor_cols],
        *[pl.col(c).std().alias(f"std_{c}") for c in factor_cols],
    ).row(0, named=True)
    log.info("  ff_factors: rows=%d date=%s..%s", len(pl_df),
             summary["date_min"], summary["date_max"])
    for c in factor_cols:
        log.info("    %s: mean=%.5f std=%.5f", c,
                 summary[f"mean_{c}"], summary[f"std_{c}"])
    mean_mkt_rf = summary.get("mean_mkt_rf")
    if mean_mkt_rf is not None and abs(mean_mkt_rf) > 0.1:
        log.warning("  mean(mkt_rf)=%.4f — units look wrong (expect decimal ~0.005)", mean_mkt_rf)
    log.info("finished ff_factors: wrote %d rows to %s", len(pl_df), dst)


# --- FRED macro ---

_FRED_SERIES = {"vix": "VIXCLS", "baa": "BAA", "aaa": "AAA",
                "gs10": "GS10", "tb3ms": "TB3MS"}


def _load_fred_per_series() -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    for short, code in _FRED_SERIES.items():
        p = RAW / f"{code}.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p)
        # FRED CSVs have columns DATE + series code (or "observation_date" newer format).
        date_col = next(c for c in df.columns if c.lower() in ("date", "observation_date"))
        val_col = next(c for c in df.columns if c != date_col)
        df = df[[date_col, val_col]].rename(columns={date_col: "date", val_col: short})
        df["date"] = pd.to_datetime(df["date"])
        df[short] = pd.to_numeric(df[short], errors="coerce")
        frames.append(df)
    merged = frames[0]
    for extra in frames[1:]:
        merged = merged.merge(extra, on="date", how="outer")
    return merged


def convert_fred_macro() -> None:
    dst = OUT / "fred_macro.parquet"
    log.info("starting fred_macro")

    combined_path = RAW / RAW_FILES["fred_macro"]
    if combined_path.exists():
        df = pd.read_csv(combined_path)
        date_col = next(c for c in df.columns if c.lower() in ("date", "observation_date"))
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"])
        for c in df.columns:
            if c != "date":
                df[c] = pd.to_numeric(df[c], errors="coerce")
    else:
        df = _load_fred_per_series()
        if df is None:
            raise FileNotFoundError(f"{combined_path} (or per-series {list(_FRED_SERIES.values())})")

    df = df.sort_values("date").reset_index(drop=True)
    # Resample to monthly, take last observation in each month.
    df_m = df.set_index("date").resample("ME").last().reset_index()
    df_m["date"] = df_m["date"].dt.date

    # Derived series.
    if "vix" in df_m.columns:
        df_m["vix_chg"] = df_m["vix"].diff()
    if {"baa", "aaa"}.issubset(df_m.columns):
        df_m["credit_spread"] = df_m["baa"] - df_m["aaa"]
    if {"gs10", "tb3ms"}.issubset(df_m.columns):
        df_m["term_spread"] = df_m["gs10"] - df_m["tb3ms"]

    for c in df_m.columns:
        if c != "date":
            df_m[c] = df_m[c].astype("float32")

    pl_df = pl.from_pandas(df_m).with_columns(pl.col("date").cast(pl.Date))
    _atomic_write(pl_df, dst)

    summary = pl_df.select(
        pl.col("date").min().alias("date_min"),
        pl.col("date").max().alias("date_max"),
    ).row(0, named=True)
    log.info("  fred_macro: rows=%d date=%s..%s", len(pl_df),
             summary["date_min"], summary["date_max"])
    for c in [c for c in pl_df.columns if c != "date"]:
        s = pl_df.select(
            pl.col(c).is_not_null().sum().alias("n"),
            pl.col(c).min().alias("min"),
            pl.col(c).max().alias("max"),
            pl.col(c).mean().alias("mean"),
        ).row(0, named=True)
        log.info("    %s: n=%d min=%.4f max=%.4f mean=%.4f",
                 c, s["n"], s["min"] or float("nan"),
                 s["max"] or float("nan"), s["mean"] or float("nan"))
    log.info("finished fred_macro: wrote %d rows to %s", len(pl_df), dst)


# --- Still-to-pull placeholders ---

def convert_crsp_monthly() -> None:
    src = RAW / RAW_FILES["crsp_monthly"]
    if not src.exists():
        log.warning("skipping crsp_monthly: %s not yet downloaded, see Data/README.md", src)
        return
    raise NotImplementedError("crsp_monthly converter not yet implemented")


def convert_sp500_constituents() -> None:
    src = RAW / RAW_FILES["sp500_consts"]
    if not src.exists():
        log.warning("skipping sp500_constituents: %s not yet downloaded, see Data/README.md", src)
        return
    dst = OUT / "sp500_constituents.parquet"
    log.info("starting sp500_constituents (%s)", src.name)

    df = pl.read_csv(src, infer_schema_length=0, null_values=["", "NA", "NaN"])
    lower = {c.lower(): c for c in df.columns}
    direct_permno = next((lower[k] for k in ("permno", "lpermno") if k in lower), None)
    direct_start = next((lower[k] for k in ("start", "start_date", "from", "beg", "begdt") if k in lower), None)
    direct_end = next((lower[k] for k in ("ending", "end", "end_date", "thru", "thrudt") if k in lower), None)
    ticker_col = next((lower[k] for k in ("ticker", "tic", "symbol") if k in lower), None)

    if direct_permno is not None and direct_start is not None:
        out = (
            df.select(
                pl.col(direct_permno).cast(pl.Int32, strict=False).alias("permno"),
                _date_parse_expr_as(direct_start, "start"),
                _date_parse_expr_as(direct_end, "ending") if direct_end is not None else pl.lit(None, dtype=pl.Date).alias("ending"),
            )
            .drop_nulls(["permno", "start"])
            .with_columns(
                pl.col("ending").fill_null(pl.lit(PANEL_END, dtype=pl.Date)).alias("ending"),
            )
            .with_columns(
                pl.max_horizontal("start", pl.lit(PANEL_START, dtype=pl.Date)).alias("start"),
                pl.min_horizontal("ending", pl.lit(PANEL_END, dtype=pl.Date)).alias("ending"),
            )
            .filter(pl.col("ending") >= pl.col("start"))
            .sort(["permno", "start"])
            .unique()
        )
        style = "wrds"
    elif ticker_col is not None and direct_start is not None:
        memberships = (
            df.select(
                pl.col(ticker_col).cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("ticker"),
                _date_parse_expr_as(direct_start, "start"),
                _date_parse_expr_as(direct_end, "ending") if direct_end is not None else pl.lit(None, dtype=pl.Date).alias("ending"),
            )
            .drop_nulls(["ticker", "start"])
            .with_columns(
                pl.col("ending").fill_null(pl.lit(PANEL_END, dtype=pl.Date)).alias("ending"),
            )
            .with_columns(
                pl.max_horizontal("start", pl.lit(PANEL_START, dtype=pl.Date)).alias("start"),
                pl.min_horizontal("ending", pl.lit(PANEL_END, dtype=pl.Date)).alias("ending"),
            )
            .filter(pl.col("ending") >= pl.col("start"))
            .with_columns(
                pl.date_ranges(
                    pl.col("start").dt.month_start(),
                    pl.col("ending").dt.month_start(),
                    interval="1mo",
                    closed="both",
                ).alias("months")
            )
            .explode("months")
            .with_columns(
                pl.col("months").dt.year().alias("_year"),
                pl.col("months").dt.month().alias("_month"),
            )
            .select(["ticker", "_year", "_month"])
            .unique()
        )
        ccm = (
            pl.scan_parquet(OUT / "ccm_merged.parquet")
            .select(["PERMNO", "Ticker", "MthCalDt"])
            .filter(
                pl.col("MthCalDt").is_between(
                    pl.lit(PANEL_START, dtype=pl.Date),
                    pl.lit(PANEL_END, dtype=pl.Date),
                )
            )
            .filter(pl.col("Ticker").is_not_null())
            .collect()
            .with_columns(
                pl.col("PERMNO").cast(pl.Int32),
                pl.col("Ticker").cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("ticker"),
                pl.col("MthCalDt").dt.year().alias("_year"),
                pl.col("MthCalDt").dt.month().alias("_month"),
            )
            .select(["PERMNO", "MthCalDt", "ticker", "_year", "_month"])
            .unique()
        )
        matched = (
            memberships.join(ccm, on=["ticker", "_year", "_month"], how="inner")
            .select(["PERMNO", "MthCalDt"])
            .unique()
            .sort(["PERMNO", "MthCalDt"])
        )
        if matched.is_empty():
            raise ValueError("ticker-style S&P 500 constituents file did not match any CCM permnos")
        out = (
            matched.with_columns(
                (pl.col("MthCalDt").dt.year() * 12 + pl.col("MthCalDt").dt.month()).alias("_ym"),
            )
            .with_columns(
                pl.col("_ym").shift(1).over("PERMNO").alias("_prev_ym"),
            )
            .with_columns(
                (
                    pl.col("_prev_ym").is_null() | ((pl.col("_ym") - pl.col("_prev_ym")) != 1)
                ).cast(pl.Int32).cum_sum().over("PERMNO").alias("_segment")
            )
            .group_by(["PERMNO", "_segment"])
            .agg(
                pl.col("MthCalDt").min().alias("start"),
                pl.col("MthCalDt").max().alias("ending"),
            )
            .select(pl.col("PERMNO").alias("permno"), "start", "ending")
            .sort(["permno", "start"])
        )
        matched_tickers = memberships.select("ticker").unique().join(
            ccm.select("ticker").unique(), on="ticker", how="inner"
        ).height
        log.info(
            "  sp500_constituents ticker-style: matched %d/%d unique tickers into %d permno intervals",
            matched_tickers,
            memberships["ticker"].n_unique(),
            out.height,
        )
        style = "ticker"
    else:
        raise ValueError(
            "sp500_constituents.csv must contain either permno/start/ending columns "
            "or ticker/start_date/end_date-style columns"
        )

    _atomic_write(out, dst)
    stats = out.select(
        pl.len().alias("n_rows"),
        pl.col("permno").n_unique().alias("n_permno"),
        pl.col("start").min().alias("date_min"),
        pl.col("ending").max().alias("date_max"),
    ).row(0, named=True)
    log.info(
        "finished sp500_constituents (%s): rows=%d permnos=%d date=%s..%s -> %s",
        style,
        stats["n_rows"],
        stats["n_permno"],
        stats["date_min"],
        stats["date_max"],
        dst,
    )


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #

def main() -> None:
    converters = [
        convert_ccm_merged,
        convert_crsp_daily,
        convert_delisting,
        convert_sp500_index,
        convert_ff_factors,
        convert_ff_factors_daily,
        convert_fred_macro,
        convert_crsp_monthly,
        convert_sp500_constituents,
    ]
    for fn in converters:
        try:
            fn()
        except FileNotFoundError as e:
            log.warning("%s: skipping — %s not found", fn.__name__, e)
        except Exception as e:
            log.error("%s: failed — %s", fn.__name__, e, exc_info=True)


if __name__ == "__main__":
    main()
