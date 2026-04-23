"""Generate final evaluation tables and figures."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.baselines import run_baselines_rolling
from eval.metrics import (
    annualized_mean,
    annualized_volatility,
    capm_alpha,
    ff5_alpha,
    max_drawdown,
    pooled_oos_r2,
    regime_conditional_oos_r2,
    sharpe_ratio,
)
from eval.portfolios import decile_long_short_from_frame
from inference.em import DEFAULT_MACRO_COLS
from rsgp.characteristics import CHAR_COLS
from rsgp.data import load_aligned_panel


def _year_month_index(dates: list, year_month: str) -> int:
    year, month = map(int, year_month.split("-"))
    for idx, date in enumerate(dates):
        if date.year == year and date.month == month:
            return idx
    raise ValueError(f"{year_month} not found in aligned dates")


def _group_monthly_arrays(frame: pd.DataFrame, pred_col: str) -> tuple[list[np.ndarray], list[np.ndarray], list[pd.Timestamp]]:
    actual = []
    predicted = []
    dates = []
    for date, group in frame.groupby("date", sort=True):
        actual.append(group["actual_return"].to_numpy(dtype=np.float64))
        predicted.append(group[pred_col].to_numpy(dtype=np.float64))
        dates.append(pd.to_datetime(date))
    return actual, predicted, dates


def _table1(
    main_predictions: pd.DataFrame,
    regimes: pd.DataFrame,
    baselines: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    regime_col = "viterbi_state" if "viterbi_state" in regimes.columns else "predicted_state"
    main_actual, main_pred, _ = _group_monthly_arrays(main_predictions, "prediction")
    states = regimes.sort_values("date")[regime_col].to_numpy(dtype=np.int32)
    main_regime = regime_conditional_oos_r2(main_actual, main_pred, states)
    rows.append({
        "model": "RSGP",
        "oos_r2": pooled_oos_r2(main_actual, main_pred),
        "regime_0_r2": main_regime.get(0, np.nan),
        "regime_1_r2": main_regime.get(1, np.nan),
    })

    state_map = regimes[["date", regime_col]].copy()
    for model, group in baselines.groupby("model"):
        merged = group.merge(state_map, on="date", how="left")
        actual, pred, dates = _group_monthly_arrays(merged, "prediction")
        states_model = (
            merged[["date", regime_col]]
            .drop_duplicates()
            .sort_values("date")[regime_col]
            .to_numpy(dtype=np.int32)
        )
        reg = regime_conditional_oos_r2(actual, pred, states_model)
        rows.append({
            "model": model,
            "oos_r2": pooled_oos_r2(actual, pred),
            "regime_0_r2": reg.get(0, np.nan),
            "regime_1_r2": reg.get(1, np.nan),
        })
    return pd.DataFrame(rows).sort_values("oos_r2", ascending=False)


def _portfolio_summary(label: str, portfolio_df: pd.DataFrame) -> dict[str, float | str]:
    returns = portfolio_df["long_short"].to_numpy(dtype=np.float64)
    return {
        "model": label,
        "ann_return": annualized_mean(returns),
        "ann_vol": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "capm_alpha": capm_alpha(portfolio_df),
        "ff5_alpha": ff5_alpha(portfolio_df),
        "max_drawdown": max_drawdown(returns),
    }


def _table2(
    main_predictions: pd.DataFrame,
    regimes: pd.DataFrame,
    baselines: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    regime_col = "viterbi_state" if "viterbi_state" in regimes.columns else "predicted_state"
    main_portfolio = decile_long_short_from_frame(main_predictions)
    rows.append(_portfolio_summary("RSGP", main_portfolio))

    merged_main = main_portfolio.merge(regimes[["date", regime_col]], on="date", how="left")
    for regime in sorted(merged_main[regime_col].dropna().unique()):
        rows.append(_portfolio_summary(f"RSGP (Regime {int(regime)})", merged_main[merged_main[regime_col] == regime]))

    for model, group in baselines.groupby("model"):
        rows.append(_portfolio_summary(model, decile_long_short_from_frame(group)))
    return pd.DataFrame(rows)


def _save_table(df: pd.DataFrame, out_base: Path) -> None:
    df.to_csv(out_base.with_suffix(".csv"), index=False)
    lines = []
    lines.append("\\begin{tabular}{" + "l" * len(df.columns) + "}")
    lines.append("\\hline")
    lines.append(" & ".join(df.columns) + " \\\\")
    lines.append("\\hline")
    for row in df.itertuples(index=False):
        vals = []
        for val in row:
            if isinstance(val, (float, np.floating)):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append(" & ".join(vals) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    with open(out_base.with_suffix(".tex"), "w") as f:
        f.write("\n".join(lines))


def _figure_lengthscales(param_history: list[dict], out_path: Path) -> None:
    k = len(param_history[0]["gp_params"])
    mean_ls = []
    for regime in range(k):
        ls = np.vstack([entry["gp_params"][regime]["lengthscales"] for entry in param_history])
        mean_ls.append(ls.mean(axis=0))

    x = np.arange(len(CHAR_COLS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    for regime, ls in enumerate(mean_ls):
        ax.bar(x + regime * width, ls, width=width, label=f"Regime {regime}")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(CHAR_COLS, rotation=45, ha="right")
    ax.set_ylabel("Average ARD Lengthscale")
    ax.set_title("Regime-Dependent ARD Lengthscales")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _figure_regime_probabilities(regimes: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    for col in sorted(c for c in regimes.columns if c.startswith("next_prob_")):
        ax.plot(pd.to_datetime(regimes["date"]), regimes[col], label=col.replace("next_prob_", "Regime "))
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Probability")
    ax.set_title("Predicted Regime Probabilities Over Time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", default="output/real")
    parser.add_argument("--tables-dir", default="output/tables")
    parser.add_argument("--figures-dir", default="output/figures")
    parser.add_argument("--oos-start", default="2010-01")
    args = parser.parse_args()

    real_dir = Path(args.real_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    main_predictions = pd.read_parquet(real_dir / "predictions.parquet")
    regimes = pd.read_parquet(real_dir / "regimes.parquet")
    with open(real_dir / "params.pkl", "rb") as f:
        param_history = pickle.load(f)

    baselines_path = real_dir / "baselines.parquet"
    if baselines_path.exists():
        baselines = pd.read_parquet(baselines_path)
    else:
        aligned = load_aligned_panel(required_macro_cols=DEFAULT_MACRO_COLS)
        oos_start_idx = _year_month_index(aligned.dates, args.oos_start)
        end_date = pd.to_datetime(main_predictions["date"]).max()
        oos_end_idx = next(
            idx + 1 for idx, date in enumerate(aligned.dates)
            if date.year == end_date.year and date.month == end_date.month
        )
        baselines = run_baselines_rolling(aligned, oos_start_idx=oos_start_idx, oos_end_idx=oos_end_idx)
        baselines.to_parquet(baselines_path, index=False)

    table1 = _table1(main_predictions, regimes, baselines)
    table2 = _table2(main_predictions, regimes, baselines)
    _save_table(table1, tables_dir / "table_1_prediction_performance")
    _save_table(table2, tables_dir / "table_2_portfolio_performance")

    _figure_lengthscales(param_history, figures_dir / "regime_lengthscales.png")
    _figure_regime_probabilities(regimes, figures_dir / "regime_probabilities.png")

    print(f"wrote {tables_dir / 'table_1_prediction_performance.csv'}")
    print(f"wrote {tables_dir / 'table_2_portfolio_performance.csv'}")
    print(f"wrote {figures_dir / 'regime_lengthscales.png'}")
    print(f"wrote {figures_dir / 'regime_probabilities.png'}")


if __name__ == "__main__":
    main()
