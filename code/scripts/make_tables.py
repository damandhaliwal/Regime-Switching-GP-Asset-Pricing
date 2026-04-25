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


def _regime_order_from_entry(entry: dict) -> np.ndarray:
    """Return old regime indices sorted by GP noise variance."""
    noise = np.array([float(gp["noise_var"]) for gp in entry["gp_params"]], dtype=np.float64)
    return np.argsort(noise, kind="stable")


def _inverse_order(order: np.ndarray) -> np.ndarray:
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order), dtype=order.dtype)
    return inverse


def _align_param_history_by_noise(param_history: list[dict]) -> list[dict]:
    aligned = []
    for entry in param_history:
        order = _regime_order_from_entry(entry)
        out = dict(entry)
        out["gp_params"] = [entry["gp_params"][idx] for idx in order]
        out["pi"] = np.asarray(entry["pi"])[order]
        out["transition_W"] = np.asarray(entry["transition_W"])[order][:, order, :]
        out["transition_b"] = np.asarray(entry["transition_b"])[order][:, order]
        aligned.append(out)
    return aligned


def _align_regimes_by_noise(regimes: pd.DataFrame, param_history: list[dict]) -> pd.DataFrame:
    """Align saved rolling labels so regime 0 is lower GP noise variance.

    Older result files may have been saved before this convention was enforced
    in run_real.py. Re-aligning here keeps tables and figures comparable across
    independent cold-start rolling fits without rerunning the full experiment.
    """
    if not param_history:
        return regimes.copy()

    orders_by_date = {
        pd.to_datetime(entry["date"]): _regime_order_from_entry(entry)
        for entry in param_history
    }
    out = regimes.copy()
    prob_prefixes = ("next_prob_", "current_prob_")
    state_cols = [c for c in ("train_state", "viterbi_state", "predicted_state") if c in out.columns]

    for idx, row in out.iterrows():
        order = orders_by_date.get(pd.to_datetime(row["date"]))
        if order is None:
            continue
        inverse = _inverse_order(order)
        for col in state_cols:
            if pd.notna(row[col]):
                out.at[idx, col] = int(inverse[int(row[col])])
        for prefix in prob_prefixes:
            old_cols = [f"{prefix}{old_idx}" for old_idx in range(len(order))]
            if not all(col in out.columns for col in old_cols):
                continue
            old_vals = np.array([row[col] for col in old_cols], dtype=np.float64)
            for new_idx, old_idx in enumerate(order):
                out.at[idx, f"{prefix}{new_idx}"] = old_vals[old_idx]
    return out


def _regime_col(regimes: pd.DataFrame) -> str:
    return "predicted_state" if "predicted_state" in regimes.columns else "viterbi_state"


def _table1(
    main_predictions: pd.DataFrame,
    regimes: pd.DataFrame,
    baselines: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    regime_col = _regime_col(regimes)
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
    regime_col = _regime_col(regimes)
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


STRESS_EPISODES = [
    ("Dot-com bust", "2000-03", "2002-10"),
    ("GFC", "2007-12", "2009-06"),
    ("COVID", "2020-02", "2020-04"),
]


def _figure_regime_probabilities(regimes: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    dates = pd.to_datetime(regimes["date"])
    dmin, dmax = dates.min(), dates.max()
    for col in sorted(c for c in regimes.columns if c.startswith("next_prob_")):
        ax.plot(dates, regimes[col], label=col.replace("next_prob_", "Regime "))
    drawn_label = False
    for name, start, end in STRESS_EPISODES:
        s = max(pd.Timestamp(start), dmin)
        e = min(pd.Timestamp(end), dmax)
        if s >= e:
            continue
        ax.axvspan(s, e, color="gray", alpha=0.25,
                   label="Stress episodes" if not drawn_label else None)
        drawn_label = True
        ax.text(s + (e - s) / 2, 1.02, name,
                ha="center", va="bottom", fontsize=8, color="dimgray")
    ax.set_ylim(0.0, 1.08)
    ax.set_xlim(dmin, dmax)
    ax.set_ylabel("Probability")
    ax.set_title("Predicted Regime Probabilities Over Time")
    ax.legend(loc="center left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _figure_alpha_comparison(table2: pd.DataFrame, out_path: Path) -> None:
    models = table2["model"].tolist()
    capm = 100.0 * table2["capm_alpha"].to_numpy(dtype=np.float64)
    ff5 = 100.0 * table2["ff5_alpha"].to_numpy(dtype=np.float64)
    y = np.arange(len(models))
    height = 0.4
    fig, ax = plt.subplots(figsize=(10, 0.7 * len(models) + 2))
    ax.barh(y - height / 2, capm, height=height, label="CAPM alpha", color="#4C72B0")
    ax.barh(y + height / 2, ff5, height=height, label="FF5 alpha", color="#DD8452")
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.axvline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel("Annualized alpha (%)")
    ax.set_title("Long-Short Portfolio Alpha by Model and Regime")
    for yi, (c, f) in enumerate(zip(capm, ff5)):
        ax.text(c, yi - height / 2, f" {c:.1f}", va="center", ha="left" if c >= 0 else "right", fontsize=8)
        ax.text(f, yi + height / 2, f" {f:.1f}", va="center", ha="left" if f >= 0 else "right", fontsize=8)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _figure_kernel_sensitivity(
    table2: pd.DataFrame,
    out_path: Path,
    primary_label: str,
    alt_label: str,
) -> None:
    rows = table2.set_index("model")
    missing = [label for label in (primary_label, alt_label) if label not in rows.index]
    if missing:
        raise ValueError(f"missing kernel sensitivity rows: {missing}")

    labels = [primary_label, alt_label]
    metrics = [
        ("ann_return", "Annualized return (%)", 100.0),
        ("sharpe", "Sharpe ratio", 1.0),
        ("ff5_alpha", "FF5 alpha (%)", 100.0),
        ("max_drawdown", "Max drawdown (%)", 100.0),
    ]
    colors = ["#4C72B0", "#55A868"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, (col, title, scale) in zip(axes.ravel(), metrics):
        values = [float(rows.loc[label, col]) * scale for label in labels]
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=colors, width=0.55)
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(["SE-ARD", "Matérn-5/2"])
        for bar, value in zip(bars, values):
            va = "bottom" if value >= 0 else "top"
            offset = 0.02 * max(1.0, max(abs(v) for v in values))
            y = value + offset if value >= 0 else value - offset
            ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:.2f}",
                    ha="center", va=va, fontsize=8)
    fig.suptitle("Kernel Sensitivity: SE-ARD vs Matérn-5/2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _figure_kernel_sensitivity(table2: pd.DataFrame, out_path: Path,
                                primary_label: str, alt_label: str) -> None:
    """Side-by-side bar chart of Sharpe and FF5 alpha for SE-ARD vs alternative kernel.

    Compares the unconditional row and the regime-conditional rows for both
    kernel families.
    """
    pairs = [
        (primary_label, alt_label),
        (f"{primary_label} (Regime 0)", f"{alt_label} (Regime 0)"),
        (f"{primary_label} (Regime 1)", f"{alt_label} (Regime 1)"),
    ]
    by_model = {row["model"]: row for _, row in table2.iterrows()}
    rows = []
    for primary, alt in pairs:
        if primary in by_model and alt in by_model:
            rows.append((primary.replace(primary_label, "RSGP"),
                         by_model[primary]["sharpe"], by_model[alt]["sharpe"],
                         100.0 * by_model[primary]["ff5_alpha"], 100.0 * by_model[alt]["ff5_alpha"]))

    if not rows:
        return

    labels = [r[0] for r in rows]
    se_sharpe = [r[1] for r in rows]
    mat_sharpe = [r[2] for r in rows]
    se_alpha = [r[3] for r in rows]
    mat_alpha = [r[4] for r in rows]

    x = np.arange(len(labels))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(x - width / 2, se_sharpe, width=width, label="SE-ARD", color="#4C72B0")
    axes[0].bar(x + width / 2, mat_sharpe, width=width, label="Matérn-5/2", color="#DD8452")
    axes[0].set_title("Annualized Sharpe")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].legend()

    axes[1].bar(x - width / 2, se_alpha, width=width, label="SE-ARD", color="#4C72B0")
    axes[1].bar(x + width / 2, mat_alpha, width=width, label="Matérn-5/2", color="#DD8452")
    axes[1].set_title("Annualized FF5 alpha (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].legend()

    fig.suptitle("Kernel sensitivity: SE-ARD vs Matérn-5/2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _transition_matrix_for_covariate(W: np.ndarray, b: np.ndarray, z: np.ndarray) -> np.ndarray:
    logits = np.einsum("jkm,m->jk", W, z) + b
    logits[:, 0] = 0.0
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    return probs / probs.sum(axis=1, keepdims=True)


def _macro_frame_for_dates(dates: pd.Series | list[pd.Timestamp]) -> pd.DataFrame:
    aligned = load_aligned_panel(required_macro_cols=DEFAULT_MACRO_COLS)
    frame = pd.DataFrame(aligned.Z, columns=aligned.z_columns)
    frame["date"] = pd.to_datetime(aligned.dates)
    want = pd.to_datetime(pd.Series(dates)).drop_duplicates()
    return frame[frame["date"].isin(set(want))].sort_values("date")


def _figure_transition_vix_sensitivity(
    param_history: list[dict],
    oos_dates: pd.Series,
    out_path: Path,
) -> None:
    macro = _macro_frame_for_dates(oos_dates)
    if macro.empty or "vix_lag1" not in macro.columns:
        raise ValueError("vix_lag1 is required for transition sensitivity figure")

    z_cols = DEFAULT_MACRO_COLS
    z_base = macro[z_cols].median().to_numpy(dtype=np.float64)
    vix_idx = z_cols.index("vix_lag1")
    vix = macro["vix_lag1"].to_numpy(dtype=np.float64)
    vix_grid = np.linspace(np.nanpercentile(vix, 5), np.nanpercentile(vix, 95), 80)

    curves = {0: [], 1: []}
    for entry in param_history:
        W = np.asarray(entry["transition_W"], dtype=np.float64)
        b = np.asarray(entry["transition_b"], dtype=np.float64)
        p_from = {0: [], 1: []}
        for v in vix_grid:
            z = z_base.copy()
            z[vix_idx] = v
            P = _transition_matrix_for_covariate(W, b, z)
            p_from[0].append(P[0, 1])
            p_from[1].append(P[1, 1])
        curves[0].append(p_from[0])
        curves[1].append(p_from[1])

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = {
        0: r"$P(s_t=1\mid s_{t-1}=0,\mathrm{VIX}_{t-1})$",
        1: r"$P(s_t=1\mid s_{t-1}=1,\mathrm{VIX}_{t-1})$",
    }
    colors = {0: "#4C72B0", 1: "#DD8452"}
    for source in (0, 1):
        arr = np.asarray(curves[source], dtype=np.float64)
        mean = arr.mean(axis=0)
        lo, hi = np.percentile(arr, [10, 90], axis=0)
        ax.plot(vix_grid, mean, color=colors[source], label=labels[source])
        ax.fill_between(vix_grid, lo, hi, color=colors[source], alpha=0.18, linewidth=0)

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Lagged VIX")
    ax.set_ylabel("Predicted transition probability")
    ax.set_title("Macro-Driven Transition Sensitivity to Lagged VIX")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _extra_kernel_rows(
    alt_dir: Path,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load an alternative RSGP run and return (table1_rows, table2_rows, aligned_regimes)."""
    preds = pd.read_parquet(alt_dir / "predictions.parquet")
    regimes = pd.read_parquet(alt_dir / "regimes.parquet")
    with open(alt_dir / "params.pkl", "rb") as f:
        params = pickle.load(f)
    regimes = _align_regimes_by_noise(regimes, params)

    regime_col = _regime_col(regimes)
    actual, pred, _ = _group_monthly_arrays(preds, "prediction")
    states = regimes.sort_values("date")[regime_col].to_numpy(dtype=np.int32)
    by_regime = regime_conditional_oos_r2(actual, pred, states)
    t1 = pd.DataFrame([{
        "model": label,
        "oos_r2": pooled_oos_r2(actual, pred),
        "regime_0_r2": by_regime.get(0, np.nan),
        "regime_1_r2": by_regime.get(1, np.nan),
    }])

    portfolio = decile_long_short_from_frame(preds)
    rows = [_portfolio_summary(label, portfolio)]
    merged = portfolio.merge(regimes[["date", regime_col]], on="date", how="left")
    for r in sorted(merged[regime_col].dropna().unique()):
        rows.append(_portfolio_summary(f"{label} (Regime {int(r)})", merged[merged[regime_col] == r]))
    t2 = pd.DataFrame(rows)
    return t1, t2, regimes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", default="output/real")
    parser.add_argument("--compare-kernel-dir", default=None,
                        help="Path to a Matérn-5/2 rolling-OOS run (e.g. output/real_matern52)."
                             " If set, adds kernel-comparison rows to Tables 1-2 and writes Figure 5.")
    parser.add_argument("--compare-kernel-label", default="RSGP (Matérn-5/2)")
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
    regimes = _align_regimes_by_noise(regimes, param_history)
    param_history = _align_param_history_by_noise(param_history)

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

    if args.compare_kernel_dir is not None:
        alt_t1, alt_t2, _ = _extra_kernel_rows(Path(args.compare_kernel_dir), args.compare_kernel_label)
        table1 = pd.concat([table1, alt_t1], ignore_index=True).sort_values("oos_r2", ascending=False)
        table2 = pd.concat([table2, alt_t2], ignore_index=True)

    _save_table(table1, tables_dir / "table_1_prediction_performance")
    _save_table(table2, tables_dir / "table_2_portfolio_performance")

    _figure_lengthscales(param_history, figures_dir / "regime_lengthscales.png")
    _figure_regime_probabilities(regimes, figures_dir / "regime_probabilities.png")
    _figure_alpha_comparison(table2, figures_dir / "alpha_comparison.png")
    _figure_transition_vix_sensitivity(
        param_history,
        main_predictions["date"],
        figures_dir / "transition_vix_sensitivity.png",
    )

    if args.compare_kernel_dir is not None:
        _figure_kernel_sensitivity(table2, figures_dir / "kernel_sensitivity.png",
                                   primary_label="RSGP", alt_label=args.compare_kernel_label)

    print(f"wrote {tables_dir / 'table_1_prediction_performance.csv'}")
    print(f"wrote {tables_dir / 'table_2_portfolio_performance.csv'}")
    print(f"wrote {figures_dir / 'regime_lengthscales.png'}")
    print(f"wrote {figures_dir / 'regime_probabilities.png'}")
    print(f"wrote {figures_dir / 'alpha_comparison.png'}")
    print(f"wrote {figures_dir / 'transition_vix_sensitivity.png'}")
    if args.compare_kernel_dir is not None:
        print(f"wrote {figures_dir / 'kernel_sensitivity.png'}")


if __name__ == "__main__":
    main()
