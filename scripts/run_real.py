"""Rolling real-data training and prediction script."""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from inference.em import DEFAULT_MACRO_COLS, EMConfig, fit_em, predict_next_month
from rsgp.data import load_aligned_panel


def _year_month_index(dates: list, year_month: str) -> int:
    year, month = map(int, year_month.split("-"))
    for idx, date in enumerate(dates):
        if date.year == year and date.month == month:
            return idx
    raise ValueError(f"{year_month} not found in aligned dates")


def _serialize_gp_params(gp_params) -> list[dict[str, object]]:
    return [
        {
            "lengthscales": hp.lengthscales.copy(),
            "signal_var": float(hp.signal_var),
            "noise_var": float(hp.noise_var),
            "diagnostics": hp.diagnostics,
        }
        for hp in gp_params
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oos-start", default="2010-01")
    parser.add_argument("--oos-end", default=None)
    parser.add_argument("--output-dir", default="output/real")
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--gp-restarts", type=int, default=2)
    parser.add_argument("--transition-restarts", type=int, default=2)
    parser.add_argument("--max-prediction-points", type=int, default=800)
    args = parser.parse_args()

    aligned = load_aligned_panel(required_macro_cols=DEFAULT_MACRO_COLS)
    oos_start_idx = _year_month_index(aligned.dates, args.oos_start)
    oos_end_idx = len(aligned.dates) if args.oos_end is None else _year_month_index(aligned.dates, args.oos_end) + 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = EMConfig(
        K=2,
        max_iter=args.max_iter,
        gp_restarts=args.gp_restarts,
        transition_restarts=args.transition_restarts,
        max_prediction_points=args.max_prediction_points,
        min_regime_mass=5.0,
    )

    prediction_rows = []
    regime_rows = []
    param_history = []

    for t in range(oos_start_idx, oos_end_idx):
        result = fit_em(
            dates=aligned.dates[:t],
            returns=aligned.returns[:t],
            X=aligned.X[:t],
            Z=aligned.Z[:t],
            config=config,
            z_columns=aligned.z_columns,
        )
        pred = predict_next_month(
            result=result,
            train_X=aligned.X[:t],
            train_returns=aligned.returns[:t],
            test_X=aligned.X[t],
            next_z=aligned.Z[t],
            max_points=args.max_prediction_points,
            seed=t,
        )

        date = pd.to_datetime(aligned.dates[t])
        train_end = pd.to_datetime(aligned.dates[t - 1])
        prediction_rows.extend(
            {
                "date": date,
                "permno": int(permno),
                "model": "RSGP",
                "prediction": float(pred_mean),
                "prediction_var": float(pred_var),
                "actual_return": float(actual),
            }
            for permno, pred_mean, pred_var, actual in zip(
                aligned.permnos[t],
                pred["mean"],
                pred["var"],
                aligned.returns[t],
            )
        )
        regime_row = {
            "date": date,
            "train_end_date": train_end,
            "train_state": int(result.viterbi_path[-1]),
            "predicted_state": int(pred["next_regime_probs"].argmax()),
            "converged": bool(result.converged),
            "log_likelihood": float(result.log_likelihood_history[-1]),
        }
        for k, prob in enumerate(pred["next_regime_probs"]):
            regime_row[f"next_prob_{k}"] = float(prob)
        for k, prob in enumerate(result.gamma[-1]):
            regime_row[f"current_prob_{k}"] = float(prob)
        regime_rows.append(regime_row)
        param_history.append(
            {
                "date": date,
                "train_end_date": train_end,
                "pi": result.pi.copy(),
                "gp_params": _serialize_gp_params(result.gp_params),
                "transition_W": result.transition_W.copy(),
                "transition_b": result.transition_b.copy(),
                "log_likelihood_history": list(result.log_likelihood_history),
            }
        )
        print(
            f"{date.date()} train_end={train_end.date()} "
            f"pred_state={regime_row['predicted_state']} "
            f"loglik={regime_row['log_likelihood']:.2f}"
        )

    pd.DataFrame(prediction_rows).to_parquet(output_dir / "predictions.parquet", index=False)
    pd.DataFrame(regime_rows).to_parquet(output_dir / "regimes.parquet", index=False)
    with open(output_dir / "params.pkl", "wb") as f:
        pickle.dump(param_history, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_dir / "config.json", "w") as f:
        json.dump({"script_args": vars(args), "em_config": asdict(config)}, f, indent=2, default=str)

    print(
        f"wrote {len(prediction_rows):,} prediction rows, "
        f"{len(regime_rows):,} regime rows, "
        f"and {len(param_history):,} parameter snapshots to {output_dir}"
    )


if __name__ == "__main__":
    main()
