"""Rolling real-data training and prediction script.

Parallelized across OOS months with a process pool. Each worker runs an
independent cold EM fit for its assigned month — we trade the warm-start
smoothness of the serial loop for wall-clock speed (~N_workers×).

Single-threaded BLAS inside each worker (MKL/OpenBLAS ignore the process
pool and will otherwise oversubscribe cores).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _worker_init() -> None:
    """Configure each worker process for single-thread BLAS + JAX CPU x64."""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    # JAX CPU x64 is set at import time in models.gp_jax.


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


def _fit_one_month(task: dict) -> dict:
    """Run one cold EM fit + next-month prediction. Must be top-level + picklable."""
    # Imports inside the worker so each child picks up the per-worker env flags.
    from inference.em import fit_em, predict_next_month
    from rsgp.data import AlignedPanel

    aligned: AlignedPanel = task["aligned"]
    config = task["config"]
    t: int = task["t"]

    t_start = time.time()
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
        max_points=config.max_prediction_points,
        seed=t,
    )

    date = pd.to_datetime(aligned.dates[t])
    train_end = pd.to_datetime(aligned.dates[t - 1])
    prediction_rows = [
        {
            "date": date,
            "permno": int(permno),
            "model": "RSGP",
            "prediction": float(pred_mean),
            "prediction_var": float(pred_var),
            "actual_return": float(actual),
        }
        for permno, pred_mean, pred_var, actual in zip(
            aligned.permnos[t], pred["mean"], pred["var"], aligned.returns[t]
        )
    ]
    regime_row = {
        "date": date,
        "train_end_date": train_end,
        "train_state": int(result.viterbi_path[-1]),
        "viterbi_state": int(result.viterbi_path[-1]),
        "predicted_state": int(pred["next_regime_probs"].argmax()),
        "converged": bool(result.converged),
        "full_log_likelihood": float(result.full_log_likelihood),
        "subsampled_log_likelihood": float(result.subsampled_log_likelihood_history[-1]),
        "n_em_iters": len(result.subsampled_log_likelihood_history),
    }
    for k, prob in enumerate(pred["next_regime_probs"]):
        regime_row[f"next_prob_{k}"] = float(prob)
    for k, prob in enumerate(result.gamma[-1]):
        regime_row[f"current_prob_{k}"] = float(prob)
    param_entry = {
        "date": date,
        "train_end_date": train_end,
        "pi": result.pi.copy(),
        "gp_params": _serialize_gp_params(result.gp_params),
        "transition_W": result.transition_W.copy(),
        "transition_b": result.transition_b.copy(),
        "subsampled_log_likelihood_history": list(result.subsampled_log_likelihood_history),
        "full_log_likelihood": float(result.full_log_likelihood),
    }
    return {
        "t": t,
        "prediction_rows": prediction_rows,
        "regime_row": regime_row,
        "param_entry": param_entry,
        "wall_seconds": time.time() - t_start,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oos-start", default="2010-01")
    parser.add_argument("--oos-end", default=None)
    parser.add_argument("--output-dir", default="output/real")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=None,
                        help="Override EMConfig.max_iter (default: use EMConfig default)")
    parser.add_argument("--gp-restarts", type=int, default=None)
    parser.add_argument("--transition-restarts", type=int, default=None)
    parser.add_argument("--max-prediction-points", type=int, default=None)
    parser.add_argument("--max-monthly-points", type=int, default=None)
    parser.add_argument("--max-months-per-regime-fit", type=int, default=None,
                        help="Cap months per regime in the M-step (None disables).")
    args = parser.parse_args()

    # Import after argparse so --help is fast and so main process has JAX ready
    # (workers import fresh inside _fit_one_month).
    from inference.em import DEFAULT_MACRO_COLS, EMConfig
    from rsgp.data import load_aligned_panel

    aligned = load_aligned_panel(required_macro_cols=DEFAULT_MACRO_COLS)
    oos_start_idx = _year_month_index(aligned.dates, args.oos_start)
    oos_end_idx = len(aligned.dates) if args.oos_end is None else _year_month_index(aligned.dates, args.oos_end) + 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_kwargs = {"K": 2, "min_regime_mass": 5.0, "strict_monotone": False}
    for attr, cli in (
        ("max_iter", args.max_iter),
        ("gp_restarts", args.gp_restarts),
        ("transition_restarts", args.transition_restarts),
        ("max_prediction_points", args.max_prediction_points),
        ("max_monthly_points", args.max_monthly_points),
        ("max_months_per_regime_fit", args.max_months_per_regime_fit),
    ):
        if cli is not None:
            config_kwargs[attr] = cli
    config = EMConfig(**config_kwargs)

    tasks = [
        {"t": t, "aligned": aligned, "config": config}
        for t in range(oos_start_idx, oos_end_idx)
    ]
    print(f"dispatching {len(tasks)} OOS months across {args.workers} workers")

    results: list[dict] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
        futures = {pool.submit(_fit_one_month, task): task["t"] for task in tasks}
        for i, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            results.append(res)
            pct = 100.0 * i / len(tasks)
            elapsed = time.time() - t0
            eta = elapsed * (len(tasks) - i) / max(i, 1)
            date_str = res["regime_row"]["date"].date().isoformat()
            print(
                f"[{i:3d}/{len(tasks)}] {date_str} "
                f"viterbi={res['regime_row']['viterbi_state']} "
                f"iters={res['regime_row']['n_em_iters']} "
                f"fit={res['wall_seconds']:.1f}s  "
                f"elapsed={elapsed:.0f}s  eta={eta:.0f}s  ({pct:.0f}%)",
                flush=True,
            )

    results.sort(key=lambda r: r["t"])
    prediction_rows = [row for r in results for row in r["prediction_rows"]]
    regime_rows = [r["regime_row"] for r in results]
    param_history = [r["param_entry"] for r in results]

    pd.DataFrame(prediction_rows).to_parquet(output_dir / "predictions.parquet", index=False)
    pd.DataFrame(regime_rows).to_parquet(output_dir / "regimes.parquet", index=False)
    with open(output_dir / "params.pkl", "wb") as f:
        pickle.dump(param_history, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_dir / "config.json", "w") as f:
        json.dump({"script_args": vars(args), "em_config": asdict(config)}, f, indent=2, default=str)

    print(
        f"wrote {len(prediction_rows):,} prediction rows, "
        f"{len(regime_rows):,} regime rows, "
        f"and {len(param_history):,} parameter snapshots to {output_dir} "
        f"in {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
