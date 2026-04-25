"""Portfolio construction helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def decile_long_short_from_frame(
    frame: pd.DataFrame,
    pred_col: str = "prediction",
    actual_col: str = "actual_return",
    n_buckets: int = 10,
) -> pd.DataFrame:
    rows = []
    for date, group in frame.groupby("date", sort=True):
        pred = group[pred_col].to_numpy(dtype=np.float64)
        actual = group[actual_col].to_numpy(dtype=np.float64)
        if pred.size < n_buckets:
            continue
        order = np.argsort(pred)
        buckets = np.array_split(order, n_buckets)
        short_idx = buckets[0]
        long_idx = buckets[-1]
        rows.append({
            "date": pd.to_datetime(date),
            "long_short": float(actual[long_idx].mean() - actual[short_idx].mean()),
            "long_leg": float(actual[long_idx].mean()),
            "short_leg": float(actual[short_idx].mean()),
            "n_long": int(len(long_idx)),
            "n_short": int(len(short_idx)),
        })
    return pd.DataFrame(rows)
