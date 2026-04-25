"""Smoke test for idio_vol OLS residual-stdev kernel."""
from __future__ import annotations

import numpy as np

from rsgp.characteristics import _idio_vol_for_window


def test_recovers_residual_stdev() -> None:
    # Average across many seeds to get a stable point estimate; single-window
    # sampling error on a 60-day OLS with true sigma=0.02 is ~σ/sqrt(2(N-p)) ≈ 0.002.
    T = 60
    beta_true = 1.2
    sigma_true = 0.02
    iv_dailies = []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        mkt = rng.normal(0.0, 0.01, size=T)
        eps = rng.normal(0.0, sigma_true, size=T)
        r = 0.0001 + beta_true * mkt + eps
        iv_monthly = _idio_vol_for_window(r, mkt)
        iv_dailies.append(iv_monthly / np.sqrt(21.0))
    iv_daily = float(np.mean(iv_dailies))
    rel_err = abs(iv_daily - sigma_true) / sigma_true
    assert rel_err < 0.05, f"recovered daily stdev {iv_daily:.5f} vs true {sigma_true:.5f} (rel_err={rel_err:.3f})"
    print(f"  test_recovers_residual_stdev OK (iv_daily={iv_daily:.5f}, true={sigma_true:.5f}, rel_err={rel_err:.3f})")


def main() -> None:
    print("test_idio_vol:")
    test_recovers_residual_stdev()
    print("all passed")


if __name__ == "__main__":
    main()
