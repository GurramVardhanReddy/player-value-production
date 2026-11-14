"""Synthetic feature tables for unit tests only. No reported result comes from these."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def synthetic_features(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # first bets spread like the real cohort: mostly February, a tail to early June
    days = np.minimum(rng.exponential(25, n).astype(int), 122)
    t0 = pd.Timestamp("2005-02-01") + pd.to_timedelta(days, unit="D")
    active = rng.integers(1, 60, n)
    stake = rng.gamma(1.2, 300, n) * active / 10
    live = stake * rng.beta(1, 4, n)
    df = pd.DataFrame({
        "UserID": np.arange(1, n + 1), "t0": t0.date, "stake_90": stake, "stake_live_90": live,
        "win_90": stake * rng.uniform(0.6, 1.1, n), "bets_90": active * rng.integers(1, 6, n),
        "active_days_90": active, "last_active_day": np.minimum(89, rng.integers(0, 90, n)),
        "stake_wk1": stake * 0.2, "stake_m1": stake * 0.4, "stake_m3": stake * 0.3,
        "active_days_m3": np.minimum(30, active // 3), "active_days_last14": np.minimum(14, active // 6),
        "max_day_stake": stake * 0.3, "max_7d_stake": stake * 0.5,
        "half_stake_day": rng.integers(0, 90, n), "live_days_90": active // 4,
        "mean_gap": rng.uniform(1, 10, n), "max_gap": rng.uniform(1, 40, n),
        "sd_gap": rng.uniform(0, 8, n), "CountryID": rng.choice([276, 792, 616, 1, 2], n),
        "Gender": rng.integers(0, 2, n), "BirthYear": rng.integers(1950, 1987, n),
        "reg_to_first_bet": days, "reg_to_first_dep": days,
    })
    df["y_turnover"] = stake * rng.gamma(2, 2, n) * (active / 30)
    df["y_ggr"] = df["y_turnover"] * rng.normal(0.08, 0.05, n)
    df["y_retained_90"] = (rng.random(n) < 0.5).astype(int)
    return df


@pytest.fixture
def features_file(tmp_path, monkeypatch):
    path = tmp_path / "features.parquet"
    synthetic_features().to_parquet(path)
    return path
