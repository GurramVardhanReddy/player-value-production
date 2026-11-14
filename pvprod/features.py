"""Model features. Same definitions as github.com/GurramVardhanReddy/player-value-model
(pvm/models.py), kept here without the Spark dependency so the scoring service stays small.

Input: one row per player from that repo's Spark SQL feature table (first 90 days of betting).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OBS_DAYS = 90
PRED_DAYS = 365
TOP_COUNTRIES = (276, 792, 616, 724, 300, 40, 380)

RAW_COLUMNS = [
    "t0", "stake_90", "stake_live_90", "win_90", "bets_90", "active_days_90", "last_active_day",
    "stake_wk1", "stake_m1", "stake_m3", "active_days_m3", "active_days_last14", "max_day_stake",
    "max_7d_stake", "half_stake_day", "live_days_90", "mean_gap", "max_gap", "sd_gap",
    "CountryID", "Gender", "BirthYear", "reg_to_first_bet", "reg_to_first_dep",
]

FEATURES = [
    "log_stake_90", "log_stake_wk1", "log_stake_m1", "log_stake_m3", "log_max_day_stake",
    "log_max_7d_stake", "log_stake_per_active_day", "log_stake_per_bet",
    "active_days_90", "active_days_m3", "active_days_last14", "bets_90", "live_days_90",
    "live_share", "payout_ratio_90", "recency", "m3_vs_m1", "peak_day_share",
    "mean_gap", "max_gap", "sd_gap", "half_stake_day",
    "reg_to_first_bet", "reg_to_first_dep", "age_at_first_bet", "Gender", "country",
]
CATEGORICAL = ["country", "Gender"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    eps = 1e-9
    d["age_at_first_bet"] = pd.to_datetime(d["t0"]).dt.year - d["BirthYear"]
    d["live_share"] = d["stake_live_90"] / (d["stake_90"] + eps)
    d["stake_per_active_day"] = d["stake_90"] / d["active_days_90"].clip(lower=1)
    d["stake_per_bet"] = d["stake_90"] / d["bets_90"].clip(lower=1)
    d["payout_ratio_90"] = d["win_90"] / (d["stake_90"] + eps)
    d["recency"] = OBS_DAYS - 1 - d["last_active_day"]
    d["m3_vs_m1"] = np.log1p(d["stake_m3"]) - np.log1p(d["stake_m1"])
    d["peak_day_share"] = d["max_day_stake"] / (d["stake_90"] + eps)
    d["country"] = np.where(d["CountryID"].isin(TOP_COUNTRIES), d["CountryID"], 0)
    for c in ("stake_90", "stake_wk1", "stake_m1", "stake_m3", "max_day_stake",
              "max_7d_stake", "stake_per_active_day", "stake_per_bet"):
        d[f"log_{c}"] = np.log1p(d[c])
    return d
