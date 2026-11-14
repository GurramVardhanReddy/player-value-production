"""Scoring service: POST a player's first-90-day betting summary, get expected next-year value.

The image ships the champion exported by the pipeline (model.joblib + metadata.json). The value
band places the player among the champion's training players (1 = top 10% by expected value).
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pvprod.features import add_features

MODEL_DIR = Path(os.environ.get("PVP_MODEL_DIR", Path(__file__).resolve().parent.parent
                                / "serving" / "model"))


class PlayerSummary(BaseModel):
    player_id: int
    first_bet_date: date
    stake_90: float = Field(ge=0, description="Total stake in the first 90 days")
    stake_live_90: float = Field(ge=0)
    win_90: float = Field(ge=0)
    bets_90: int = Field(ge=0)
    active_days_90: int = Field(ge=1, le=90)
    last_active_day: int = Field(ge=0, le=89)
    stake_wk1: float = Field(ge=0)
    stake_m1: float = Field(ge=0)
    stake_m3: float = Field(ge=0)
    active_days_m3: int = Field(ge=0, le=30)
    active_days_last14: int = Field(ge=0, le=14)
    max_day_stake: float = Field(ge=0)
    max_7d_stake: float = Field(ge=0)
    half_stake_day: int = Field(ge=0, le=89)
    live_days_90: int = Field(ge=0, le=90)
    mean_gap: float | None = None
    max_gap: float | None = None
    sd_gap: float | None = None
    country_id: int
    gender: int = Field(ge=0, le=1)
    birth_year: int | None = None
    reg_to_first_bet: int = Field(ge=0)
    reg_to_first_dep: int | None = None


class Score(BaseModel):
    player_id: int
    expected_turnover: float
    expected_ggr: float
    value_band: int | None


def _load():
    if not (MODEL_DIR / "model.joblib").exists():
        return None, {}
    meta_path = MODEL_DIR / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return joblib.load(MODEL_DIR / "model.joblib"), meta


MODEL, META = _load()
app = FastAPI(title="Player value scoring", version="1.0")


def to_frame(players: list[PlayerSummary]) -> pd.DataFrame:
    df = pd.DataFrame([p.model_dump() for p in players]).rename(columns={
        "first_bet_date": "t0", "country_id": "CountryID", "gender": "Gender",
        "birth_year": "BirthYear"})
    return add_features(df)


def band(values: np.ndarray) -> list[int | None]:
    edges = META.get("value_deciles")
    if not edges:
        return [None] * len(values)
    # edges are the 10th..90th percentiles of training-player value, highest band = 1
    return [int(10 - np.searchsorted(edges, v, side="right")) for v in values]


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if MODEL is not None else "no model loaded"}


@app.get("/model")
def model_info() -> dict:
    if MODEL is None:
        raise HTTPException(503, "no model loaded")
    return META


@app.post("/score", response_model=list[Score])
def score(players: list[PlayerSummary]) -> list[Score]:
    if MODEL is None:
        raise HTTPException(503, "no model loaded")
    if not players:
        return []
    x = to_frame(players)
    turnover, value = MODEL.predict_turnover(x), MODEL.predict_value(x)
    return [Score(player_id=p.player_id, expected_turnover=round(float(t), 2),
                  expected_ggr=round(float(v), 2), value_band=b)
            for p, t, v, b in zip(players, turnover, value, band(value), strict=True)]
