"""The production steps the Airflow DAG runs, one function per task.

State lives in two places under PVP_STATE: MLflow (runs, registry, champion alias) and a SQLite
database of scores, drift results, performance results and pipeline events.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd

from pvprod import monitor, registry, timeline
from pvprod.features import FEATURES, add_features
from pvprod.model import evaluate, train_value_model

ROOT = Path(__file__).resolve().parent.parent
STATE = Path(os.environ.get("PVP_STATE", ROOT / "state"))
FEATURE_TABLE = Path(os.environ.get("PVP_FEATURES", ROOT / "data" / "features.parquet"))
SERVING_DIR = Path(os.environ.get("PVP_SERVING_DIR", ROOT / "serving" / "model"))

# Retraining: on the first run of each month, if new outcomes have arrived since the last training.
# A challenger replaces the champion only if it captures at least as much next-year GGR in its
# top 10% on the newest matured batches (which neither model trained on) and its total turnover
# forecast is no further from actual.
HOLDOUT_WEEKS = 4


def _db() -> sqlite3.Connection:
    STATE.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(STATE / "monitor.db")
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS scores (player_id INTEGER, batch TEXT, scored_on TEXT,
            model_version TEXT, pred_value REAL, pred_turnover REAL,
            PRIMARY KEY (player_id));
        CREATE TABLE IF NOT EXISTS drift (batch TEXT, run_date TEXT, model_version TEXT,
            feature TEXT, psi REAL, level TEXT, PRIMARY KEY (batch, feature));
        CREATE TABLE IF NOT EXISTS performance (batch TEXT PRIMARY KEY, matured_on TEXT,
            model_version TEXT, n INTEGER, top10_ggr_share REAL, turnover_gini REAL,
            turnover_total_ratio REAL);
        CREATE TABLE IF NOT EXISTS events (run_date TEXT, kind TEXT, detail TEXT);
        """
    )
    return con


def _event(con, run_date: date, kind: str, detail: dict) -> None:
    con.execute("INSERT INTO events VALUES (?, ?, ?)", [str(run_date), kind, json.dumps(detail)])
    con.commit()


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(FEATURE_TABLE)
    df = timeline.assign_batches(add_features(df))
    return df.rename(columns={"UserID": "player_id"})


def bootstrap() -> str:
    """Train and register the first champion on the stand-in prior cohort (first two weeks)."""
    registry.setup(STATE)
    if registry.champion_version():
        return registry.champion_version()
    df = load_features()
    train = df[df["batch"] < timeline.BOOTSTRAP_BEFORE]
    model = train_value_model(train)
    version = registry.log_and_register(
        model, train[FEATURES],
        params={"train_rows": len(train), "train_batches": train["batch"].nunique(),
                "trained_through": str(max(train["batch"])), "margin": round(model.margin, 4)},
        metrics={}, tags={"reason": "bootstrap"})
    registry.promote(version)
    with _db() as con:
        _event(con, timeline.BOOTSTRAP_BEFORE, "promoted",
               {"version": version, "reason": "bootstrap", "train_rows": len(train)})
    return version


def score_new_batches(run_date: date) -> list[str]:
    registry.setup(STATE)
    df = load_features()
    batches = timeline.due(df["batch"].unique(), run_date, timeline.scoreable_on)
    if not batches:
        return []
    version = registry.champion_version()
    model, reference = registry.load(version)
    ref_score = model.predict_value(reference)
    con = _db()
    for b in batches:
        cur = df[df["batch"] == b]
        pred_value, pred_turn = model.predict_value(cur), model.predict_turnover(cur)
        con.executemany(
            "INSERT OR REPLACE INTO scores VALUES (?, ?, ?, ?, ?, ?)",
            [(int(p), str(b), str(run_date), version, float(v), float(t))
             for p, v, t in zip(cur["player_id"], pred_value, pred_turn, strict=True)])
        rep = monitor.drift_report(reference, cur[FEATURES], ref_score, pred_value)
        con.executemany(
            "INSERT OR REPLACE INTO drift VALUES (?, ?, ?, ?, ?, ?)",
            [(str(b), str(run_date), version, r.feature, r.psi, r.level) for r in rep.itertuples()])
        flagged = rep[rep["level"] != "ok"]
        _event(con, run_date, "scored", {"batch": str(b), "players": len(cur), "version": version,
                                         "drift": flagged[["feature", "psi", "level"]]
                                         .round(3).to_dict("records")})
    con.commit()
    con.close()
    return [str(b) for b in batches]


def evaluate_matured(run_date: date) -> list[dict]:
    """Grade each batch whose outcome just arrived, with the version that actually scored it."""
    registry.setup(STATE)
    df = load_features()
    batches = timeline.due(df["batch"].unique(), run_date, timeline.matured_on)
    con = _db()
    out = []
    for b in batches:
        (version,) = con.execute("SELECT model_version FROM scores WHERE batch = ? LIMIT 1",
                                 [str(b)]).fetchone()
        model, _ = registry.load(version)
        m = evaluate(model, df[df["batch"] == b])
        con.execute("INSERT OR REPLACE INTO performance VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [str(b), str(timeline.matured_on(b)), version, m["n"], m["top10_ggr_share"],
                     m["turnover_gini"], m["turnover_total_ratio"]])
        out.append({"batch": str(b), "version": version, **m})
        _event(con, run_date, "evaluated", out[-1])
    con.commit()
    con.close()
    return out


def maybe_retrain(run_date: date) -> dict:
    """First run of the month: train a challenger if outcomes arrived since the last training."""
    if run_date.day > 7:
        return {"action": "skip", "why": "not the first run of the month"}
    registry.setup(STATE)
    df = load_features()
    matured = sorted(b for b in df["batch"].unique()
                     if b >= timeline.BOOTSTRAP_BEFORE and timeline.matured_on(b) <= run_date)
    if len(matured) <= HOLDOUT_WEEKS:
        return {"action": "skip", "why": f"{len(matured)} matured batches, need more than {HOLDOUT_WEEKS}"}
    holdout_batches = matured[-HOLDOUT_WEEKS:]
    train = df[(df["batch"] < timeline.BOOTSTRAP_BEFORE) | df["batch"].isin(matured[:-HOLDOUT_WEEKS])]
    holdout = df[df["batch"].isin(holdout_batches)]
    champ_version = registry.champion_version()
    champion, _ = registry.load(champ_version)
    con = _db()
    last = con.execute("SELECT detail FROM events WHERE kind = 'trained' ORDER BY run_date DESC "
                       "LIMIT 1").fetchone()
    if last and json.loads(last[0])["train_rows"] == len(train):
        con.close()
        return {"action": "skip", "why": "no new outcomes since the last training"}
    challenger = train_value_model(train)
    c_m, ch_m = evaluate(challenger, holdout), evaluate(champion, holdout)
    better_rank = c_m["top10_ggr_share"] >= ch_m["top10_ggr_share"]
    better_total = abs(c_m["turnover_total_ratio"] - 1) <= abs(ch_m["turnover_total_ratio"] - 1)
    version = registry.log_and_register(
        challenger, train[FEATURES],
        params={"train_rows": len(train), "train_batches": train["batch"].nunique(),
                "trained_through": str(max(train["batch"])), "margin": round(challenger.margin, 4),
                "holdout_batches": f"{holdout_batches[0]}..{holdout_batches[-1]}"},
        metrics={f"holdout_{k}": v for k, v in c_m.items()},
        tags={"reason": "monthly retrain", "run_date": str(run_date)})
    decision = {"challenger": version, "champion": champ_version, "train_rows": len(train),
                "holdout_rows": len(holdout), "challenger_metrics": c_m, "champion_metrics": ch_m,
                "promoted": bool(better_rank and better_total)}
    _event(con, run_date, "trained", decision)
    if decision["promoted"]:
        registry.promote(version)
        _event(con, run_date, "promoted", {"version": version, "reason": "beat champion on holdout"})
    con.close()
    return decision


def export_champion() -> dict:
    """Write the champion and its metadata where the scoring service image picks it up."""
    registry.setup(STATE)
    version = registry.champion_version()
    model, reference = registry.load(version)
    SERVING_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SERVING_DIR / "model.joblib")
    # value-band edges for the API: 10th..90th percentiles of the champion's training players
    deciles = pd.Series(model.predict_value(reference)).quantile([i / 10 for i in range(1, 10)])
    meta = {"model": registry.MODEL_NAME, "version": version, "margin": model.margin,
            "value_deciles": [round(float(v), 4) for v in deciles],
            "exported_on": str(date.today())}
    (SERVING_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def weekly(run_date: date) -> dict:
    """Everything one weekly run does, in order (used by tests and the plain-Python replay)."""
    return {"scored": score_new_batches(run_date), "evaluated": evaluate_matured(run_date),
            "retrain": maybe_retrain(run_date)}


def previous_monday(d: date) -> date:
    return d - timedelta(days=d.weekday())
