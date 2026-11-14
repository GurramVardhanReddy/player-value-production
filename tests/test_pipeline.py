"""Runs the real pipeline steps (MLflow registry, scoring, drift, grading, retrain) on synthetic data."""

import importlib
import json
import sqlite3


def test_end_to_end(features_file, tmp_path, monkeypatch):
    monkeypatch.setenv("PVP_FEATURES", str(features_file))
    monkeypatch.setenv("PVP_STATE", str(tmp_path / "state"))
    monkeypatch.setenv("PVP_SERVING_DIR", str(tmp_path / "serving"))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    from pvprod import steps, timeline
    importlib.reload(steps)

    assert steps.bootstrap() == "1"
    df = steps.load_features()
    for d in timeline.run_dates(df):
        steps.weekly(d)
    meta = steps.export_champion()

    con = sqlite3.connect(tmp_path / "state" / "monitor.db")
    scored = con.execute("select count(*) from scores").fetchone()[0]
    graded = con.execute("select count(*) from performance").fetchone()[0]
    trained = con.execute("select count(*) from events where kind = 'trained'").fetchone()[0]
    prod = df[df["batch"] >= timeline.BOOTSTRAP_BEFORE]
    assert scored == len(prod)                      # every production player scored once
    assert graded == prod["batch"].nunique()        # every batch graded when it matured
    assert trained >= 1                             # at least one monthly retrain happened
    assert json.loads((tmp_path / "serving" / "metadata.json").read_text()) == meta
