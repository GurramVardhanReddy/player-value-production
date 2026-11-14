import importlib

from fastapi.testclient import TestClient

PLAYER = {
    "player_id": 7, "first_bet_date": "2005-03-01", "stake_90": 900.0, "stake_live_90": 100.0,
    "win_90": 820.0, "bets_90": 60, "active_days_90": 20, "last_active_day": 85,
    "stake_wk1": 50.0, "stake_m1": 300.0, "stake_m3": 400.0, "active_days_m3": 9,
    "active_days_last14": 4, "max_day_stake": 120.0, "max_7d_stake": 300.0, "half_stake_day": 40,
    "live_days_90": 5, "mean_gap": 4.0, "max_gap": 12.0, "sd_gap": 3.0, "country_id": 276,
    "gender": 0, "birth_year": 1978, "reg_to_first_bet": 3, "reg_to_first_dep": 1,
}


def test_score_endpoint(features_file, tmp_path, monkeypatch):
    import joblib
    import pandas as pd

    from pvprod.features import add_features
    from pvprod.model import train_value_model

    df = add_features(pd.read_parquet(features_file))
    joblib.dump(train_value_model(df), tmp_path / "model.joblib")
    (tmp_path / "metadata.json").write_text('{"version": "9", "value_deciles": [1,2,3,4,5,6,7,8,9]}')
    monkeypatch.setenv("PVP_MODEL_DIR", str(tmp_path))
    from pvprod import serve
    importlib.reload(serve)

    client = TestClient(serve.app)
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/model").json()["version"] == "9"
    out = client.post("/score", json=[PLAYER]).json()
    assert out[0]["player_id"] == 7 and out[0]["expected_turnover"] >= 0
    assert 1 <= out[0]["value_band"] <= 10
    assert client.post("/score", json=[{**PLAYER, "active_days_90": 0}]).status_code == 422
