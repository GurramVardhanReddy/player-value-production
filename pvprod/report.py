"""Results of a replay, from the monitoring database and the MLflow registry."""

from __future__ import annotations

import json
import sqlite3

import pandas as pd
from mlflow.tracking import MlflowClient
from scipy.stats import spearmanr

from pvprod import registry, steps, timeline


def summary(write: bool = True) -> dict:
    registry.setup(steps.STATE)
    con = sqlite3.connect(steps.STATE / "monitor.db")
    q = lambda sql: pd.read_sql_query(sql, con)  # noqa: E731
    scores = q("select batch, model_version, count(*) n from scores group by 1, 2 order by 1")
    perf = q("select * from performance order by batch")
    drift = q("select batch, feature, psi, level from drift")
    events = q("select * from events order by run_date")

    score_psi = drift[drift["feature"] == "__score__"].set_index("batch")["psi"]
    feat = drift[drift["feature"] != "__score__"]
    worst = feat.sort_values("psi", ascending=False).groupby("batch").head(1).set_index("batch")
    per_batch = perf.set_index("batch").join(score_psi.rename("score_psi")).join(
        worst[["feature", "psi"]].rename(columns={"feature": "top_drift_feature",
                                                   "psi": "top_drift_psi"}))
    per_batch["players"] = scores.set_index("batch")["n"]

    trained = [json.loads(d) for d in events.loc[events["kind"] == "trained", "detail"]]
    promoted = events[events["kind"] == "promoted"]
    rho = spearmanr(per_batch["top_drift_psi"], per_batch["top10_ggr_share"]).statistic

    out = {
        "weekly_runs": len(timeline.run_dates(steps.load_features())),
        "runs_with_activity": int(events["run_date"].nunique()),
        "production_batches": int(len(per_batch)),
        "players_scored": int(scores["n"].sum()),
        "scored_by_versions": sorted(scores["model_version"].unique().tolist()),
        "drift": {
            "batches_with_feature_alert": int(feat[feat["level"] == "alert"]["batch"].nunique()),
            "batches_with_score_alert": int((score_psi > 0.25).sum()),
            "alert_features": feat[feat["level"] == "alert"]["feature"].value_counts().to_dict(),
        },
        "performance": {
            "top10_ggr_share_first_4_batches": round(float(per_batch["top10_ggr_share"].head(4).mean()), 4),
            "top10_ggr_share_last_4_batches": round(float(per_batch["top10_ggr_share"].tail(4).mean()), 4),
            "turnover_total_ratio_range": [round(float(per_batch["turnover_total_ratio"].min()), 3),
                                           round(float(per_batch["turnover_total_ratio"].max()), 3)],
            "spearman_top_drift_psi_vs_top10_share": round(float(rho), 3),
        },
        "retraining": {
            "challengers_trained": len(trained),
            "promotions_after_bootstrap": int(len(promoted) - 1),
            "decisions": [{k: d[k] for k in ("challenger", "champion", "train_rows", "holdout_rows",
                                              "promoted")}
                          | {"challenger_top10": round(d["challenger_metrics"]["top10_ggr_share"], 4),
                             "champion_top10": round(d["champion_metrics"]["top10_ggr_share"], 4),
                             "challenger_total_ratio":
                                 round(d["challenger_metrics"]["turnover_total_ratio"], 3),
                             "champion_total_ratio":
                                 round(d["champion_metrics"]["turnover_total_ratio"], 3)}
                          for d in trained],
        },
        "registry_versions": len(MlflowClient().search_model_versions(f"name='{registry.MODEL_NAME}'")),
        "final_champion": registry.champion_version(),
    }
    if write:
        res = steps.ROOT / "results"
        res.mkdir(exist_ok=True)
        (res / "replay_summary.json").write_text(json.dumps(out, indent=2))
        per_batch.round(4).reset_index().to_csv(res / "batches.csv", index=False)
    con.close()
    return out

