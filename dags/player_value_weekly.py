"""Weekly player value pipeline.

Each Monday: make sure a champion exists, score the players who just reached 90 days with the
champion (and check them for drift), grade batches whose next-year outcome just arrived, retrain on
the first Monday of the month if new outcomes exist (promote only if the challenger wins on a
holdout), and export the champion for the scoring service.

Tasks run in the project's own Python environment (PVP_PYTHON) so Airflow's dependencies and the
model's (MLflow, scikit-learn) never conflict.
"""

from __future__ import annotations

import os
from datetime import datetime

from airflow.sdk import dag, task

PVP_PYTHON = os.environ.get("PVP_PYTHON", "/opt/pvp/venv/bin/python")


@dag(
    dag_id="player_value_weekly",
    schedule="0 6 * * 1",
    start_date=datetime(2005, 5, 23),
    end_date=datetime(2006, 9, 4),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2},
    tags=["player-value", "mlops"],
)
def player_value_weekly():
    @task.external_python(python=PVP_PYTHON, expect_airflow=False)
    def ensure_champion() -> str:
        from pvprod import steps
        return steps.bootstrap()

    @task.external_python(python=PVP_PYTHON, expect_airflow=False)
    def score_new_batches(run_date: str, champion: str) -> list:
        from datetime import date

        from pvprod import steps
        return steps.score_new_batches(date.fromisoformat(run_date))

    @task.external_python(python=PVP_PYTHON, expect_airflow=False)
    def evaluate_matured(run_date: str, scored: list) -> list:
        from datetime import date

        from pvprod import steps
        return steps.evaluate_matured(date.fromisoformat(run_date))

    @task.external_python(python=PVP_PYTHON, expect_airflow=False)
    def maybe_retrain(run_date: str, evaluated: list) -> dict:
        from datetime import date

        from pvprod import steps
        return steps.maybe_retrain(date.fromisoformat(run_date))

    @task.external_python(python=PVP_PYTHON, expect_airflow=False)
    def export_champion(retrain: dict) -> dict:
        from pvprod import steps
        return steps.export_champion()

    @task.external_python(python=PVP_PYTHON, expect_airflow=False)
    def publish_scores(scored: list, exported: dict) -> dict:
        from pvprod import publish
        return publish.publish_scores(scored or None) if scored else {"skipped": "no new batches"}

    champion = ensure_champion()
    scored = score_new_batches("{{ ds }}", champion)
    evaluated = evaluate_matured("{{ ds }}", scored)
    retrain = maybe_retrain("{{ ds }}", evaluated)
    exported = export_champion(retrain)
    publish_scores(scored, exported)


player_value_weekly()
