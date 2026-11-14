"""MLflow tracking and model registry.

Every trained model is logged as a run (params, metrics, training window) and as an MLflow pyfunc
model that carries two artifacts: the fitted model and its training-feature reference (used for
drift). Each is registered as a new version of "player-value"; the alias "champion" marks the
version that scores new players.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient

from pvprod.model import ValueModel

MODEL_NAME = "player-value"
CHAMPION = "champion"


class PlayerValuePyfunc(mlflow.pyfunc.PythonModel):
    """Lets MLflow serve the model too: input is the feature table, output is expected GGR."""

    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input: pd.DataFrame, params=None):
        return self.model.predict_value(model_input)


def setup(state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{state_dir / 'mlflow.db'}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    mlflow.set_experiment("player-value-production")


def log_and_register(model: ValueModel, reference: pd.DataFrame, params: dict,
                     metrics: dict, tags: dict) -> str:
    with mlflow.start_run(run_name=tags.get("reason", "train")):
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})  # drop NaN
        mlflow.set_tags(tags)
        with tempfile.TemporaryDirectory() as tmp:
            model_path, ref_path = Path(tmp) / "model.joblib", Path(tmp) / "reference.parquet"
            joblib.dump(model, model_path)
            reference.to_parquet(ref_path)
            info = mlflow.pyfunc.log_model(
                name="player_value", python_model=PlayerValuePyfunc(),
                artifacts={"model": str(model_path), "reference": str(ref_path)},
                input_example=reference.head(3))
        mv = mlflow.register_model(info.model_uri, MODEL_NAME)
    return str(mv.version)


def promote(version: str) -> None:
    MlflowClient().set_registered_model_alias(MODEL_NAME, CHAMPION, version)


def champion_version() -> str | None:
    try:
        return str(MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION).version)
    except mlflow.exceptions.MlflowException:
        return None


def load(version: str) -> tuple[ValueModel, pd.DataFrame]:
    path = Path(mlflow.artifacts.download_artifacts(f"models:/{MODEL_NAME}/{version}"))
    model = joblib.load(next(path.rglob("model.joblib")))
    reference = pd.read_parquet(next(path.rglob("reference.parquet")))
    return model, reference
