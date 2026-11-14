"""CI only: train a throwaway model on synthetic data so the image can be built and smoke-tested.
The deployed image ships the real champion exported by the pipeline instead."""

import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from conftest import synthetic_features  # noqa: E402

from pvprod.features import add_features  # noqa: E402
from pvprod.model import train_value_model  # noqa: E402

out = Path("serving/model")
out.mkdir(parents=True, exist_ok=True)
joblib.dump(train_value_model(add_features(synthetic_features())), out / "model.joblib")
(out / "metadata.json").write_text(json.dumps({"version": "ci-smoke", "synthetic": True}))
