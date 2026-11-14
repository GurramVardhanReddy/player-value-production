"""The value model: next-year turnover (gradient boosting, Poisson loss) times house margin.

The house margin is measured on the training players' first 90 days only, so a model never
uses information from after its training date. (The research repo split margin by fixed-odds
and live betting using full histories; the feature table here does not split winnings by product.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from pvprod.features import CATEGORICAL, FEATURES


@dataclass
class ValueModel:
    turnover: HistGradientBoostingRegressor
    margin: float

    def predict_turnover(self, x: pd.DataFrame) -> np.ndarray:
        return self.turnover.predict(x[FEATURES])

    def predict_value(self, x: pd.DataFrame) -> np.ndarray:
        """Expected next-year GGR (gross gaming revenue) = predicted turnover x house margin."""
        return self.predict_turnover(x) * self.margin


def house_margin(train: pd.DataFrame) -> float:
    return float(1 - train["win_90"].sum() / train["stake_90"].sum())


def train_value_model(train: pd.DataFrame, seed: int = 0) -> ValueModel:
    model = HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=30,
        categorical_features=[FEATURES.index(c) for c in CATEGORICAL], random_state=seed,
    )
    model.fit(train[FEATURES], train["y_turnover"])
    return ValueModel(model, house_margin(train))


def evaluate(model: ValueModel, df: pd.DataFrame) -> dict:
    """Ranking and calibration on players whose next-year outcome is known."""
    pred = model.predict_value(df)
    actual = df["y_ggr"].to_numpy()
    turn_pred, turn_actual = model.predict_turnover(df), df["y_turnover"].to_numpy()
    k = max(1, round(len(df) * 0.10))
    top = np.argsort(-pred, kind="stable")[:k]
    return {
        "n": int(len(df)),
        "top10_ggr_share": float(actual[top].sum() / actual.sum()) if actual.sum() else float("nan"),
        "turnover_gini": _gini(turn_pred, turn_actual),
        "turnover_total_ratio": float(turn_pred.sum() / turn_actual.sum()),
    }


def _gini(pred: np.ndarray, actual: np.ndarray) -> float:
    def g(order):
        a = actual[np.argsort(-order, kind="stable")]
        cum = np.cumsum(a) / a.sum()
        n = len(a)
        return float(cum.sum() / n - (n + 1) / (2 * n))
    return g(pred) / g(actual)
