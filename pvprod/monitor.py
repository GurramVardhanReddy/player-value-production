"""Drift monitoring: population stability index (PSI) of each feature and of the model score.

PSI compares a batch's distribution with the champion's training data, using 10 bins cut at the
training quantiles. Rule of thumb used in credit and marketing models: below 0.1 stable,
0.1 to 0.25 moderate shift, above 0.25 major shift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pvprod.features import CATEGORICAL, FEATURES

PSI_WARN = 0.10
PSI_ALERT = 0.25
EPS = 1e-4


def _numeric_psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _categorical_psi(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref.unique()) | set(cur.unique()))
    r = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    c = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi(ref: pd.Series | np.ndarray, cur: pd.Series | np.ndarray, categorical: bool = False) -> float:
    if categorical:
        return _categorical_psi(pd.Series(ref), pd.Series(cur))
    return _numeric_psi(np.asarray(ref, float), np.asarray(cur, float))


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, ref_score, cur_score) -> pd.DataFrame:
    rows = [{"feature": f, "psi": psi(ref[f], cur[f], f in CATEGORICAL)} for f in FEATURES]
    rows.append({"feature": "__score__", "psi": psi(ref_score, cur_score)})
    out = pd.DataFrame(rows)
    out["level"] = np.select([out["psi"] > PSI_ALERT, out["psi"] > PSI_WARN],
                             ["alert", "warn"], "ok")
    return out
