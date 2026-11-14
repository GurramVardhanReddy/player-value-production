"""The replay clock: when each weekly batch of new players can be scored and when its outcome is known.

A batch is the players whose first stake fell in one calendar week (Monday to Sunday).
- Scoreable: once every player in the batch has 90 days of history (week end + 90 days).
- Matured: once the next-year outcome exists (week end + 90 + 365 days).

The first two weeks (1 to 13 Feb 2005) stand in for a previous year's cohort: the first champion
is trained on them. A real team in May 2005 would not yet have their outcomes; that is the one
shortcut in the replay, and it only affects the starting model.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from pvprod.features import OBS_DAYS, PRED_DAYS

BOOTSTRAP_BEFORE = date(2005, 2, 14)


def week_start(d) -> date:
    d = pd.Timestamp(d).date()
    return d - timedelta(days=d.weekday())


def assign_batches(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["batch"] = pd.to_datetime(out["t0"]).map(week_start)
    return out


def scoreable_on(batch: date) -> date:
    return batch + timedelta(days=6 + OBS_DAYS)


def matured_on(batch: date) -> date:
    return batch + timedelta(days=6 + OBS_DAYS + PRED_DAYS)


def run_dates(df: pd.DataFrame) -> list[date]:
    """Weekly Mondays from the first scoreable batch to the last matured one."""
    batches = sorted(b for b in df["batch"].unique() if b >= BOOTSTRAP_BEFORE)
    first = week_start(scoreable_on(batches[0]) + timedelta(days=6))
    last = week_start(matured_on(batches[-1]) + timedelta(days=6))
    out, d = [], first
    while d <= last:
        out.append(d)
        d += timedelta(days=7)
    return out


def due(batches, run_date: date, when) -> list[date]:
    """Batches whose event (scoreable_on or matured_on) fell in the 7 days up to run_date."""
    return sorted(b for b in batches
                  if b >= BOOTSTRAP_BEFORE and run_date - timedelta(days=7) < when(b) <= run_date)
