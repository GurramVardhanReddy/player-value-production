from datetime import date

import numpy as np
import pandas as pd

from pvprod import monitor, timeline


def test_batch_dates():
    b = date(2005, 2, 14)  # a Monday
    assert timeline.scoreable_on(b) == date(2005, 5, 21)
    assert timeline.matured_on(b) == date(2006, 5, 21)
    assert timeline.week_start("2005-02-17") == b


def test_due_finds_events_in_the_last_week_only():
    batches = [date(2005, 2, 14), date(2005, 2, 21)]
    assert timeline.due(batches, date(2005, 5, 23), timeline.scoreable_on) == [date(2005, 2, 14)]
    assert timeline.due(batches, date(2005, 5, 16), timeline.scoreable_on) == []
    # bootstrap batches are never scored as production batches
    assert timeline.due([date(2005, 2, 7)], date(2005, 5, 16), timeline.scoreable_on) == []


def test_psi_stable_vs_shifted():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 5000)
    assert monitor.psi(ref, rng.normal(0, 1, 5000)) < monitor.PSI_WARN
    assert monitor.psi(ref, rng.normal(1.5, 1, 5000)) > monitor.PSI_ALERT


def test_categorical_psi():
    ref = pd.Series([1] * 80 + [2] * 20)
    assert monitor.psi(ref, ref, categorical=True) < 1e-9
    assert monitor.psi(ref, pd.Series([1] * 20 + [2] * 80), categorical=True) > monitor.PSI_ALERT
