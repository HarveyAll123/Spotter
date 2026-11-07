"""Calendar features.

These describe the calendar rather than a specific date, so they still mean
something in Nov-Dec, which is outside the training period.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def years_since_start(dates: pd.Series) -> np.ndarray:
    """Elapsed time, used for the steady upward drift in rates."""
    return (dates - config.TIME_ORIGIN).dt.days.to_numpy() / 365.0


def quarter_end_progress(dates: pd.Series) -> np.ndarray:
    """0 until 30 days before quarter end, then rising to 1 on the last day.

    Rates climb through the last month of each quarter and drop back on the
    1st. The same pattern appears in March, June and September, so expressing
    it as progress towards quarter end carries it over to December.
    """
    quarter_end = dates + pd.offsets.QuarterEnd(0)
    days_left = (quarter_end - dates).dt.days.to_numpy()
    return np.clip(30 - days_left, 0, 30) / 30.0
