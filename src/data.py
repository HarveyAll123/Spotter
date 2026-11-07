"""Load the raw files and repair the feature problems found during EDA.

`quote_signal` is deliberately never used. It tracks the target at +0.99 in
some months and -0.99 in others, and carries no information in Nov-Dec, so it
would teach a model a relationship that does not hold at prediction time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Ordinary weekly changes in the index stay within 0.01 of normal, and the three
# real steps are all above 0.10, so this threshold sits in a wide empty gap.
MARKET_STEP_THRESHOLD = 0.08


def load_labeled() -> pd.DataFrame:
    return pd.read_csv(config.TRAIN_PATH, parse_dates=["date"])


def load_validation() -> pd.DataFrame:
    return pd.read_csv(config.VALIDATION_PATH, parse_dates=["date"])


def load_december() -> pd.DataFrame:
    return pd.read_csv(config.DECEMBER_PATH, parse_dates=["date"])


def daily_market_index(*frames: pd.DataFrame) -> pd.Series:
    """Average market index per date.

    The index varies by only 0.025 between loads on the same day, against 0.17
    between days, so this is a good stand-in where a load has no value. It uses
    features only, so no labels are involved.
    """
    combined = pd.concat([frame[["date", "market_index"]] for frame in frames])
    return combined.groupby("date")["market_index"].mean()


def market_steps(market_by_day: pd.Series, threshold: float = MARKET_STEP_THRESHOLD) -> pd.Series:
    """Find abrupt jumps in the index and return their sizes, indexed by date.

    Comparing each day with the same weekday a week earlier cancels the weekly
    cycle. A step then shows up as seven days in a row of unusually large
    change, measured against the typical change over the surrounding month.
    """
    weekly_change = market_by_day.sort_index().diff(7)
    typical = weekly_change.rolling(29, center=True, min_periods=8).median()
    excess = (weekly_change - typical).dropna()

    is_step = excess.abs() > threshold
    run_id = (is_step != is_step.shift()).cumsum()
    runs = excess[is_step].groupby(run_id[is_step])
    return pd.Series(runs.mean().to_numpy(), index=runs.apply(lambda run: run.index[0])).rename("size")


def cumulative_market_steps(dates: pd.Series, steps: pd.Series) -> np.ndarray:
    """Total size of all index steps that have happened by each date."""
    total = np.zeros(len(dates))
    for day, size in steps.items():
        total += (dates >= day).to_numpy() * size
    return total


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """Lookup of city to lat/lon. The coordinates are unique per city."""
    stops = []
    for frame in frames:
        for side in ("pickup", "delivery"):
            stops.append(
                frame[[side, f"{side}_lat", f"{side}_lon"]].set_axis(
                    ["city", "lat", "lon"], axis=1
                )
            )
    return pd.concat(stops).drop_duplicates("city").set_index("city")


def add_coordinates(frame: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    """Attach coordinates to a frame that has city names only (December file)."""
    result = frame.copy()
    for side in ("pickup", "delivery"):
        result[f"{side}_lat"] = result[side].map(cities["lat"])
        result[f"{side}_lon"] = result[side].map(cities["lon"])
    return result


def clean_features(
    frame: pd.DataFrame, market_by_day: pd.Series, weight_fill: float
) -> pd.DataFrame:
    """Return a copy with repaired `weight_clean` and `market_clean` columns.

    Weight: negative values are sign errors, because their absolute value fits
    the price as well as a normal weight does. Missing weights take the median.
    Market index: missing values take the mean for that date, and
    `market_step_total` adds the running total of abrupt index steps.
    """
    result = frame.copy()
    result["weight_clean"] = result["weight"].abs().fillna(weight_fill)
    if "market_index" in result:
        known = result["market_index"]
    else:
        known = pd.Series(np.nan, index=result.index)
    result["market_clean"] = known.fillna(result["date"].map(market_by_day))
    result["market_step_total"] = cumulative_market_steps(result["date"], market_steps(market_by_day))
    return result
