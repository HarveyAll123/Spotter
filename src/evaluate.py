"""Backtesting that mirrors the validation task.

The validation file differs from the training data in three ways, and each one
is reproduced here:
  1. It is later in time, so every fold trains on the past and tests on the
     next two months, which is the same horizon as Nov-Dec.
  2. 8 cities in it never appear in training, so each repeat removes 8 random
     cities from the training rows. Test loads touching one are scored as "cold".
  3. It is scored on labels we cannot see, so metrics are given on all labels
     and with corrupted labels excluded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import config
from .baselines import GEO, LightGBMRate
from .data import clean_features
from .model import StructuralRateModel


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: str
    test_start: str
    test_end: str


FOLDS = [
    Fold("May-Jun", "2025-04-30", "2025-05-01", "2025-06-30"),
    Fold("Jun-Jul", "2025-05-31", "2025-06-01", "2025-07-31"),
    Fold("Jul-Aug", "2025-06-30", "2025-07-01", "2025-08-31"),
    Fold("Aug-Sep", "2025-07-31", "2025-08-01", "2025-09-30"),
    Fold("Sep-Oct", "2025-08-31", "2025-09-01", "2025-10-31"),
]
HELD_OUT_CITIES = 8


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - actual
    return {
        "mape_pct": float(np.mean(np.abs(error) / actual) * 100),
        "mae_usd": float(np.mean(np.abs(error))),
        "rmse_usd": float(np.sqrt(np.mean(error**2))),
        "r2": float(1 - np.sum(error**2) / np.sum((actual - actual.mean()) ** 2)),
        "bias_pct": float(np.mean(np.log(predicted / actual)) * 100),
    }


def flag_label_errors(labeled: pd.DataFrame, market_by_day: pd.Series) -> pd.Series:
    """Flag corrupted labels. Used for scoring only, never for fold training.

    Every fold repeats this filter on its own training rows, so no test label
    influences a model. The separation is so wide that the flags are the same
    whichever rows the reference model is fitted on.
    """
    frame = clean_features(labeled, market_by_day, labeled["weight"].abs().median())
    model = StructuralRateModel(use_lanes=False).fit(frame)
    residual = np.log(frame[config.TARGET] / model.predict(frame))
    return residual.abs() >= config.LABEL_ERROR_LOG_THRESHOLD


def run_backtest(
    labeled: pd.DataFrame,
    market_by_day: pd.Series,
    builders: dict[str, Callable[[], object]],
    label_error: pd.Series,
    folds: list[Fold] = FOLDS,
    city_seeds: tuple[int, ...] = (11, 22, 33),
) -> pd.DataFrame:
    """Return one row per model, fold, city draw, segment and label set."""
    all_cities = np.array(sorted(set(labeled["pickup"]) | set(labeled["delivery"])))
    rows = []
    for fold in folds:
        for seed in city_seeds:
            held = np.random.default_rng(seed).choice(all_cities, HELD_OUT_CITIES, replace=False)
            pickup_held = labeled["pickup"].isin(held)
            delivery_held = labeled["delivery"].isin(held)

            in_train = (labeled["date"] <= fold.train_end) & ~pickup_held & ~delivery_held
            in_test = labeled["date"].between(fold.test_start, fold.test_end)
            in_test &= ~(pickup_held & delivery_held)

            weight_fill = labeled.loc[in_train, "weight"].abs().median()
            train = clean_features(labeled[in_train], market_by_day, weight_fill)
            test = clean_features(labeled[in_test], market_by_day, weight_fill)
            cold = (test["pickup"].isin(held) | test["delivery"].isin(held)).to_numpy()
            corrupted = label_error[in_test].to_numpy()
            actual = test[config.TARGET].to_numpy()

            for name, build in builders.items():
                predicted = build().fit(train).predict(test)
                masks = {
                    ("warm", "clean"): ~cold & ~corrupted,
                    ("cold", "clean"): cold & ~corrupted,
                    ("warm", "all"): ~cold,
                    ("cold", "all"): cold,
                }
                for (segment, labels), mask in masks.items():
                    rows.append(
                        {
                            "model": name,
                            "fold": fold.name,
                            "seed": seed,
                            "segment": segment,
                            "labels": labels,
                            "n": int(mask.sum()),
                            **regression_metrics(actual[mask], predicted[mask]),
                        }
                    )
    return pd.DataFrame(rows)


def blend_segments(results: pd.DataFrame, cold_share: float) -> pd.DataFrame:
    """Average across folds and weight warm/cold by the validation mix."""
    metric_columns = ["mape_pct", "mae_usd", "rmse_usd", "r2", "bias_pct"]
    by_segment = results.groupby(["model", "labels", "segment"])[metric_columns].mean().reset_index()
    weight = by_segment["segment"].map({"warm": 1 - cold_share, "cold": cold_share})
    blended = by_segment[metric_columns].mul(weight, axis=0)
    blended[["model", "labels"]] = by_segment[["model", "labels"]]
    return blended.groupby(["model", "labels"]).sum().reset_index()


def quote_signal_experiment(
    labeled: pd.DataFrame, market_by_day: pd.Series, label_error: pd.Series
) -> pd.DataFrame:
    """Score the same LightGBM with and without quote_signal under three splits.

    A random split mixes every month into training, so it rewards the leak.
    August is the only labeled month where quote_signal is unrelated to price,
    which is how it behaves in Nov-Dec.
    """
    base = GEO + ["distance", "weight_clean", "market_clean", "equipment_code", "day_of_week", "month"]
    is_random_test = pd.Series(np.random.default_rng(7).random(len(labeled)) < 0.2, index=labeled.index)
    dates = labeled["date"]
    splits = {
        "Random 80/20": (~is_random_test, is_random_test),
        "Train Jan-Aug, test Sep-Oct": (dates <= "2025-08-31", dates > "2025-08-31"),
        "Train Jan-Jul, test Aug": (dates <= "2025-07-31", dates.dt.month == 8),
    }
    weight_fill = labeled["weight"].abs().median()
    rows = []
    for split, (train_mask, test_mask) in splits.items():
        train = clean_features(labeled[train_mask], market_by_day, weight_fill)
        test = clean_features(labeled[test_mask & ~label_error], market_by_day, weight_fill)
        for label, columns in {"with": base + ["quote_signal"], "without": base}.items():
            predicted = LightGBMRate(columns).fit(train).predict(test)
            metrics = regression_metrics(test[config.TARGET].to_numpy(), predicted)
            rows.append({"split": split, "quote_signal": label, **metrics})
    return pd.DataFrame(rows)


def quote_signal_validation_shift(
    labeled: pd.DataFrame, validation: pd.DataFrame, market_by_day: pd.Series
) -> float:
    """Mean % change in validation predictions from adding quote_signal.

    Needs no labels. quote_signal is unrelated to price in Nov-Dec, so any
    movement it causes in the predictions is pure added error.
    """
    columns = GEO + ["distance", "weight_clean", "market_clean", "equipment_code", "day_of_week", "month"]
    weight_fill = labeled["weight"].abs().median()
    train = clean_features(labeled, market_by_day, weight_fill)
    test = clean_features(validation, market_by_day, weight_fill)
    without = LightGBMRate(columns).fit(train).predict(test)
    with_signal = LightGBMRate(columns + ["quote_signal"]).fit(train).predict(test)
    return float(np.mean(np.abs(np.log(with_signal / without))) * 100)
