"""Comparison models. None of these is the final model."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config
from .features import quarter_end_progress, years_since_start

GEO = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
LGBM_PARAMS = dict(
    objective="l1",
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    verbose=-1,
    n_jobs=8,
    random_state=7,
)


def lightgbm_features(frame: pd.DataFrame) -> pd.DataFrame:
    """All candidate columns. Each model below selects from these."""
    features = frame[GEO + ["distance"]].copy()
    features["weight_clean"] = frame["weight_clean"]
    features["market_clean"] = frame["market_clean"]
    features["market_step_total"] = frame["market_step_total"]
    features["equipment_code"] = frame["equipment"].map({"Dry Van": 0, "Reefer": 1, "Flatbed": 2})
    features["day_of_week"] = frame["date"].dt.dayofweek
    features["month"] = frame["date"].dt.month
    features["quote_signal"] = frame["quote_signal"] if "quote_signal" in frame else np.nan
    features["quarter_end_progress"] = quarter_end_progress(frame["date"])
    features["years_since_start"] = years_since_start(frame["date"])
    return features


class LaneMedianBaseline:
    """Median rate per mile on the lane, else across all loads."""

    def fit(self, frame: pd.DataFrame) -> "LaneMedianBaseline":
        per_mile = frame[config.TARGET] / frame["distance"]
        self.overall_ = float(per_mile.median())
        self.by_lane_ = per_mile.groupby([frame["pickup"], frame["delivery"]]).median()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        keys = pd.MultiIndex.from_frame(frame[["pickup", "delivery"]])
        per_mile = self.by_lane_.reindex(keys).to_numpy()
        return np.where(np.isnan(per_mile), self.overall_, per_mile) * frame["distance"].to_numpy()


class LightGBMRate:
    """Gradient boosting on log(rate per mile) with a robust (L1) loss."""

    def __init__(self, columns: list[str], **overrides):
        self.columns = columns
        self.params = {**LGBM_PARAMS, **overrides}

    def fit(self, frame: pd.DataFrame) -> "LightGBMRate":
        target = np.log(frame[config.TARGET] / frame["distance"])
        self.model_ = lgb.LGBMRegressor(**self.params).fit(
            lightgbm_features(frame)[self.columns], target
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        log_per_mile = self.model_.predict(lightgbm_features(frame)[self.columns])
        return np.exp(log_per_mile) * frame["distance"].to_numpy()


class StructuralPlusBoosting:
    """Structural model, then LightGBM on whatever residual structure is left."""

    def __init__(self, structural, columns: list[str], **overrides):
        self.structural = structural
        self.columns = columns
        self.params = {
            **LGBM_PARAMS,
            "objective": "l2",
            "n_estimators": 300,
            "num_leaves": 31,
            "min_child_samples": 50,
            **overrides,
        }

    def fit(self, frame: pd.DataFrame) -> "StructuralPlusBoosting":
        self.structural.fit(frame)
        residual = np.log(frame[config.TARGET].to_numpy() / self.structural.predict(frame))
        clean = np.abs(residual) < config.LABEL_ERROR_LOG_THRESHOLD
        self.model_ = lgb.LGBMRegressor(**self.params).fit(
            lightgbm_features(frame)[self.columns][clean], residual[clean]
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        correction = self.model_.predict(lightgbm_features(frame)[self.columns])
        return self.structural.predict(frame) * np.exp(correction)
