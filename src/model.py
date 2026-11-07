"""Structural pricing model for freight rates.

Exploration showed that log(rate) is close to additive, with Gaussian noise:

    log(rate) = s(log distance) + equipment + h(weight)
              + g(pickup location) + g(delivery location)
              + lane(pickup -> delivery)
              + beta * market_index + delta * market_step_total + gamma * time
              + quarter-end ramp (by equipment) + noise

Writing that down directly gives two things a tree ensemble cannot do here:
  * Extrapolate in time. Nov-Dec lie beyond the training dates, and trees hold
    their last value flat, so they miss the upward drift.
  * Price unseen cities. g() is a smooth surface over latitude and longitude,
    so a new city is priced from its coordinates.

Lane effects are ridge-penalised, which is the same as treating them as random
effects: a lane with few loads is pulled towards zero, and a lane never seen
before gets exactly zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import solve as dense_solve
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

from . import config
from .features import quarter_end_progress, years_since_start

EQUIPMENT = ("Dry Van", "Reefer", "Flatbed")
FIXED_EFFECT_PENALTY = 1e-8  # numerical stability only
FILTER_PASSES = 3


class StructuralRateModel:
    """Penalised least squares on log(rate). Predicts dollars."""

    def __init__(
        self,
        lane_penalty: float = 2.0,
        geo_degree: int = 4,
        use_market_steps: bool = True,
        use_trend: bool = True,
        use_ramp: bool = True,
        use_lanes: bool = True,
    ):
        self.lane_penalty = lane_penalty
        self.geo_degree = geo_degree
        self.use_market_steps = use_market_steps
        self.use_trend = use_trend
        self.use_ramp = use_ramp
        self.use_lanes = use_lanes

    # ------------------------------------------------------------------ fit
    def fit(self, frame: pd.DataFrame) -> "StructuralRateModel":
        self._learn_transforms(frame)
        log_rate = np.log(frame[config.TARGET].to_numpy())

        blocks = self._numeric_blocks(frame)
        numeric = sp.csr_matrix(np.hstack(list(blocks.values())))
        keep = self._find_clean_labels(numeric, log_rate)
        self.n_label_errors_ = int((~keep).sum())

        design, penalty = numeric, np.full(numeric.shape[1], FIXED_EFFECT_PENALTY)
        if self.use_lanes:
            lanes = self._lane_terms(frame)
            design = sp.hstack([numeric, lanes], format="csr")
            penalty = np.concatenate([penalty, np.full(lanes.shape[1], self.lane_penalty)])
        self.coefficients_ = self._solve(design[keep], log_rate[keep], penalty)

        self.block_slices_, start = {}, 0
        for name, block in blocks.items():
            self.block_slices_[name] = slice(start, start + block.shape[1])
            start += block.shape[1]
        self.lane_effects_ = self.coefficients_[start:]
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        design = sp.csr_matrix(np.hstack(list(self._numeric_blocks(frame).values())))
        if self.use_lanes:
            design = sp.hstack([design, self._lane_terms(frame)], format="csr")
        return np.exp(design @ self.coefficients_)

    def _find_clean_labels(self, numeric: sp.csr_matrix, log_rate: np.ndarray) -> np.ndarray:
        """Drop corrupted labels before the final fit.

        This runs without lane effects because a lane with few loads could
        otherwise absorb part of a bad label.
        """
        penalty = np.full(numeric.shape[1], FIXED_EFFECT_PENALTY)
        keep = np.ones(len(log_rate), dtype=bool)
        for _ in range(FILTER_PASSES):
            coefficients = self._solve(numeric[keep], log_rate[keep], penalty)
            residual = log_rate - numeric @ coefficients
            updated = np.abs(residual) < config.LABEL_ERROR_LOG_THRESHOLD
            if np.array_equal(updated, keep):
                break
            keep = updated
        return keep

    @staticmethod
    def _solve(design: sp.csr_matrix, target: np.ndarray, penalty: np.ndarray) -> np.ndarray:
        normal = (design.T @ design).toarray()
        normal[np.diag_indices_from(normal)] += penalty
        return dense_solve(normal, design.T @ target, assume_a="pos")

    # --------------------------------------------------------------- design
    def _learn_transforms(self, frame: pd.DataFrame) -> None:
        self.distance_spline_ = SplineTransformer(
            n_knots=10, degree=3, include_bias=False, extrapolation="linear"
        ).fit(np.log(frame[["distance"]].to_numpy()))
        self.weight_spline_ = SplineTransformer(
            n_knots=8, degree=3, include_bias=False, extrapolation="constant"
        ).fit(frame[["weight_clean"]].to_numpy())

        coordinates = np.vstack(
            [
                frame[["pickup_lat", "pickup_lon"]].to_numpy(),
                frame[["delivery_lat", "delivery_lon"]].to_numpy(),
            ]
        )
        self.geo_center_ = coordinates.mean(axis=0)
        self.geo_scale_ = coordinates.std(axis=0)
        self.geo_poly_ = PolynomialFeatures(self.geo_degree, include_bias=False).fit(np.zeros((1, 2)))

        lanes = (frame["pickup"] + ">" + frame["delivery"]).unique()
        self.lane_index_ = {lane: i for i, lane in enumerate(sorted(lanes))}

    def _location_surface(self, frame: pd.DataFrame, side: str) -> np.ndarray:
        coordinates = frame[[f"{side}_lat", f"{side}_lon"]].to_numpy()
        return self.geo_poly_.transform((coordinates - self.geo_center_) / self.geo_scale_)

    def _numeric_blocks(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        equipment = frame["equipment"].to_numpy()
        blocks = {
            "intercept": np.ones((len(frame), 1)),
            "distance": self.distance_spline_.transform(np.log(frame[["distance"]].to_numpy())),
            "weight": self.weight_spline_.transform(frame[["weight_clean"]].to_numpy()),
            "reefer": (equipment == "Reefer").astype(float)[:, None],
            "flatbed": (equipment == "Flatbed").astype(float)[:, None],
            "market": (frame["market_clean"].to_numpy() - 1.0)[:, None],
            # One shared surface: pickup and delivery city effects correlate at 0.99.
            "location": self._location_surface(frame, "pickup")
            + self._location_surface(frame, "delivery"),
        }
        if self.use_market_steps:
            blocks["market_steps"] = frame["market_step_total"].to_numpy()[:, None]
        if self.use_trend:
            blocks["trend"] = years_since_start(frame["date"])[:, None]
        if self.use_ramp:
            progress = quarter_end_progress(frame["date"])
            for kind in EQUIPMENT:
                blocks[f"ramp {kind}"] = (progress * (equipment == kind))[:, None]
        return blocks

    def _lane_terms(self, frame: pd.DataFrame) -> sp.csr_matrix:
        """One effect per directed lane. A->B and B->A effects are unrelated."""
        position = (frame["pickup"] + ">" + frame["delivery"]).map(self.lane_index_).to_numpy(dtype=float)
        known = ~np.isnan(position)
        return sp.csr_matrix(
            (np.ones(known.sum()), (np.flatnonzero(known), position[known].astype(int))),
            shape=(len(frame), len(self.lane_index_)),
        )

    # -------------------------------------------------------------- reading
    def effect_summary(self) -> dict:
        """Named coefficients for the report, on the log(rate) scale."""
        value = lambda name: self.coefficients_[self.block_slices_[name]]
        summary = {
            "market_beta": float(value("market")[0]),
            "reefer_premium": float(value("reefer")[0]),
            "flatbed_premium": float(value("flatbed")[0]),
            "label_errors_removed": self.n_label_errors_,
        }
        if self.use_market_steps:
            summary["market_step_extra"] = float(value("market_steps")[0])
        if self.use_trend:
            summary["trend_per_year"] = float(value("trend")[0])
        if self.use_ramp:
            summary["ramp_end_of_quarter"] = {
                kind: float(value(f"ramp {kind}")[0]) for kind in EQUIPMENT
            }
        return summary
