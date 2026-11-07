"""Exploration results used in the report: data-quality facts and figures."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .data import clean_features, market_steps
from .evaluate import FOLDS
from .features import quarter_end_progress
from .model import EQUIPMENT, StructuralRateModel

INK = "#0b0b0b"
INK_SOFT = "#52514e"
RULE = "#e4e3df"
NEUTRAL = "#b9b8b2"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SERIES = {"Dry Van": BLUE, "Reefer": ORANGE, "Flatbed": AQUA}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _style(axis: plt.Axes) -> None:
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(NEUTRAL)
    axis.tick_params(colors=INK_SOFT, length=0, labelsize=8.5)
    axis.grid(axis="y", color=RULE, linewidth=0.7)
    axis.set_axisbelow(True)


def repair_evidence(frame: pd.DataFrame, labeled: pd.DataFrame, label_error: pd.Series) -> dict:
    """Price error on repaired rows, using a model fitted on untouched rows only."""
    normal = (labeled["weight"] > 0) & labeled["market_index"].notna() & ~label_error
    model = StructuralRateModel().fit(frame[normal])

    def spread(rows: pd.DataFrame) -> float:
        return float(np.log(rows[config.TARGET] / model.predict(rows)).std())

    negative = frame[(labeled["weight"] < 0) & ~label_error]
    return {
        "normal": spread(frame[normal]),
        "negative_flipped": spread(negative),
        "negative_median_filled": spread(negative.assign(weight_clean=labeled["weight"].abs().median())),
        "missing_weight": spread(frame[labeled["weight"].isna() & ~label_error]),
        "missing_market": spread(frame[labeled["market_index"].isna() & ~label_error]),
    }


def data_quality_facts(
    labeled: pd.DataFrame, validation: pd.DataFrame, label_error: pd.Series, model: StructuralRateModel,
    market_by_day: pd.Series,
) -> dict:
    """Every data-quality number quoted in the report."""
    frame = clean_features(labeled, market_by_day, labeled["weight"].abs().median())
    multiplier = (frame[config.TARGET] / model.predict(frame))[label_error]
    residual = np.log(frame[config.TARGET] / model.predict(frame)).abs()
    within_day = labeled["market_index"] - labeled["date"].map(market_by_day)
    known = set(labeled["pickup"]) | set(labeled["delivery"])
    new_cities = sorted((set(validation["pickup"]) | set(validation["delivery"])) - known)
    cold = validation["pickup"].isin(new_cities) | validation["delivery"].isin(new_cities)
    seen_lanes = set(zip(labeled["pickup"], labeled["delivery"]))
    lane_seen = [lane in seen_lanes for lane in zip(validation["pickup"], validation["delivery"])]

    def share(mask: pd.Series) -> dict:
        return {"count": int(mask.sum()), "pct": float(mask.mean() * 100)}

    return {
        "rows": {"train": len(labeled), "validation": len(validation)},
        "label_errors": {
            **share(label_error),
            "high_count": int((multiplier > 1).sum()),
            "low_count": int((multiplier < 1).sum()),
            "high_range": [float(multiplier[multiplier > 1].min()), float(multiplier.max())],
            "low_range": [float(multiplier.min()), float(multiplier[multiplier < 1].max())],
        },
        "negative_weight": {
            "train": share(labeled["weight"] < 0),
            "validation": share(validation["weight"] < 0),
        },
        "missing_weight": {
            "train": share(labeled["weight"].isna()),
            "validation": share(validation["weight"].isna()),
        },
        "weight_at_cap": {
            "train": share(labeled["weight"].abs() == 47_500),
            "validation": share(validation["weight"].abs() == 47_500),
        },
        "missing_market": {
            "train": share(labeled["market_index"].isna()),
            "validation": share(validation["market_index"].isna()),
        },
        "new_cities": new_cities,
        "cold_loads": share(cold),
        "validation_lane_seen_pct": float(np.mean(lane_seen) * 100),
        "duplicate_ids": int(labeled["load_id"].duplicated().sum() + validation["load_id"].duplicated().sum()),
        "largest_clean_residual": float(residual[~label_error].max()),
        "smallest_corrupted_residual": float(residual[label_error].min()),
        "market_sd_within_day": float(within_day.std()),
        "market_sd_between_days": float(market_by_day.std()),
        "market_steps": {
            day.strftime("%b %d").replace(" 0", " "): float(size)
            for day, size in market_steps(market_by_day).items()
        },
        "repairs": repair_evidence(frame, labeled, label_error),
    }


def quote_signal_by_month(
    labeled: pd.DataFrame, validation: pd.DataFrame, label_error: pd.Series, predicted_validation: np.ndarray
) -> pd.DataFrame:
    """Rank correlation of quote_signal with rate per mile, for all 12 months.

    Nov-Dec have no labels, so those months use the model prediction. The model
    never sees quote_signal, which makes this a fair label-free check.
    """
    clean = labeled[~label_error]
    rows = []
    for month, group in clean.groupby(clean["date"].dt.month):
        per_mile = group[config.TARGET] / group["distance"]
        rows.append({"month": month, "correlation": group["quote_signal"].corr(per_mile, method="spearman")})
    per_mile = pd.Series(predicted_validation, index=validation.index) / validation["distance"]
    for month, group in validation.groupby(validation["date"].dt.month):
        rows.append(
            {"month": month, "correlation": group["quote_signal"].corr(per_mile[group.index], method="spearman")}
        )
    return pd.DataFrame(rows)


def plot_quote_signal(by_month: pd.DataFrame, path) -> None:
    figure, axis = plt.subplots(figsize=(6.9, 1.7), dpi=220)
    values = by_month["correlation"].to_numpy()
    colors = [BLUE if v > 0.5 else ORANGE if v < -0.5 else NEUTRAL for v in values]
    axis.bar(range(12), values, color=colors, width=0.72)
    axis.axhline(0, color=INK_SOFT, linewidth=0.8)
    axis.axvline(9.5, color=INK_SOFT, linewidth=0.8, linestyle=(0, (3, 3)))
    axis.set_xticks(range(12), MONTHS)
    axis.set_ylim(-1.12, 1.25)
    axis.set_yticks([-1, 0, 1], ["-1", "0", "+1"])
    _style(axis)
    axis.spines["bottom"].set_visible(False)
    axis.tick_params(axis="x", pad=2)
    axis.set_ylabel("Correlation with\nrate per mile", color=INK_SOFT, fontsize=8.5)
    axis.text(7, 0.1, "unrelated", ha="center", va="bottom", fontsize=8, color=INK_SOFT)
    axis.text(10.5, 0.1, "unrelated", ha="center", va="bottom", fontsize=8, color=INK_SOFT)
    axis.text(10.5, 1.2, "validation period", ha="center", va="top", fontsize=8, color=INK_SOFT)
    figure.tight_layout(pad=0.4)
    figure.savefig(path)
    plt.close(figure)


def plot_quarter_end_ramp(
    frame: pd.DataFrame, label_error: pd.Series, model: StructuralRateModel, path
) -> None:
    """Average price by day, with every model effect except the ramp removed."""
    clean = frame[~label_error]
    residual = np.log(clean[config.TARGET] / model.predict(clean))
    amplitude = clean["equipment"].map(model.effect_summary()["ramp_end_of_quarter"])
    ramp_only = residual + amplitude * quarter_end_progress(clean["date"])
    daily = ramp_only.groupby([clean["date"], clean["equipment"]]).mean().unstack() * 100
    smooth = daily.rolling(3, center=True, min_periods=1).mean()

    figure, axis = plt.subplots(figsize=(6.9, 2.3), dpi=220)
    for boundary in ("2025-04-01", "2025-07-01", "2025-10-01"):
        axis.axvline(pd.Timestamp(boundary), color=NEUTRAL, linewidth=0.8, linestyle=(0, (3, 3)))
    for kind in EQUIPMENT:
        axis.plot(smooth.index, smooth[kind], color=SERIES[kind], linewidth=1.5, label=kind)
    _style(axis)
    axis.set_ylabel("Price effect (%)", color=INK_SOFT, fontsize=8.5)
    axis.set_xticks(pd.date_range("2025-01-01", "2025-10-01", freq="MS"), MONTHS[:10])
    axis.set_xlim(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-10-31"))
    axis.set_ylim(-2, 10)
    axis.set_yticks([0, 4, 8])
    axis.legend(frameon=False, fontsize=8, ncol=3, loc="upper left", labelcolor=INK_SOFT, handlelength=1.6)
    summary = model.effect_summary()["ramp_end_of_quarter"]
    for kind in EQUIPMENT:
        lift = (np.exp(summary[kind]) - 1) * 100
        axis.text(
            pd.Timestamp("2025-10-03"), summary[kind] * 100, f"{kind} +{lift:.1f}%",
            fontsize=7.5, color=INK_SOFT, va="center",
        )
    figure.tight_layout(pad=0.4)
    figure.savefig(path)
    plt.close(figure)


def plot_folds(path) -> None:
    """Show which months each backtest fold trains and tests on."""
    figure, axis = plt.subplots(figsize=(6.9, 1.95), dpi=220)
    gap = 0.06
    rows = [
        (fold.name, pd.Timestamp(fold.train_end).month, pd.Timestamp(fold.test_start).month,
         pd.Timestamp(fold.test_end).month)
        for fold in FOLDS
    ]
    names = []
    for row, (name, train_end, test_start, test_end) in enumerate(rows):
        names.append(f"Fold {row + 1}")
        axis.barh(row, train_end - gap, left=0, height=0.62, color=NEUTRAL)
        axis.barh(row, test_end - test_start + 1 - gap, left=test_start - 1, height=0.62, color=BLUE)
    final_row = len(rows)
    names.append("Final")
    axis.barh(final_row, 10 - gap, left=0, height=0.62, color=NEUTRAL, label="Train")
    axis.barh(final_row, 2 - gap, left=10, height=0.62, color=ORANGE, label="Predict (validation.csv)")
    axis.barh(0, 0, color=BLUE, label="Test")
    axis.set_yticks(range(final_row + 1), names)
    axis.invert_yaxis()
    axis.set_xticks(np.arange(12) + 0.5, MONTHS)
    axis.set_xlim(0, 12)
    axis.xaxis.tick_top()
    axis.spines[["top", "right", "left", "bottom"]].set_visible(False)
    axis.tick_params(colors=INK_SOFT, length=0, labelsize=8.5)
    handles, labels = axis.get_legend_handles_labels()
    order = [labels.index(name) for name in ("Train", "Test", "Predict (validation.csv)")]
    axis.legend(
        [handles[i] for i in order], [labels[i] for i in order], frameon=False, fontsize=8, ncol=3,
        loc="upper center", bbox_to_anchor=(0.5, -0.01), labelcolor=INK_SOFT, handlelength=1.2,
    )
    figure.tight_layout(pad=0.4)
    figure.savefig(path)
    plt.close(figure)
