"""Validate the model and write the result tables to reports/.

Usage: python run_backtest.py
"""
from __future__ import annotations

import pandas as pd

from src import config, data
from src.baselines import GEO, LaneMedianBaseline, LightGBMRate, StructuralPlusBoosting
from src.evaluate import (
    blend_segments,
    flag_label_errors,
    quote_signal_experiment,
    quote_signal_validation_shift,
    run_backtest,
)
from src.model import StructuralRateModel

BASE_COLUMNS = GEO + ["distance", "weight_clean", "market_clean", "equipment_code", "day_of_week"]
RAW_COLUMNS = BASE_COLUMNS + ["month"]
ENGINEERED_COLUMNS = BASE_COLUMNS + ["quarter_end_progress", "years_since_start", "market_step_total"]
RESIDUAL_COLUMNS = BASE_COLUMNS + ["quarter_end_progress"]

MODELS = {
    "Structural (final)": StructuralRateModel,
    "Structural + LightGBM on residuals": lambda: StructuralPlusBoosting(
        StructuralRateModel(), RESIDUAL_COLUMNS, n_estimators=100, num_leaves=15, min_child_samples=200
    ),
    "LightGBM, engineered features": lambda: LightGBMRate(ENGINEERED_COLUMNS),
    "LightGBM, raw features": lambda: LightGBMRate(RAW_COLUMNS),
    "Lane median": LaneMedianBaseline,
    "Structural without quarter-end ramp": lambda: StructuralRateModel(use_ramp=False),
    "Structural without trend": lambda: StructuralRateModel(use_trend=False),
    "Structural without lane effects": lambda: StructuralRateModel(use_lanes=False),
    "Structural without market steps": lambda: StructuralRateModel(use_market_steps=False),
}


def main() -> None:
    labeled = data.load_labeled()
    validation = data.load_validation()
    market_by_day = data.daily_market_index(labeled, validation)
    label_error = flag_label_errors(labeled, market_by_day)

    known_cities = set(labeled["pickup"]) | set(labeled["delivery"])
    is_cold = ~(validation["pickup"].isin(known_cities) & validation["delivery"].isin(known_cities))
    cold_share = float(is_cold.mean())
    print(f"Corrupted labels: {label_error.sum()} ({label_error.mean():.2%})")
    print(f"Validation loads touching an unseen city: {is_cold.sum()} ({cold_share:.2%})")

    results = run_backtest(labeled, market_by_day, MODELS, label_error, city_seeds=(11, 22, 33, 44, 55))
    config.REPORTS_DIR.mkdir(exist_ok=True)
    results.to_csv(config.REPORTS_DIR / "backtest_all_runs.csv", index=False)

    summary = blend_segments(results, cold_share)
    summary.to_csv(config.REPORTS_DIR / "backtest_summary.csv", index=False)

    final = results[results["model"] == "Structural (final)"]
    by_fold = (
        final.groupby(["labels", "fold", "segment"], sort=False)[
            ["n", "mape_pct", "mae_usd", "rmse_usd", "r2", "bias_pct"]
        ]
        .mean()
        .reset_index()
    )
    by_fold.to_csv(config.REPORTS_DIR / "backtest_final_by_fold.csv", index=False)

    bias = results[results["labels"] == "clean"].groupby(["model", "fold"], sort=False)["bias_pct"].mean()
    bias.unstack().to_csv(config.REPORTS_DIR / "backtest_bias_by_fold.csv")

    experiment = quote_signal_experiment(labeled, market_by_day, label_error)
    experiment.to_csv(config.REPORTS_DIR / "quote_signal_experiment.csv", index=False)
    shift = quote_signal_validation_shift(labeled, validation, market_by_day)

    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 100)
    print("\nWeighted to the validation mix, averaged over all folds:")
    print(
        summary[summary["labels"] == "clean"]
        .sort_values("mape_pct")[["model", "mape_pct", "mae_usd", "rmse_usd", "r2", "bias_pct"]]
        .round(3)
        .to_string(index=False)
    )
    print("\nFinal model on all labels, including corrupted ones:")
    print(summary[(summary["labels"] == "all") & (summary["model"] == "Structural (final)")].round(3).to_string(index=False))
    print("\nFinal model by fold:")
    print(by_fold[by_fold["labels"] == "clean"].round(3).to_string(index=False))
    print("\nBias by fold (%):")
    print(bias.unstack().round(2).to_string())
    print("\nquote_signal experiment (clean labels):")
    print(experiment.round(3).to_string(index=False))
    print(f"\nAdding quote_signal moves validation predictions by {shift:.2f}% on average.")
    (config.REPORTS_DIR / "quote_signal_shift.txt").write_text(f"{shift:.2f}\n")


if __name__ == "__main__":
    main()
