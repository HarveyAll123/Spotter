"""Fit the final model on all labeled data and write both prediction files.

Usage: python train_predict.py
"""
from __future__ import annotations

import json

import pandas as pd

from src import config, data
from src.data import add_coordinates, city_coordinates, clean_features
from src.model import StructuralRateModel


def main() -> None:
    labeled = data.load_labeled()
    validation = data.load_validation()
    market_by_day = data.daily_market_index(labeled, validation)
    weight_fill = labeled["weight"].abs().median()

    model = StructuralRateModel().fit(clean_features(labeled, market_by_day, weight_fill))

    features = clean_features(validation, market_by_day, weight_fill)
    predicted = pd.Series(model.predict(features).round(2), index=validation["load_id"])
    template = pd.read_csv(config.TEMPLATE_PATH)
    template["predicted_rate"] = template["load_id"].map(predicted)
    template[["load_id", "predicted_rate"]].to_csv(config.PREDICTIONS_PATH, index=False)

    # The December file has no coordinates or market index. Coordinates come
    # from the city table, and the market index from all loads on that date.
    december = pd.read_csv(config.DECEMBER_PATH)
    inputs = add_coordinates(data.load_december(), city_coordinates(labeled, validation))
    december["predicted_rate"] = model.predict(
        clean_features(inputs, market_by_day, weight_fill)
    ).round(2)
    december.to_csv(config.DECEMBER_PATH, index=False)

    config.REPORTS_DIR.mkdir(exist_ok=True)
    effects = model.effect_summary()
    (config.REPORTS_DIR / "model_effects.json").write_text(json.dumps(effects, indent=2))
    print(json.dumps(effects, indent=2))
    print(f"Wrote {config.PREDICTIONS_PATH.name} ({len(template):,} rows)")
    print(f"Filled {config.DECEMBER_PATH.name} ({len(december)} rows)")


if __name__ == "__main__":
    main()
