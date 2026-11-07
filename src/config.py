"""Paths and constants shared by the pipeline."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"

PREDICTIONS_PATH = ROOT / "validation_predictions.csv"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TIME_ORIGIN = pd.Timestamp("2025-01-01")
TARGET = "posted_rate"

# On a log scale, every clean load is within 0.08 of the model and every
# corrupted label is at least 0.81 away. Nothing falls in between, so any
# threshold in that empty range gives the same result.
LABEL_ERROR_LOG_THRESHOLD = 0.35
