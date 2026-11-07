# Freight Rate Prediction

Predicts `posted_rate` for 12,000 loads in Nov-Dec 2025 from 48,000 labeled loads in Jan-Oct 2025. This is my solution to the Spotter Machine Learning Engineer assessment.

In a backtest built to mirror the task, the final model reaches **1.24% MAPE** ($29 MAE) on clean labels with almost no bias, compared with 1.75% for LightGBM on the same features. See the [report](reports/freight_rate_report.pdf) for the data validation, split design and results.

## Setup

Tested on Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`.

The assessment data is not in this repository. Put the four provided files in `data/` with these names:

```
data/train_test.csv
data/validation.csv
data/validation_predictions_template.csv
data/december_chart_inputs.csv
```

## Run

Fit the final model and write both prediction files (a few seconds). This creates `validation_predictions.csv` and fills the `predicted_rate` column in `data/december_chart_inputs.csv`:

```bash
python train_predict.py
```

Validate the outputs and create the December chart, using the provided scorer:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Re-run the backtest and write the result tables to `reports/` (about 2 minutes):

```bash
python run_backtest.py
```

Rebuild the PDF report. This needs the outputs of the three commands above:

```bash
python make_report.py
```

Results are deterministic, so these commands reproduce the committed outputs.

## How it works

**Data repairs (`src/data.py`).** Negative weights are sign errors, so they take their absolute value. Missing weights take the training median, and missing `market_index` values take the same-day average. Corrupted labels (1.4% of rows) are removed from training inside the model.

**`quote_signal` is excluded.** It tracks price at +0.99 in some months and -0.99 in others, and has no relationship with price in Nov-Dec. A random split makes it look useful, but it is a trap.

**Model (`src/model.py`).** A penalised linear model on log(rate), because rates are close to additive on a log scale with Gaussian noise:

- A spline for distance and a spline for weight
- Equipment type
- A smooth surface over latitude and longitude, shared by pickup and delivery, which also prices unseen cities
- A shrunken premium for each directed lane
- `market_index`, plus an extra term for abrupt steps in the index, which are detected from the index itself
- A linear time trend
- A quarter-end ramp by equipment (`src/features.py`)

**Validation (`src/evaluate.py`).** Five rolling folds each train on the past and test on the next two months. In each fold, 8 random cities are also removed from training, to copy the unseen cities in validation. Results are given with and without corrupted labels, because the answer key is hidden.

## Files

| Path | Purpose |
|---|---|
| `train_predict.py` | Fits the final model and writes the predictions |
| `run_backtest.py` | Runs the validation and model comparison |
| `make_report.py` | Builds the PDF report |
| `score.py` | Scorer provided by Spotter, unmodified |
| `src/` | Data repairs, features, model, comparison models, backtest and report figures |
| `reports/freight_rate_report.pdf` | Final report. The scripts also write figures and result tables to `reports/` |
| `scorer_results/candidate_december.png` | December chart from `score.py` |
| `validation_predictions.csv` | Final predictions (`load_id,predicted_rate`) |
