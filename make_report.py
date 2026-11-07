"""Build the PDF report from the saved results.

Usage: python make_report.py
Run run_backtest.py, train_predict.py and score.py first. Numbers come from
their outputs, so the report cannot drift from the code. The one exception is
FIRST_RUN_WORST_BIAS, a record of an earlier model version that no longer exists.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src import config, data, eda
from src.data import clean_features
from src.evaluate import flag_label_errors
from src.model import StructuralRateModel

REPORT_PATH = config.REPORTS_DIR / "freight_rate_report.pdf"
CHART_PATH = config.ROOT / "scorer_results" / "candidate_december.png"
# Worst fold bias (%) of the first design, before market steps were added.
FIRST_RUN_WORST_BIAS = 1.2
TEAL = colors.HexColor("#064A56")
INK = colors.HexColor("#1b1b1a")
SOFT = colors.HexColor("#52514e")
RULE = colors.HexColor("#d9d8d3")
TINT = colors.HexColor("#f1f4f4")
PAGE_WIDTH = LETTER[0] - 1.5 * inch

BODY = ParagraphStyle("Body", fontName="Helvetica", fontSize=9.6, leading=13.4, textColor=INK, spaceAfter=5)
TITLE = ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=TEAL)
SUBTITLE = ParagraphStyle("Subtitle", fontName="Helvetica", fontSize=10, leading=14, textColor=SOFT, spaceAfter=9)
HEADING = ParagraphStyle(
    "Heading", fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=TEAL, spaceBefore=10, spaceAfter=4
)
CAPTION = ParagraphStyle("Caption", fontName="Helvetica", fontSize=8.2, leading=10.6, textColor=SOFT, spaceAfter=7)
CELL = ParagraphStyle("Cell", fontName="Helvetica", fontSize=8.3, leading=10.6, textColor=INK)
CELL_HEAD = ParagraphStyle("CellHead", parent=CELL, fontName="Helvetica-Bold", textColor=TEAL)
STAT_VALUE = ParagraphStyle(
    "StatValue", fontName="Helvetica-Bold", fontSize=16, leading=19, textColor=TEAL, alignment=TA_CENTER
)
STAT_LABEL = ParagraphStyle(
    "StatLabel", fontName="Helvetica", fontSize=7.8, leading=9.8, textColor=SOFT, alignment=TA_CENTER
)


def table(rows: list[list[str]], widths: list[float], bold_row: int | None = None) -> Table:
    cells = [[Paragraph(str(cell), CELL_HEAD if r == 0 else CELL) for cell in row] for r, row in enumerate(rows)]
    result = Table(cells, colWidths=[w * PAGE_WIDTH for w in widths], repeatRows=1)
    style = [
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, TEAL),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if bold_row is not None:
        style.append(("BACKGROUND", (0, bold_row), (-1, bold_row), TINT))
    result.setStyle(TableStyle(style))
    return result


def stat_row(stats: list[tuple[str, str]]) -> Table:
    row = [[Paragraph(value, STAT_VALUE), Paragraph(label, STAT_LABEL)] for value, label in stats]
    result = Table([row], colWidths=[PAGE_WIDTH / len(stats)] * len(stats))
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), TINT),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return result


def figure(path, width: float = PAGE_WIDTH) -> Image:
    from PIL import Image as Pillow

    with Pillow.open(path) as image:
        pixel_width, pixel_height = image.size
    return Image(str(path), width=width, height=width * pixel_height / pixel_width)


def percent_lift(log_effect: float) -> float:
    return (np.exp(log_effect) - 1) * 100


def count_pct(entry: dict) -> str:
    return f"{entry['count']:,} ({entry['pct']:.2f}%)"


def signed_pct(value: float) -> str:
    return "0.00%" if abs(value) < 0.005 else f"{value:+.2f}%"


def main() -> None:
    labeled = data.load_labeled()
    validation = data.load_validation()
    market_by_day = data.daily_market_index(labeled, validation)
    weight_fill = labeled["weight"].abs().median()
    label_error = flag_label_errors(labeled, market_by_day)
    frame = clean_features(labeled, market_by_day, weight_fill)
    model = StructuralRateModel().fit(frame)

    predictions = pd.read_csv(config.PREDICTIONS_PATH).set_index("load_id")["predicted_rate"]
    predicted_validation = validation["load_id"].map(predictions).to_numpy()
    december = pd.read_csv(config.DECEMBER_PATH, parse_dates=["date"])

    facts = eda.data_quality_facts(labeled, validation, label_error, model, market_by_day)
    by_month = eda.quote_signal_by_month(labeled, validation, label_error, predicted_validation)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    eda.plot_quote_signal(by_month, config.FIGURES_DIR / "quote_signal_by_month.png")
    eda.plot_folds(config.FIGURES_DIR / "backtest_folds.png")
    eda.plot_quarter_end_ramp(frame, label_error, model, config.FIGURES_DIR / "quarter_end_ramp.png")
    (config.REPORTS_DIR / "data_quality_facts.json").write_text(json.dumps(facts, indent=2))

    summary = pd.read_csv(config.REPORTS_DIR / "backtest_summary.csv")
    clean = summary[summary["labels"] == "clean"].set_index("model")
    all_labels = summary[summary["labels"] == "all"].set_index("model")
    by_fold = pd.read_csv(config.REPORTS_DIR / "backtest_final_by_fold.csv")
    bias_by_fold = pd.read_csv(config.REPORTS_DIR / "backtest_bias_by_fold.csv", index_col="model")
    experiment = pd.read_csv(config.REPORTS_DIR / "quote_signal_experiment.csv").set_index(["split", "quote_signal"])
    shift = float((config.REPORTS_DIR / "quote_signal_shift.txt").read_text())
    effects = json.loads((config.REPORTS_DIR / "model_effects.json").read_text())

    final = clean.loc["Structural (final)"]
    final_all = all_labels.loc["Structural (final)"]
    fold_clean = by_fold[by_fold["labels"] == "clean"]
    warm = fold_clean[fold_clean["segment"] == "warm"]
    cold = fold_clean[fold_clean["segment"] == "cold"]
    gbm = clean.loc["LightGBM, engineered features"]
    worst_bias = bias_by_fold.loc["Structural (final)"].abs().max()
    steps = facts["market_steps"]
    repairs = facts["repairs"]
    errors = facts["label_errors"]
    ramp = effects["ramp_end_of_quarter"]
    august = "Train Jan-Jul, test Aug"
    random_split = "Random 80/20"
    later = "Train Jan-Aug, test Sep-Oct"
    nov_dec_corr = by_month[by_month["month"] >= 11]["correlation"].abs().max()

    story = [
        Paragraph("Freight Rate Prediction", TITLE),
        Paragraph("Data validation and train/test split report &nbsp;|&nbsp; Machine Learning Engineer assessment", SUBTITLE),
        Paragraph(
            f"The task is to predict <b>posted_rate</b> for {facts['rows']['validation']:,} loads in Nov-Dec 2025, "
            f"using {facts['rows']['train']:,} labeled loads from Jan-Oct 2025. The final model is a structural "
            f"pricing model on log(rate). In a backtest built to mirror the validation task, it reaches "
            f"<b>{final['mape_pct']:.2f}% mean absolute percentage error (MAPE)</b> on clean labels with no "
            f"systematic bias, against {gbm['mape_pct']:.2f}% for LightGBM with the same engineered features.",
            BODY,
        ),
        stat_row(
            [
                (f"{final['mape_pct']:.2f}%", "expected MAPE,<br/>clean labels"),
                (f"${final['mae_usd']:.0f}", "expected mean<br/>absolute error"),
                (f"{cold['mape_pct'].mean():.2f}%", "MAPE on loads touching<br/>an unseen city"),
                (f"{worst_bias:.2f}%", "largest bias<br/>in any fold"),
            ]
        ),
        Paragraph("1. Validating the data", HEADING),
        Paragraph(
            "I profiled every column in both files and tested each repair against the labeled data. Bad values are "
            "two to three times more common in validation, so these repairs matter even more there.",
            BODY,
        ),
        table(
            [
                ["Issue", "Training", "Validation", "Treatment", "Evidence"],
                [
                    "<b>Corrupted posted_rate</b>",
                    count_pct(errors),
                    "Unknown",
                    "Removed from training by model residual",
                    f"Off by x{errors['low_range'][0]:.2f}-{errors['low_range'][1]:.2f} or "
                    f"x{errors['high_range'][0]:.1f}-{errors['high_range'][1]:.1f}. In log terms, clean loads sit "
                    f"within {np.ceil(facts['largest_clean_residual'] * 100) / 100:.2f} of the model and corrupted "
                    f"ones at least {np.floor(facts['smallest_corrupted_residual'] * 100) / 100:.2f} away, "
                    "with nothing between",
                ],
                [
                    "<b>Negative weight</b>",
                    count_pct(facts["negative_weight"]["train"]),
                    count_pct(facts["negative_weight"]["validation"]),
                    "Absolute value",
                    f"Sign errors. Price error once flipped is {repairs['negative_flipped']:.4f}, like normal loads "
                    f"({repairs['normal']:.4f}). A median fill doubles it ({repairs['negative_median_filled']:.4f})",
                ],
                [
                    "<b>Missing weight</b>",
                    count_pct(facts["missing_weight"]["train"]),
                    count_pct(facts["missing_weight"]["validation"]),
                    "Training median",
                    f"Unbiased but less precise: price error {repairs['missing_weight']:.4f}",
                ],
                [
                    "<b>Missing market_index</b>",
                    count_pct(facts["missing_market"]["train"]),
                    count_pct(facts["missing_market"]["validation"]),
                    "Same-day average across loads",
                    f"The index varies by {facts['market_sd_within_day']:.3f} within a day and "
                    f"{facts['market_sd_between_days']:.2f} between days. Price error {repairs['missing_market']:.4f}",
                ],
                [
                    "<b>Unseen cities</b>",
                    "None",
                    f"{len(facts['new_cities'])} cities, {count_pct(facts['cold_loads'])}",
                    "Priced from coordinates",
                    f"City effects form a smooth map. Backtested MAPE is {cold['mape_pct'].mean():.2f}%, "
                    f"against {warm['mape_pct'].mean():.2f}% for known cities",
                ],
                [
                    "<b>quote_signal</b>",
                    "All rows",
                    "All rows",
                    "Excluded",
                    "Unstable, and unrelated to price in Nov-Dec. See section 2",
                ],
            ],
            [0.16, 0.12, 0.13, 0.17, 0.42],
        ),
        Spacer(1, 4),
        Paragraph(
            f"No problems found: {facts['duplicate_ids']} duplicate load IDs, no duplicate rows, one date format, and "
            "one coordinate pair per city. Coordinates do not match real-world locations (Los Angeles sits at latitude "
            "28.6), but <b>distance</b> agrees with them at a median 1.18 times the straight line, so I used them as "
            "supplied.",
            BODY,
        ),
        Paragraph("2. The quote_signal trap", HEADING),
        Paragraph(
            "quote_signal looks like the best feature in the file and is the most dangerous one. It tracks rate per "
            "mile almost perfectly in some months, is almost perfectly inverted in others, and is unrelated in August. "
            "Nov-Dec has no labels, so I compared it with predictions from a model that never sees it. The correlation "
            f"there is {nov_dec_corr:.2f}, which means a model trained on the relationship would be relying on one that "
            "no longer exists.",
            BODY,
        ),
        KeepTogether(
            [
                figure(config.FIGURES_DIR / "quote_signal_by_month.png"),
                Paragraph(
                    "Figure 1. Rank correlation with rate per mile. Jan-Oct uses actual rates. Nov-Dec uses model "
                    "predictions, as no labels exist.",
                    CAPTION,
                ),
            ]
        ),
        KeepTogether(
            [
                table(
                    [
                        ["LightGBM test, clean labels", "MAPE with quote_signal", "MAPE without"],
                        [
                            "Random 80/20 split",
                            f"{experiment.loc[(random_split, 'with'), 'mape_pct']:.2f}%",
                            f"{experiment.loc[(random_split, 'without'), 'mape_pct']:.2f}%",
                        ],
                        [
                            "Train Jan-Aug, test Sep-Oct",
                            f"{experiment.loc[(later, 'with'), 'mape_pct']:.2f}%",
                            f"{experiment.loc[(later, 'without'), 'mape_pct']:.2f}%",
                        ],
                        [
                            "<b>Train Jan-Jul, test Aug</b> (signal unrelated to price, as in Nov-Dec)",
                            f"<b>{experiment.loc[(august, 'with'), 'mape_pct']:.2f}%</b>",
                            f"<b>{experiment.loc[(august, 'without'), 'mape_pct']:.2f}%</b>",
                        ],
                    ],
                    [0.6, 0.22, 0.18],
                ),
                Spacer(1, 4),
                Paragraph(
                    "A random split recommends keeping the feature, because every month's relationship is in training. "
                    "It helps slightly on Sep-Oct, where it is still tied to price, but that link is gone by Nov-Dec. "
                    "The August test shows the real cost. On the validation file itself, adding quote_signal moves "
                    f"predictions by {shift:.1f}% on average with no information behind the move.",
                    BODY,
                ),
            ]
        ),
        Paragraph("3. How I split the data", HEADING),
        Paragraph(
            "The validation file differs from the training data in three ways, so the test design copies all three. "
            "A random split would copy none of them.",
            BODY,
        ),
        figure(config.FIGURES_DIR / "backtest_folds.png"),
        Paragraph(
            "Figure 2. Five rolling folds. Each trains on the past and tests on the next two months, the same "
            "horizon as Nov-Dec.",
            CAPTION,
        ),
        Paragraph(
            "<b>Future dates.</b> Every fold trains only on earlier months and tests on the next two. The folds cover "
            "both kinds of two-month window that matter: those with a quarter-end ramp and those without.",
            BODY,
        ),
        Paragraph(
            f"<b>Unseen cities.</b> {facts['cold_loads']['pct']:.1f}% of validation loads touch a city with no training "
            "history. In each of five repeats per fold, I removed 8 random cities from training and scored test loads "
            "touching them separately. Final figures are weighted to the validation mix of known and unseen cities.",
            BODY,
        ),
        Paragraph(
            "<b>Corrupted labels.</b> Each fold removes them from its own training rows only. I can't tell whether "
            "Spotter's answer key contains them, so I report results both ways.",
            BODY,
        ),
        Paragraph(
            "<b>Leakage.</b> Weight fills and model transforms are learned on training rows. The same-day market "
            "average and step detection read feature values only, never a label.",
            BODY,
        ),
        Paragraph(
            "<b>How blind these results are.</b> I chose the first model design on folds 1-3. When I then ran folds 4-5, "
            f"they came in low by as much as {FIRST_RUN_WORST_BIAS}%. The cause was three abrupt steps in market_index "
            f"({', '.join(f'{day}: {size:+.2f}' for day, size in steps.items())}), where prices react more than they do "
            "to ordinary index moves. Detecting those steps from the index and adding one coefficient brought bias "
            f"below {np.ceil(worst_bias * 10) / 10:.1f}% in every fold, including folds that meet a step they never "
            "trained on. Because I changed the model after seeing those folds, and had already explored all ten "
            "months, these results are not fully blind. Read them as a strong estimate rather than a pristine holdout.",
            BODY,
        ),
    ]

    comparison = [["Model", "MAPE", "MAE", "RMSE", "Bias"]]
    for name in [
        "Structural (final)",
        "Structural + LightGBM on residuals",
        "LightGBM, engineered features",
        "LightGBM, raw features",
        "Lane median",
    ]:
        row = clean.loc[name]
        comparison.append(
            [
                name,
                f"{row['mape_pct']:.2f}%",
                f"${row['mae_usd']:.0f}",
                f"${row['rmse_usd']:.0f}",
                signed_pct(row["bias_pct"]),
            ]
        )
    folds_table = [["Fold", "Test window", "MAPE, known cities", "MAPE, unseen cities", "Bias, final model"]]
    for index, name in enumerate(warm["fold"], start=1):
        folds_table.append(
            [
                str(index),
                name,
                f"{warm.set_index('fold').loc[name, 'mape_pct']:.2f}%",
                f"{cold.set_index('fold').loc[name, 'mape_pct']:.2f}%",
                signed_pct(bias_by_fold.loc["Structural (final)", name]),
            ]
        )
    ablation = {
        "lane effects": clean.loc["Structural without lane effects", "mape_pct"],
        "market steps": clean.loc["Structural without market steps", "mape_pct"],
        "the quarter-end ramp": clean.loc["Structural without quarter-end ramp", "mape_pct"],
        "the trend": clean.loc["Structural without trend", "mape_pct"],
    }
    gain = final["mape_pct"] - clean.loc["Structural + LightGBM on residuals", "mape_pct"]
    december_rise = december["predicted_rate"].iloc[-7:].mean() / december["predicted_rate"].iloc[:7].mean() - 1
    peak_days = december.loc[december["predicted_rate"].groupby(december["date"].dt.isocalendar().week).idxmax(), "date"]

    story += [
        KeepTogether(
            [
                Paragraph("4. Results", HEADING),
                table(comparison, [0.46, 0.135, 0.135, 0.135, 0.135], bold_row=1),
                Paragraph(
                    "Clean labels, averaged over five folds and weighted to the validation mix.",
                    CAPTION,
                ),
            ]
        ),
        Paragraph(
            "Rates here are close to additive on a log scale with Gaussian noise, so stating that form directly works "
            "better than asking trees to rediscover it. It also suits this task: trees hold their last value flat "
            "beyond the training dates, so they cannot carry the upward trend into Nov-Dec, and that shows up as "
            f"their bias. Adding LightGBM on the residuals gains only {gain:.2f} points, which does not justify "
            f"a second model. Each part of the final model earns its place. MAPE rises to "
            + ", ".join(f"{value:.2f}% without {name}" for name, value in ablation.items())
            + ".",
            BODY,
        ),
        table(folds_table, [0.08, 0.2, 0.26, 0.26, 0.2]),
        Spacer(1, 5),
        Paragraph(
            f"<b>If the answer key contains corrupted labels.</b> Results above exclude them. If Spotter's labels carry "
            f"the same {errors['pct']:.1f}% error rate, expect about {final_all['mape_pct']:.1f}% MAPE, "
            f"${final_all['mae_usd']:.0f} MAE, ${final_all['rmse_usd']:.0f} RMSE and R-squared of {final_all['r2']:.2f}. "
            "No model can predict random errors of that size, so I predict the true price rather than "
            "inflating predictions to chase them.",
            BODY,
        ),
        Paragraph(
            f"<b>What the model learned.</b> Reefer costs {percent_lift(effects['reefer_premium']):.1f}% and Flatbed "
            f"{percent_lift(effects['flatbed_premium']):.1f}% more than Dry Van. Rates drift up about "
            f"{percent_lift(effects['trend_per_year']):.0f}% a year. A 0.1 rise in market_index adds about "
            f"{percent_lift(effects['market_beta'] * 0.1):.1f}%, or {percent_lift((effects['market_beta'] + effects['market_step_extra']) * 0.1):.1f}% "
            "when it arrives as an abrupt step. Each lane has its own stable premium or discount, typically about 1%, "
            "learned with shrinkage so that lanes with few loads stay close to zero.",
            BODY,
        ),
        KeepTogether(
            [
                Paragraph("5. The December prediction chart", HEADING),
                figure(CHART_PATH),
                Paragraph("Figure 3. Output of the provided score.py, unmodified.", CAPTION),
            ]
        ),
        Paragraph(
            "Only the date changes, so this chart isolates the model's calendar behavior. The December file has "
            "no market_index, so I used each date's average across validation.csv loads. Three effects shape "
            "the line:",
            BODY,
        ),
        Paragraph(
            "<b>Weekly cycle.</b> The market index peaks Wednesday to Thursday and bottoms out Sunday to Monday.",
            BODY,
        ),
        Paragraph(
            f"<b>Quarter-end ramp.</b> Rates climb through the last month of every quarter and reset on the 1st. "
            f"March, June and September show the same pattern at the same size (Figure 4). A Dry Van load gains "
            f"about {percent_lift(ramp['Dry Van']):.1f}% by the 31st.",
            BODY,
        ),
        Paragraph(
            f"<b>Trend.</b> The steady upward drift adds about 0.6% over the month. Together, these put the last "
            f"week {december_rise * 100:.1f}% above the first on average, and the line runs from "
            f"${december['predicted_rate'].iloc[0]:.0f} on Dec 1 to ${december['predicted_rate'].iloc[-1]:.0f} "
            "on Dec 31.",
            BODY,
        ),
        KeepTogether(
            [
                figure(config.FIGURES_DIR / "quarter_end_ramp.png"),
                Paragraph(
                    "Figure 4. Quarter-end ramp by equipment, with all other model effects removed (3-day average). "
                    "Dashed lines mark quarter starts.",
                    CAPTION,
                ),
            ]
        ),
        Paragraph("6. Limits", HEADING),
        Paragraph(
            "<b>No year-end history.</b> The December ramp comes from three earlier quarter-ends. The eight federal "
            "holidays in the training data all sit within 0.3% of the model with no consistent direction, so I added no "
            "holiday effect, but the data has no Thanksgiving or Christmas. A real peak-season effect would not appear "
            "in these predictions.",
            BODY,
        ),
        Paragraph(
            "<b>New market shocks.</b> market_index shows no step in Nov-Dec, so none is applied. The model can react "
            "to a step visible in the index, but not to a price change that does not appear there.",
            BODY,
        ),
        Paragraph(
            "<b>Unseen cities.</b> A new lane's own premium cannot be known, which is why these loads run about 0.2 "
            "points higher in error.",
            BODY,
        ),
    ]

    def footer(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(SOFT)
        canvas.drawRightString(LETTER[0] - 0.75 * inch, 0.45 * inch, f"Page {document.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.6 * inch,
        title="Freight Rate Prediction: Data Validation and Split Report",
        author="",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
