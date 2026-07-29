"""
Altair visualizations for the linear regression composite-risk model.

Exports PDF figures only to figures/.
"""

from pathlib import Path
import altair as alt
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
DATA_PATH = OUT_DIR / "ML_Ready_Sheet - ML_Ready_Sheet.csv"
TARGET_COL = "My_Composite_Risk_Score"

FEATURE_COLS = [
    "Agricultural_Land_Percent",
    "Livestock_Density_per_sq_mi",
    "Irrigated_Land_Acres",
    "Interstate_Mileage_mi",
    "Road_Density_mi_per_sq_mi",
    "Biosolids_Application_Sites",
    "WWTP_Count",
    "WWTP_Total_Treatment_Capacity_MGD",
    "Population",
    "Population_Density_per_sq_mi",
    "Urbanization_Percent",
]

#  axis labels for long CSV column names
FEATURE_LABELS = {
    "Agricultural_Land_Percent": "Agricultural land %",
    "Livestock_Density_per_sq_mi": "Livestock density",
    "Irrigated_Land_Acres": "Irrigated land (acres)",
    "Interstate_Mileage_mi": "Interstate mileage",
    "Road_Density_mi_per_sq_mi": "Road density",
    "Biosolids_Application_Sites": "Biosolids sites",
    "WWTP_Count": "WWTP count",
    "WWTP_Total_Treatment_Capacity_MGD": "WWTP capacity (MGD)",
    "Population": "Population",
    "Population_Density_per_sq_mi": "Population density",
    "Urbanization_Percent": "Urbanization %",
}

COLORS = {
    "positive": "#1F6F6A",
    "negative": "#C45C48",
    "train": "#3D5A6C",
    "test": "#D4A017",
    "zero": "#8A9399",
    "grid": "#E6E9EB",
    "text": "#1C2429",
    "muted": "#5C6B73",
    "bg": "#FAFBFC",
    "panel": "#FFFFFF",
}


def _theme() -> alt.theme.ThemeConfig:
    return {
        "config": {
            "background": COLORS["bg"],
            "view": {"stroke": "transparent", "fill": COLORS["panel"]},
            "axis": {
                "labelFont": "Helvetica Neue, Helvetica, Arial, sans-serif",
                "titleFont": "Helvetica Neue, Helvetica, Arial, sans-serif",
                "labelColor": COLORS["muted"],
                "titleColor": COLORS["text"],
                "labelFontSize": 11,
                "titleFontSize": 12,
                "titleFontWeight": 600,
                "gridColor": COLORS["grid"],
                "domainColor": "#C5CDD2",
                "tickColor": "#C5CDD2",
            },
            "legend": {
                "labelFont": "Helvetica Neue, Helvetica, Arial, sans-serif",
                "titleFont": "Helvetica Neue, Helvetica, Arial, sans-serif",
                "labelColor": COLORS["muted"],
                "titleColor": COLORS["text"],
                "labelFontSize": 11,
                "titleFontSize": 12,
                "titleFontWeight": 600,
            },
            "title": {
                "font": "Helvetica Neue, Helvetica, Arial, sans-serif",
                "fontSize": 15,
                "fontWeight": 650,
                "color": COLORS["text"],
                "subtitleFont": "Helvetica Neue, Helvetica, Arial, sans-serif",
                "subtitleFontSize": 12,
                "subtitleColor": COLORS["muted"],
                "anchor": "start",
                "offset": 12,
            },
            "range": {
                "category": [COLORS["train"], COLORS["test"]],
            },
        }
    }


@alt.theme.register("lr_report", enable=True)
def lr_report_theme() -> alt.theme.ThemeConfig:
    return _theme()


alt.data_transformers.disable_max_rows()


def load_data():
    coef = pd.read_csv(OUT_DIR / "lr_coefficients.csv")
    pred = pd.read_csv(OUT_DIR / "lr_predictions.csv")
    metrics = pd.read_csv(OUT_DIR / "lr_metrics.csv")
    return coef, pred, metrics


def chart_standardized_coefficients(coef: pd.DataFrame) -> alt.Chart:
    """Horizontal bar chart of standardized coefficients (exclude intercept)."""
    d = coef.dropna(subset=["standardized_coefficient"]).copy()
    d["label"] = d["feature"].map(FEATURE_LABELS).fillna(d["feature"])
    d["direction"] = d["standardized_coefficient"].apply(
        lambda v: "Positive" if v >= 0 else "Negative"
    )
    # Sort by absolute magnitude for ranking importance
    d = d.sort_values("standardized_coefficient", key=abs, ascending=True)

    bars = (
        alt.Chart(d)
        .mark_bar(cornerRadiusEnd=3, height=18)
        .encode(
            x=alt.X(
                "standardized_coefficient:Q",
                title="Standardized coefficient",
                axis=alt.Axis(format=".3f", tickCount=6),
            ),
            y=alt.Y(
                "label:N",
                sort=None,
                title=None,
            ),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Positive", "Negative"],
                    range=[COLORS["positive"], COLORS["negative"]],
                ),
                legend=alt.Legend(title="Effect direction", orient="top"),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Feature"),
                alt.Tooltip("standardized_coefficient:Q", title="Std. coef", format=".4f"),
                alt.Tooltip("coefficient:Q", title="Raw coef", format=".4e"),
            ],
        )
    )

    zero = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(color=COLORS["zero"], strokeWidth=1.5, strokeDash=[4, 3])
        .encode(x="x:Q")
    )

    return (
        (bars + zero)
        .properties(
            width=480,
            height=320,
            title={
                "text": "Standardized coefficients",
            
            },
        )
    )


def chart_predicted_vs_actual(pred: pd.DataFrame, metrics: pd.DataFrame) -> alt.Chart:
    """Scatter of predicted vs actual, colored by train/test, with y = x reference."""
    d = pred.copy()
    d["split"] = d["split"].str.title()

    test_r2 = float(metrics.loc[metrics["split"] == "test", "R2"].iloc[0])
    train_r2 = float(metrics.loc[metrics["split"] == "train", "R2"].iloc[0])

    lo = min(d[TARGET_COL].min(), d["predicted"].min())
    hi = max(d[TARGET_COL].max(), d["predicted"].max())
    pad = (hi - lo) * 0.05
    lo, hi = lo - pad, hi + pad
    line_df = pd.DataFrame({TARGET_COL: [lo, hi], "predicted": [lo, hi]})

    points = (
        alt.Chart(d)
        .mark_circle(size=55, opacity=0.75)
        .encode(
            x=alt.X(
                f"{TARGET_COL}:Q",
                title="Actual My_Composite_Risk_Score",
                scale=alt.Scale(domain=[lo, hi]),
                axis=alt.Axis(format=".2f"),
            ),
            y=alt.Y(
                "predicted:Q",
                title="Predicted score",
                scale=alt.Scale(domain=[lo, hi]),
                axis=alt.Axis(format=".2f"),
            ),
            color=alt.Color(
                "split:N",
                scale=alt.Scale(
                    domain=["Train", "Test"],
                    range=[COLORS["train"], COLORS["test"]],
                ),
                legend=alt.Legend(title="Split", orient="top"),
            ),
            tooltip=[
                alt.Tooltip("County:N"),
                alt.Tooltip("FIPS:N"),
                alt.Tooltip("split:N", title="Split"),
                alt.Tooltip(f"{TARGET_COL}:Q", title="Actual", format=".4f"),
                alt.Tooltip("predicted:Q", title="Predicted", format=".4f"),
                alt.Tooltip("residual:Q", title="Residual", format=".4f"),
            ],
        )
    )

    ref = (
        alt.Chart(line_df)
        .mark_line(color=COLORS["zero"], strokeDash=[5, 4], strokeWidth=1.5)
        .encode(x=f"{TARGET_COL}:Q", y="predicted:Q")
    )

    return (
        (ref + points)
        .properties(
            width=420,
            height=420,
            title={
                "text": "Predicted vs actual",
    
            },
        )
    )


def chart_residuals(pred: pd.DataFrame) -> alt.Chart:
    """Residual vs predicted scatter + residual histogram"""
    d = pred.copy()
    d["split"] = d["split"].str.title()

    scatter = (
        alt.Chart(d)
        .mark_circle(size=50, opacity=0.75)
        .encode(
            x=alt.X(
                "predicted:Q",
                title="Predicted score",
                axis=alt.Axis(format=".2f"),
            ),
            y=alt.Y(
                "residual:Q",
                title="Residual (actual − predicted)",
                axis=alt.Axis(format=".3f"),
            ),
            color=alt.Color(
                "split:N",
                scale=alt.Scale(
                    domain=["Train", "Test"],
                    range=[COLORS["train"], COLORS["test"]],
                ),
                legend=alt.Legend(title="Split", orient="top"),
            ),
            tooltip=[
                alt.Tooltip("County:N"),
                alt.Tooltip("split:N", title="Split"),
                alt.Tooltip("predicted:Q", title="Predicted", format=".4f"),
                alt.Tooltip("residual:Q", title="Residual", format=".4f"),
            ],
        )
        .properties(width=380, height=280)
    )

    zero = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color=COLORS["zero"], strokeWidth=1.5, strokeDash=[4, 3])
        .encode(y="y:Q")
    )

    residual_scatter = (scatter + zero).properties(
        title={
            "text": "Residuals vs predicted",
           
        }
    )

    hist = (
        alt.Chart(d)
        .mark_bar(opacity=0.85, binSpacing=1)
        .encode(
            x=alt.X(
                "residual:Q",
                bin=alt.Bin(maxbins=28),
                title="Residual (actual − predicted)",
                axis=alt.Axis(format=".3f"),
            ),
            y=alt.Y("count():Q", title="Count of counties"),
            color=alt.Color(
                "split:N",
                scale=alt.Scale(
                    domain=["Train", "Test"],
                    range=[COLORS["train"], COLORS["test"]],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("split:N", title="Split"),
                alt.Tooltip("count():Q", title="Count"),
            ],
        )
        .properties(
            width=380,
            height=280,
            title={
                "text": "Residual distribution",
                
            },
        )
    )

    return alt.hconcat(residual_scatter, hist, spacing=28).resolve_scale(color="independent")


def chart_correlation_heatmap() -> alt.Chart:
    """Pearson correlation heatmap of model features (multicollinearity check)."""
    raw = pd.read_csv(DATA_PATH)
    X = raw[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    # Median-fill so every county contributes (same idea as the model pipeline)
    X = X.fillna(X.median(numeric_only=True))

    corr = X.corr(method="pearson")
    labels = [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS]
    corr.index = labels
    corr.columns = labels

    # Long form for Altair; keep feature order for a readable matrix
    long = (
        corr.reset_index(names="feature_y")
        .melt(id_vars="feature_y", var_name="feature_x", value_name="correlation")
    )
    long["feature_x"] = pd.Categorical(long["feature_x"], categories=labels, ordered=True)
    long["feature_y"] = pd.Categorical(long["feature_y"], categories=labels, ordered=True)
    # Text on cells with |r| high enough to read; skip near-zero clutter
    long["label"] = long["correlation"].apply(
        lambda r: f"{r:.2f}" if abs(r) >= 0.35 else ""
    )

    # Also export the numeric matrix for tables / papers
    corr_path = OUT_DIR / "lr_feature_correlations.csv"
    corr.to_csv(corr_path)
    print(f"  Wrote {corr_path.relative_to(OUT_DIR)}")

    heat = (
        alt.Chart(long)
        .mark_rect(stroke=COLORS["panel"], strokeWidth=1)
        .encode(
            x=alt.X(
                "feature_x:N",
                title=None,
                sort=labels,
                axis=alt.Axis(labelAngle=-40, labelLimit=140),
            ),
            y=alt.Y(
                "feature_y:N",
                title=None,
                sort=labels,
                axis=alt.Axis(labelLimit=140),
            ),
            color=alt.Color(
                "correlation:Q",
                title="Pearson r",
                scale=alt.Scale(
                    domain=[-1, 0, 1],
                    range=[COLORS["negative"], "#F4F5F6", COLORS["positive"]],
                ),
                legend=alt.Legend(orient="right", gradientLength=160),
            ),
            tooltip=[
                alt.Tooltip("feature_x:N", title="Feature X"),
                alt.Tooltip("feature_y:N", title="Feature Y"),
                alt.Tooltip("correlation:Q", title="r", format=".3f"),
            ],
        )
    )

    text = (
        alt.Chart(long)
        .mark_text(fontSize=9, fontWeight=500)
        .encode(
            x=alt.X("feature_x:N", sort=labels),
            y=alt.Y("feature_y:N", sort=labels),
            text="label:N",
            color=alt.condition(
                "abs(datum.correlation) > 0.65",
                alt.value("#FFFFFF"),
                alt.value(COLORS["text"]),
            ),
        )
    )

    return (
        (heat + text)
        .properties(
            width=420,
            height=420,
            title={
                "text": "Feature correlation heatmap",
            },
        )
    )


def save_chart(chart: alt.Chart, stem: str) -> None:
    """Save chart as PDF."""
    FIG_DIR.mkdir(exist_ok=True)
    pdf_path = FIG_DIR / f"{stem}.pdf"
    chart.save(str(pdf_path))
    print(f"  Wrote {pdf_path.relative_to(OUT_DIR)}")


def main() -> None:
    coef, pred, metrics = load_data()

    coef_chart = chart_standardized_coefficients(coef)
    scatter_chart = chart_predicted_vs_actual(pred, metrics)
    resid_chart = chart_residuals(pred)
    corr_chart = chart_correlation_heatmap()

    save_chart(coef_chart, "coef_standardized")
    save_chart(scatter_chart, "predicted_vs_actual")
    save_chart(resid_chart, "residuals")
    save_chart(corr_chart, "correlation_heatmap")

    dashboard = (
        alt.vconcat(
            coef_chart,
            scatter_chart,
            resid_chart,
            corr_chart,
            spacing=36,
        )
        .properties(
            title={
                "text": "Linear regression diagnostics",
                
            }
        )
        .configure_view(stroke=None)
    )

    save_chart(dashboard, "lr_dashboard")


if __name__ == "__main__":
    main()
