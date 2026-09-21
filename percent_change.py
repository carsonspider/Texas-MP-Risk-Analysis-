"""
percent_change.py

For each sample, measure how far a feature sits from a baseline:

    percent change = (sample - baseline) / abs(baseline) * 100

A positive number means the feature is larger than the baseline.
A negative number means it is smaller. abs() is in the denominator so a
negative baseline (tail slope) does not flip that meaning.

THE 5 CURVE-SHAPE FEATURES
  1. Arc diameter
  2. Low-frequency tail slope
  3. Bode plateau
  4. Bode transition frequency
  5. Bode transition |Z|

"""

import pandas as pd
import numpy as np

FEATURES = [
    "Arc_Diameter_ohm",
    "Low_Freq_Tail_Slope",
    "Bode_Plateau_ohm",
    "Bode_Transition_Freq_Hz",
    "Bode_Transition_Zmag_ohm",
]

# Control is unusable when it is smaller than this fraction of a typical sample.
MIN_CONTROL_FRACTION = 0.05

df = pd.read_csv("curve_shape_features.csv")
control = df[df["Polymer"] == "None (control)"].iloc[0]
samples = df[df["Polymer"] != "None (control)"]


def percent_change(value, baseline):
    """(value - baseline) / abs(baseline) * 100, or blank if that is undefined."""
    if pd.isna(value) or pd.isna(baseline) or baseline == 0:
        return np.nan
    return (value - baseline) / abs(baseline) * 100


rows = []
for feature in FEATURES:
    control_value = control[feature]
    typical = np.nanmedian(np.abs(samples[feature].astype(float)))
    control_ok = (
        pd.notna(control_value)
        and typical > 0
        and abs(control_value) >= MIN_CONTROL_FRACTION * typical
    )

    if control_ok:
        print(f"{feature}: baseline = plain-soil control ({control_value:.4g})")
    else:
        why = "missing" if pd.isna(control_value) else f"too small ({control_value:.4g})"
        print(f"{feature}: control is {why}, so baseline = each polymer's 0.5% sample")

    for polymer in ["PP", "PE", "PCL"]:
        polymer_rows = samples[samples["Polymer"] == polymer].sort_values("Concentration_pct")
        baseline_row = polymer_rows[polymer_rows["Concentration_pct"] == 0.5]
        if baseline_row.empty:
            raise SystemExit(f"No 0.5% sample for {polymer}")
        polymer_baseline = baseline_row.iloc[0][feature]

        for _, sample in polymer_rows.iterrows():
            if control_ok:
                baseline = control_value
                source = "control"
            else:
                baseline = polymer_baseline
                source = f"{polymer} 0.5%"

            rows.append({
                "Sample_ID": sample["Sample_ID"],
                "Polymer": polymer,
                "Concentration_pct": sample["Concentration_pct"],
                "Feature": feature,
                "Value": sample[feature],
                "Naive_pct_vs_control": percent_change(sample[feature], control_value),
                "Fixed_baseline": baseline,
                "Fixed_baseline_source": source,
                "Fixed_pct_change": percent_change(sample[feature], baseline),
            })

result = pd.DataFrame(rows)
result.to_csv("percent_change.csv", index=False)

print("\nFixed percent change (blank means the baseline itself was missing):")
for feature in FEATURES:
    print(f"\n--- {feature} ---")
    view = result[result["Feature"] == feature]
    pivot = view.pivot(index="Concentration_pct", columns="Polymer", values="Fixed_pct_change")
    print(pivot.round(1).to_string())

print("\nSaved percent_change.csv")
