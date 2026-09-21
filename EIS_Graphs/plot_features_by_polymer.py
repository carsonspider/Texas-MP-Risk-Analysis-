"""
plot_features_by_polymer.py

plot_curve_shape_features.py only draws two features against concentration the
arc diameter and low-frequency tail slope.

This script draws every extracted feature the same way, one chart per
feature, grouped into one figure per polymer (PP, PE, PCL).

Features come from combined_features.csv, which already joins:
  - the three frequency checkpoints (high ~45409 Hz, mid ~877 Hz, low ~1.07 Hz)
  - characteristic frequency
  - the curve-shape features (arc, tail slope, Bode plateau, Bode transition)

"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("combined_features.csv")
control = df[df["Polymer"] == "None (control)"]

# (column name, axis title, use a log y-axis?)
FEATURES = [
    ("High_Zmag_ohm", "High-freq |Z| (ohms)", False),
    ("Mid_Zmag_ohm", "Mid-freq |Z| (ohms)", False),
    ("Low_Zmag_ohm", "Low-freq |Z| (ohms)", False),
    ("High_ReZ_ohm", "High-freq Re(Z) (ohms)", False),
    ("Mid_ReZ_ohm", "Mid-freq Re(Z) (ohms)", False),
    ("Low_ReZ_ohm", "Low-freq Re(Z) (ohms)", False),
    ("High_Phase_deg", "High-freq phase (deg)", False),
    ("Mid_Phase_deg", "Mid-freq phase (deg)", False),
    ("Low_Phase_deg", "Low-freq phase (deg)", False),
    ("Arc_Diameter_ohm", "Arc diameter (ohms)", False),
    ("Low_Freq_Tail_Slope", "Low-freq tail slope", False),
    ("Bode_Plateau_ohm", "Bode plateau (ohms)", False),
    ("Bode_Transition_Freq_Hz", "Bode transition freq (Hz)", False),
    ("Bode_Transition_Zmag_ohm", "Bode transition |Z| (ohms)", False),
    ("Characteristic_Frequency_Hz", "Characteristic frequency (Hz)", True),
    ("Min_Phase_deg", "Most negative phase (deg)", False),
]

# These are measured where the control reading is not real.
SKIP_CONTROL_PREFIXES = ("Mid_", "Low_")
SKIP_CONTROL_EXACT = {
    "Low_Freq_Tail_Slope",
    "Bode_Transition_Freq_Hz",
    "Bode_Transition_Zmag_ohm",
    "Characteristic_Frequency_Hz",
    "Min_Phase_deg",
}

COLORS = {"PP": "tab:blue", "PE": "tab:orange", "PCL": "tab:green"}


def keep_control(column_name):
    if column_name in SKIP_CONTROL_EXACT:
        return False
    if column_name.startswith(SKIP_CONTROL_PREFIXES):
        return False
    return True


for polymer in ["PP", "PE", "PCL"]:
    this_polymer = df[df["Polymer"] == polymer]
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))

    for ax, (column, ylabel, log_y) in zip(axes.flat, FEATURES):
        if keep_control(column):
            table = pd.concat([control, this_polymer])
        else:
            table = this_polymer
        table = table.sort_values("Concentration_pct")

        ax.plot(table["Concentration_pct"], table[column], marker="o", color=COLORS[polymer])
        ax.set_title(ylabel, fontsize=10)
        ax.set_xlabel("Concentration (%)")
        ax.grid(alpha=0.3)
        if log_y:
            ax.set_yscale("log")

    fig.suptitle(f"{polymer}: every extracted feature vs concentration", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{polymer}_all_features.png", dpi=150)
    plt.close(fig)
    print(f"Saved {polymer}_all_features.png")
