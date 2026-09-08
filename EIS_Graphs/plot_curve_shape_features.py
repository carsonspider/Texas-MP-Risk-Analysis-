"""
plot_curve_shape_features.py

loads curve_shape_features.csv,  plots
Concentration vs Arc_Diameter_ohm and Concentration vs Low_Freq_Tail_Slope
on two side-by-side charts. control included
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("curve_shape_features.csv")
control = df[df["Polymer"] == "None (control)"]

for polymer in ["PP", "PE", "PCL"]:
    this_polymer = df[df["Polymer"] == polymer]
    table = pd.concat([control, this_polymer]).sort_values("Concentration_pct")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(table["Concentration_pct"], table["Arc_Diameter_ohm"], marker="o")
    ax1.set_title(f"{polymer}: Arc Diameter vs Concentration")
    ax1.set_xlabel("Concentration (%)")
    ax1.set_ylabel("Arc Diameter (ohms)")
    ax1.grid(alpha=0.3)

    ax2.plot(table["Concentration_pct"], table["Low_Freq_Tail_Slope"], marker="o", color="orange")
    ax2.set_title(f"{polymer}: Low-Freq Tail Slope vs Concentration")
    ax2.set_xlabel("Concentration (%)")
    ax2.set_ylabel("Tail Slope")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{polymer}_curve_shape_features.png", dpi=150)
    plt.close(fig)
    print(f"Saved {polymer}_curve_shape_features.png")
