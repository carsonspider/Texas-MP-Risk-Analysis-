"""
plot_characteristic_frequency.py

loads characteristic_frequencies.csv, plots Concentration vs
Characteristic_Frequency_Hz, 
Y-axis uses log scale 
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("characteristic_frequencies.csv")
control = df[df["Polymer"] == "None (control)"]

fig, ax = plt.subplots(figsize=(7, 5))

for polymer in ["PP", "PE", "PCL"]:
    this_polymer = df[df["Polymer"] == polymer]
    table = pd.concat([control, this_polymer]).sort_values("Concentration_pct")
    ax.plot(table["Concentration_pct"], table["Characteristic_Frequency_Hz"], marker="o", label=polymer)

ax.set_yscale("log")
ax.set_xlabel("Concentration (%)")
ax.set_ylabel("Characteristic Frequency (Hz, log scale)")
ax.set_title("Characteristic Frequency vs Concentration")
ax.legend()
ax.grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig("characteristic_frequency_comparison.png", dpi=150)
print("Saved characteristic_frequency_comparison.png")
