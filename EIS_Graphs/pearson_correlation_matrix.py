"""
correlation_matrix.py
(not split by polymer) loads combined_features.csv, calculates correlation matrix for
Concentration_pct, Low_Zmag_ohm, Low_ReZ_ohm, Low_Phase_deg, Arc_Diameter_ohm, Low_Freq_Tail_Slope, and
plots a heatmap of the correlation matrix with the correlation values written inside each box.
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("combined_features.csv")

FEATURES = [
    "Concentration_pct",
    "Low_Zmag_ohm",
    "Low_ReZ_ohm",
    "Low_Phase_deg",
    "Arc_Diameter_ohm",
    "Low_Freq_Tail_Slope",
]

corr_matrix = df[FEATURES].corr()

print(corr_matrix.round(2).to_string())

# --- Heatmap ---
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)

ax.set_xticks(range(len(FEATURES)))
ax.set_yticks(range(len(FEATURES)))
ax.set_xticklabels(FEATURES, rotation=45, ha="right")
ax.set_yticklabels(FEATURES)

# Write the correlation number inside each box
for i in range(len(FEATURES)):
    for j in range(len(FEATURES)):
        ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="black")

ax.set_title("Correlation Matrix (all 16 samples)")
fig.colorbar(im, label="Correlation (r)")
fig.tight_layout()
fig.savefig("correlation_matrix.png", dpi=150)
print("\nSaved correlation_matrix.png")
