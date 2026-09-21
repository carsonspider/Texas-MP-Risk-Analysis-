"""
spearman_correlation.py


This script uses the same table and the same six columns as
correlation_matrix.py, so the two heatmaps can be compared directly.

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


corr_matrix = df[FEATURES].corr(method="spearman")

print(corr_matrix.round(2).to_string())
corr_matrix.to_csv("spearman_correlation.csv")

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)

ax.set_xticks(range(len(FEATURES)))
ax.set_yticks(range(len(FEATURES)))
ax.set_xticklabels(FEATURES, rotation=45, ha="right")
ax.set_yticklabels(FEATURES)

for i in range(len(FEATURES)):
    for j in range(len(FEATURES)):
        ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="black")

ax.set_title("Spearman Correlation (all 16 samples)")
fig.colorbar(im, label="Spearman correlation")
fig.tight_layout()
fig.savefig("spearman_correlation.png", dpi=150)
print("\nSaved spearman_correlation.png and spearman_correlation.csv")
