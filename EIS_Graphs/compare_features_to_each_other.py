"""
compare_features_to_each_other.py

 scatter plot with Arc_Diameter_ohm on the x-axis and
Low_Freq_Tail_Slope on the y-axis. 
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("combined_features.csv")

colors = {"None (control)": "gray", "PP": "tab:blue", "PE": "tab:orange", "PCL": "tab:green"}

fig, ax = plt.subplots(figsize=(7, 6))

for polymer, color in colors.items():
    subset = df[df["Polymer"] == polymer]
    ax.scatter(subset["Arc_Diameter_ohm"], subset["Low_Freq_Tail_Slope"],
               s=100, color=color, label=polymer, edgecolor="black", alpha=0.8)

ax.set_xlabel("Arc Diameter (ohms)")
ax.set_ylabel("Low-Freq Tail Slope")
ax.set_title("Arc Diameter vs Tail Slope")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("feature_vs_feature_comparison.png", dpi=150)
print("Saved feature_vs_feature_comparison.png")

# Also print the direct correlation between these two features
correlation = df["Arc_Diameter_ohm"].corr(df["Low_Freq_Tail_Slope"])
print(f"\nOverall correlation (all samples, all polymers combined): r = {correlation:+.3f}")
