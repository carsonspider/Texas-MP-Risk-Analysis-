from pathlib import Path
import pandas as pd

folder = Path(__file__).resolve().parent
pred = pd.read_csv(folder / "lr_predictions.csv")

# Outlier rule: |residual| more than 2 SD from the mean residual
mean_r = pred["residual"].mean()
sd_r = pred["residual"].std()
cutoff = 2 * sd_r

outliers = pred[pred["residual"].abs() > cutoff].copy()
outliers["residual_SDs"] = (outliers["residual"] - mean_r) / sd_r
outliers = outliers.sort_values("residual", key=abs, ascending=False)

out_path = folder / "lr_residual_outliers.csv"
outliers[
    ["County", "FIPS", "split", "My_Composite_Risk_Score", "predicted", "residual", "residual_SDs"]
].to_csv(out_path, index=False)
print(f"Wrote {out_path.name} ({len(outliers)} outliers, cutoff={cutoff:.4f})")
