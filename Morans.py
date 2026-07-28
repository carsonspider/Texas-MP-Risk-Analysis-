import os
import ssl
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from libpysal.weights import Queen
from esda.moran import Moran

# Avoid SSL certificate issues when reading the Census shapefile
ssl._create_default_https_context = ssl._create_unverified_context

# Read your county-level data
csv_path = "/Users/juliettecarson/Desktop/Plots/Texas_Soil_Risk_with_Indices.csv"
df = pd.read_csv(csv_path)

print("CSV columns:")
print(df.columns.tolist())
print("\nCSV preview:")
print(df.head())

# Read county boundaries from the Census TIGER shapefile
url = "https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_county_500k.zip"
counties = gpd.read_file(url)

# Keep only Texas counties
if "STATEFP" in counties.columns:
    counties = counties[counties["STATEFP"] == "48"]
elif "STATE_NAME" in counties.columns:
    counties = counties[counties["STATE_NAME"] == "Texas"]
else:
    raise KeyError("Could not find a Texas state identifier column in the county shapefile")

print("\nTexas counties preview:")
print(counties[["NAME", "STATEFP"]].head())

# Standardize county names so they match the CSV
# Example: "Travis County" -> "Travis"
df["County"] = (
    df["County"].astype(str)
    .str.replace(r"\s+County$", "", regex=True)
    .str.strip()
)
counties["NAME"] = counties["NAME"].astype(str).str.strip()

# Merge the spatial data with your risk data
merged = counties.merge(
    df,
    left_on="NAME",
    right_on="County",
    how="inner"
)

print(f"\nMatched counties: {len(merged)}")
print(merged[["NAME", "County"]].head())

# Use a numeric risk column for Moran's I
value_col = "Composite_Risk_Score"
if value_col not in merged.columns:
    raise KeyError(f"Column '{value_col}' not found in merged data")

merged = merged.dropna(subset=[value_col])

# Build a Queen contiguity spatial weights matrix
weights = Queen.from_dataframe(merged, geom_col="geometry")
weights.transform = "r"

risk = merged[value_col].astype(float).to_numpy()
moran = Moran(risk, weights)

print("\nMoran's I:", moran.I)
print("p-value:", moran.p_sim)

# Save the merged county data as a GeoJSON file
output_dir = "/Users/juliettecarson/Desktop/Plots"
os.makedirs(output_dir, exist_ok=True)
output_geojson = os.path.join(output_dir, "texas_counties_moran.geojson")
merged.to_file(output_geojson, driver="GeoJSON")
print(f"\nSaved merged county data to: {output_geojson}")

# Create a simple PNG containing the numeric results only
fig, ax = plt.subplots(figsize=(7, 4))
ax.axis("off")
ax.text(
    0.5,
    0.7,
    f"Moran's I: {moran.I:.4f}",
    ha="center",
    va="center",
    fontsize=16,
    weight="bold",
)
ax.text(
    0.5,
    0.35,
    f"p-value: {moran.p_sim:.3f}",
    ha="center",
    va="center",
    fontsize=14,
)
ax.text(
    0.5,
    0.05,
    f"Matched counties: {len(merged)}",
    ha="center",
    va="center",
    fontsize=12,
)

output_png = "/Users/juliettecarson/Desktop/Plots/texas_counties_moran_summary.png"
plt.savefig(output_png, dpi=300, bbox_inches="tight")
print(f"Saved numeric summary PNG to: {output_png}")

plt.close(fig)
