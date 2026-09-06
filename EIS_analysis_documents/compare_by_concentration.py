"""
compare_by_concentration.py

1. Loads eis_master.csv 
2. Picks threes frequencies to compare at (set below) - low, med, high
3. For every sample, finds the row closest to that frequency and takes |Z| value 
4. Builds three  tables -- one for PP, one for PE, one for PCL
   each showing Concentration vs Impedance, with the control (0%)
   included in all three.
5. Prints each table and saves them as CSV files.
"""
import pandas as pd
 
MASTER_FILE = "eis_master.csv"
 
TARGET_FREQS_HZ = {
    "High_50000Hz": 50000,
    "Mid_1000Hz": 1000,
    "Low_1Hz": 1,
}
 
df = pd.read_csv(MASTER_FILE)
 
 
def closest_value(group, target_freq):
    """Find the row in this sample's data closest to target_freq, return its |Z|."""
    idx = (group["Frequency_Hz"] - target_freq).abs().idxmin()
    return group.loc[idx, "Z_mag_ohm"]
 
 
sample_info = df[["Sample_ID", "Polymer", "Concentration_pct"]].drop_duplicates()
 
#get its |Z| at each of the three target frequencies
rows = []
for sample_id, group in df.groupby("Sample_ID"):
    row = {"Sample_ID": sample_id}
    for label, freq in TARGET_FREQS_HZ.items():
        row[label] = closest_value(group, freq)
    rows.append(row)
 
impedance_table = pd.DataFrame(rows)
 
# attach Polymer/Concentration info
full_table = sample_info.merge(impedance_table, on="Sample_ID")
 
control = full_table[full_table["Polymer"] == "None (control)"]
 
for polymer in ["PP", "PE", "PCL"]:
    this_polymer = full_table[full_table["Polymer"] == polymer]
    table = pd.concat([control, this_polymer]).sort_values("Concentration_pct")
 
    print(f"\n--- {polymer} ---")
    cols = ["Concentration_pct"] + list(TARGET_FREQS_HZ.keys())
    print(table[cols].to_string(index=False))
 
    table.to_csv(f"comparison_{polymer}.csv", index=False)
 
print("\nDone. Saved comparison_PP.csv, comparison_PE.csv, comparison_PCL.csv")