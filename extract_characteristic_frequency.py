"""

For each sample, finds the frequency where Phase(Z) is most negative

Saves everything to characteristic_frequencies.csv.
"""

import pandas as pd

MASTER_FILE = "eis_master.csv"
OUTPUT_FILE = "characteristic_frequencies.csv"

df = pd.read_csv(MASTER_FILE)

rows = []

for sample_id, group in df.groupby("Sample_ID"):
    polymer = group["Polymer"].iloc[0]
    concentration = group["Concentration_pct"].iloc[0]

    # Find the row with the most negative phase
    idx_min_phase = group["Phase_deg"].idxmin()
    char_row = group.loc[idx_min_phase]


    rows.append({
        "Sample_ID": sample_id,
        "Polymer": polymer,
        "Concentration_pct": concentration,
        "Characteristic_Frequency_Hz": char_row["Frequency_Hz"],
        "Min_Phase_deg": char_row["Phase_deg"],
       
    })

result = pd.DataFrame(rows).sort_values(["Polymer", "Concentration_pct"])

print(result.to_string(index=False))

result.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved to {OUTPUT_FILE}")