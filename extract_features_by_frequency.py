"""
extract_features_by_frequency.py

WHAT THIS DOES (simple version):
For every sample, look at three frequency checkpoints (high, mid, low --
you can change the numbers below). At each checkpoint, grab three values:
Re(Z), |Z|, and Phase. Put it all in one table so you can compare samples
side by side at the same frequencies.

This replaces the older "compare_by_concentration.py" but now grabs
Re(Z) and Phase too, not just |Z|.
"""

import pandas as pd

MASTER_FILE = "eis_master.csv"
OUTPUT_FILE = "features_by_frequency.csv"

# --- Adjustable frequency checkpoints (change these anytime) ---
HIGH_FREQ_HZ = 50000
MID_FREQ_HZ = 1000
LOW_FREQ_HZ = 1

df = pd.read_csv(MASTER_FILE)


def closest_row(group, target_freq):
    """Find the row in this sample closest to target_freq."""
    idx = (group["Frequency_Hz"] - target_freq).abs().idxmin()
    return group.loc[idx]


rows = []
for sample_id, group in df.groupby("Sample_ID"):
    polymer = group["Polymer"].iloc[0]
    concentration = group["Concentration_pct"].iloc[0]

    high = closest_row(group, HIGH_FREQ_HZ)
    mid = closest_row(group, MID_FREQ_HZ)
    low = closest_row(group, LOW_FREQ_HZ)

    rows.append({
        "Sample_ID": sample_id,
        "Polymer": polymer,
        "Concentration_pct": concentration,

        "High_actual_Hz": high["Frequency_Hz"],
        "High_ReZ_ohm": high["Re_Z_ohm"],
        "High_Zmag_ohm": high["Z_mag_ohm"],
        "High_Phase_deg": high["Phase_deg"],

        "Mid_actual_Hz": mid["Frequency_Hz"],
        "Mid_ReZ_ohm": mid["Re_Z_ohm"],
        "Mid_Zmag_ohm": mid["Z_mag_ohm"],
        "Mid_Phase_deg": mid["Phase_deg"],

        "Low_actual_Hz": low["Frequency_Hz"],
        "Low_ReZ_ohm": low["Re_Z_ohm"],
        "Low_Zmag_ohm": low["Z_mag_ohm"],
        "Low_Phase_deg": low["Phase_deg"],
    })

result = pd.DataFrame(rows).sort_values(["Polymer", "Concentration_pct"])
print(result.to_string(index=False))

result.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved to {OUTPUT_FILE}")
