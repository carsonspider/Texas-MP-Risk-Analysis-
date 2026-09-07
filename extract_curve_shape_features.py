"""
extract_curve_shape_features.py


1. Arc diameter (Nyquist)  how "big" the arc is, estimated as
   (largest Re(Z) minus smallest Re(Z)) across the sweep. Bigger number =
   bigger arc. 

2. Low-frequency tail slope (Nyquist) at the lowest few frequencies,
   is the curve still curving, or does it go in a straight diagonal line?
   We fit a straight line to the last few points and report its slope.

3. Bode transition point 

"""

import pandas as pd
import numpy as np
import subprocess

MASTER_FILE = "eis_master.csv"
OUTPUT_FILE = "curve_shape_features.csv"

N_LOW_FREQ_POINTS_FOR_SLOPE = 5   # how many of the lowest-frequency points to use for the tail slope
N_HIGH_FREQ_POINTS_FOR_PLATEAU = 5  # how many of the highest-frequency points define the "flat" baseline
TRANSITION_THRESHOLD = 1.5        # transition = where |Z| exceeds this multiple of the plateau

df = pd.read_csv(MASTER_FILE)

rows = []

for sample_id, group in df.groupby("Sample_ID"):
    polymer = group["Polymer"].iloc[0]
    concentration = group["Concentration_pct"].iloc[0]

    sorted_by_freq = group.sort_values("Frequency_Hz")  # ascending: lowest freq first

    arc_diameter = sorted_by_freq["Re_Z_ohm"].max() - sorted_by_freq["Re_Z_ohm"].min()

    tail_points = sorted_by_freq.head(N_LOW_FREQ_POINTS_FOR_SLOPE)
    x = tail_points["Re_Z_ohm"].values
    y = -tail_points["Im_Z_ohm"].values  # standard Nyquist convention
    if len(x) >= 2 and np.ptp(x) != 0:
        slope, _ = np.polyfit(x, y, 1)
    else:
        slope = np.nan

    high_freq_points = sorted_by_freq.tail(N_HIGH_FREQ_POINTS_FOR_PLATEAU)  # highest frequencies
    plateau_value = high_freq_points["Z_mag_ohm"].mean()

    transition_freq = np.nan
    transition_zmag = np.nan
    # scan from highest frequency down to lowest, find first point that clearly rises above plateau
    scan_order = sorted_by_freq.sort_values("Frequency_Hz", ascending=False)
    for _, point in scan_order.iterrows():
        if point["Z_mag_ohm"] > TRANSITION_THRESHOLD * plateau_value:
            transition_freq = point["Frequency_Hz"]
            transition_zmag = point["Z_mag_ohm"]
            break

    rows.append({
        "Sample_ID": sample_id,
        "Polymer": polymer,
        "Concentration_pct": concentration,
        "Arc_Diameter_ohm": arc_diameter,
        "Low_Freq_Tail_Slope": slope,
        "Bode_Plateau_ohm": plateau_value,
        "Bode_Transition_Freq_Hz": transition_freq,
        "Bode_Transition_Zmag_ohm": transition_zmag,
    })

result = pd.DataFrame(rows).sort_values(["Polymer", "Concentration_pct"])
print(result.to_string(index=False))

result.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved to {OUTPUT_FILE}")

# --- Regenerate the Nyquist and Bode plots using the existing script ---
print("\nRegenerating Nyquist and Bode plots...")
subprocess.run(["python3", "plot_nyquist_bode.py"])
