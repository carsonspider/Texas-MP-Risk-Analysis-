"""
plot_low_frequency_zoom.py


This script redraws Z and phase using only frequencies at or below
LOW_FREQ_MAX_HZ (100 Hz). can Change that number to zoom tighter or wider.

PCL at 5%, 10%, and 20% was recorded for three cycles. Only cycle 1 is
plotted, so those curves are not drawn three times on top of each other.
"""

import pandas as pd
import matplotlib.pyplot as plt

MASTER_FILE = "eis_master.csv"
LOW_FREQ_MAX_HZ = 100

df = pd.read_csv(MASTER_FILE)
df = df[df["Cycle_number"] == 1]
df = df[df["Frequency_Hz"] <= LOW_FREQ_MAX_HZ]

for polymer in ["PP", "PE", "PCL"]:
    subset = df[df["Polymer"] == polymer]
    sample_order = (
        subset[["Sample_ID", "Concentration_pct"]]
        .drop_duplicates()
        .sort_values("Concentration_pct")
    )

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    for sample_id in sample_order["Sample_ID"]:
        sample = subset[subset["Sample_ID"] == sample_id].sort_values("Frequency_Hz")
        conc = sample["Concentration_pct"].iloc[0]
        label = f"{conc:g}%"
        ax_mag.plot(sample["Frequency_Hz"], sample["Z_mag_ohm"], marker="o", markersize=3, label=label)
        ax_phase.plot(sample["Frequency_Hz"], sample["Phase_deg"], marker="o", markersize=3, label=label)

    for ax in (ax_mag, ax_phase):
        ax.set_xscale("log")
        ax.grid(alpha=0.3, which="both")

    ax_mag.set_ylabel("|Z| [ohms]")
    ax_mag.set_title(f"{polymer}: low-frequency zoom (≤ {LOW_FREQ_MAX_HZ:g} Hz, control omitted)")
    ax_mag.legend(fontsize=8)

    ax_phase.set_ylabel("Phase(Z) [degrees]")
    ax_phase.set_xlabel("Frequency [Hz] (log scale)")

    fig.tight_layout()
    fig.savefig(f"{polymer}_low_frequency_zoom.png", dpi=150)
    plt.close(fig)
    print(f"Saved {polymer}_low_frequency_zoom.png")
