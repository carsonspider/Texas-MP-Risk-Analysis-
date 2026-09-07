"""
plot_nyquist_bode.py

For each polymer (PP, PE, PCL), makes two figures:

1. Nyquist plot: Re(Z) on x-axis  Im(Z) on y-axis + control 
2. Bode plot: two panels stacked, both using log-scaled frequency on the
   x-axis
     top panel:    |Z|  vs frequency
     bottom panel: Phase(Z) vs frequency
   

"""
import pandas as pd
import matplotlib.pyplot as plt
 
MASTER_FILE = "eis_master.csv"
CONTROL_ARTIFACT_CUTOFF_HZ = 9469.4  # below this, the control reads near-zero (not real)
 
df = pd.read_csv(MASTER_FILE)
 
POLYMERS = ["PP", "PE", "PCL"]
 
 
def samples_for_polymer(polymer):
    """Return the control sample plus every sample of this polymer, sorted by concentration."""
    subset = df[(df["Polymer"] == polymer) | (df["Polymer"] == "None (control)")]
    order = subset[["Sample_ID", "Concentration_pct"]].drop_duplicates().sort_values("Concentration_pct")
    return order["Sample_ID"].tolist()
 
 
for polymer in POLYMERS:
    sample_ids = samples_for_polymer(polymer)
 
    # ---------- Nyquist plot ----------
    fig, ax = plt.subplots(figsize=(6, 6))
    for sample_id in sample_ids:
        sample = df[df["Sample_ID"] == sample_id].sort_values("Frequency_Hz")
        conc = sample["Concentration_pct"].iloc[0]
        label = "Control (0%)" if conc == 0 else f"{conc}%"
        ax.plot(sample["Re_Z_ohm"], -sample["Im_Z_ohm"], marker="o", markersize=3, label=label)
 
    ax.set_title(f"{polymer}: Nyquist Plot (all concentrations)")
    ax.set_xlabel("Re(Z) [ohms]")
    ax.set_ylabel("-Im(Z) [ohms]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{polymer}_nyquist.png", dpi=150)
    plt.close(fig)
 
    # ---------- Bode plot (magnitude + phase) ----------
    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
 
    for sample_id in sample_ids:
        sample = df[df["Sample_ID"] == sample_id].sort_values("Frequency_Hz")
        conc = sample["Concentration_pct"].iloc[0]
        label = "Control (0%)" if conc == 0 else f"{conc}%"
        ax_mag.plot(sample["Frequency_Hz"], sample["Z_mag_ohm"], marker="o", markersize=3, label=label)
        ax_phase.plot(sample["Frequency_Hz"], sample["Phase_deg"], marker="o", markersize=3, label=label)
 
    for ax in (ax_mag, ax_phase):
        ax.set_xscale("log")
        ax.grid(alpha=0.3, which="both")
 
    ax_mag.set_ylabel("|Z| [ohms]")
    ax_mag.set_title(f"{polymer}: Bode Plot (all concentrations)")
    ax_mag.legend(fontsize=8)
 
    ax_phase.set_ylabel("Phase(Z) [degrees]")
    ax_phase.set_xlabel("Frequency [Hz] (log scale)")
 
    fig.tight_layout()
    fig.savefig(f"{polymer}_bode.png", dpi=150)
    plt.close(fig)
 
    print(f"Saved {polymer}_nyquist.png and {polymer}_bode.png")
 
print("\nDone.")