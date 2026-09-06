"""
This script processes EIS (Electrochemical Impedance Spectroscopy) data from multiple sheets 
in an Excel file and combines them into a single, standardized CSV file.
"""

import pandas as pd

INVENTORY_FILE = "sample_inventory.xlsx"
RAW_FILE = "eis_AutoRecovered_.xlsx"
OUTPUT_FILE = "eis_master.csv"

SKIP_SHEETS = [
    "PE graph", "PP graph", "20% graph", "5% graph", "10% graph", "1% graph",  # chart tabs, no data
    "Combined PP PE",  
]

CLEAN_COLUMNS = [
    "Frequency_Hz",
    "Re_Z_ohm",
    "Im_Z_ohm",     
    "Z_mag_ohm",
    "Phase_deg",
    "Time_s",
    "Cycle_number",
]


def main():
  
    inventory = pd.read_excel(INVENTORY_FILE, sheet_name="Sample Inventory")

    
    raw_sheets = pd.read_excel(RAW_FILE, sheet_name=None)

    all_tables = [] 

    for sheet_name, df in raw_sheets.items():

        if sheet_name in SKIP_SHEETS:
            continue 

        df = df.copy()
        df.columns = CLEAN_COLUMNS

        match = inventory.loc[inventory["Sheet_Name"] == sheet_name]

        if match.empty:
            print(f"WARNING: '{sheet_name}' has no matching row in the inventory. Skipping.")
            continue

        df["Sample_ID"] = match["Sample_ID"].values[0]
        df["Polymer"] = match["Polymer"].values[0]
        df["Concentration_pct"] = match["Concentration_pct"].values[0]
        df["Sheet_Name"] = sheet_name

        all_tables.append(df)

    master = pd.concat(all_tables, ignore_index=True)

    master.to_csv(OUTPUT_FILE, index=False)

    print(f"Done. Saved {len(master)} rows from {master['Sample_ID'].nunique()} samples to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()