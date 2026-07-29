from pathlib import Path
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the county data (same folder as this script)
df = pd.read_csv(Path(__file__).resolve().parent / "ML_Ready_Sheet - ML_Ready_Sheet.csv")

# Predictors to check for multicollinearity
feature_cols = ['Agricultural_Land_Percent','Irrigated_Land_Acres','Livestock_Density_per_sq_mi',
                'Greenhouse_Nursery_Operations','WWTP_Count','WWTP_Total_Treatment_Capacity_MGD',
                'Biosolids_Application_Sites','Urbanization_Percent','Road_Density_mi_per_sq_mi',
                'Annual_Average_Daily_Traffic','Interstate_Mileage_mi','Population',
                'Population_Density_per_sq_mi']

# Keep only complete rows (VIF needs no missing values)
X = df[feature_cols].apply(pd.to_numeric, errors="coerce").dropna()

# VIF > 5–10 suggests multicollinearity
vif_df = pd.DataFrame({
    'feature': X.columns,
    'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})

# Highest VIF first
print(vif_df.sort_values('VIF', ascending=False))
