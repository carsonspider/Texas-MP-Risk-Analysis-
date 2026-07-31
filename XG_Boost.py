'''
exports feature importance weight and gain (png plots)
csv of R^2, RMSE, MAE

'''

import os
import pandas as panda
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder 
from xgboost import XGBRegressor, plot_importance
  #0-1, not categorical
#XGBClassifier for yes/ no categories 
#plot importance for feature importance 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import altair as alt


#dataset
df = panda.read_csv('/Users/juliettecarson/Desktop/HIRES Research/ML/ML_Ready_Sheet.csv', delimiter=',')

# clear dataset (drop flood freq bc empty)
FEATURE_COLS = [
    "Agricultural_Land_Percent",       # cropland / ag land %
    "Livestock_Density_per_sq_mi",     # livestock density
    "Irrigated_Land_Acres",            # irrigated land (acres; no % column in CSV)
    "Interstate_Mileage_mi",           # interstate length (miles)
    "Road_Density_mi_per_sq_mi",       # road density
    "Biosolids_Application_Sites",     # biosolid application sites
    "WWTP_Count",                      # wastewater treatment plant count
    "WWTP_Total_Treatment_Capacity_MGD",  # WWTP total capacity (MGD)
    "Population",                      # county population
    "Population_Density_per_sq_mi",    # population density
    "Urbanization_Percent",            # urbanization %
]

# Target (Y) —  composite risk score to predict
TARGET_COL = "My_Composite_Risk_Score"


print(df.head())

X = df[FEATURE_COLS].copy() #copy to avoid modifying original

# Impute missing values before modeling so feature columns are complete and consistent
missing_before = X.isnull().sum()
print("Missing values before imputation:")
print(missing_before[missing_before > 0])

X = X.fillna(X.median(numeric_only=True))
X = X.fillna(0)

print("Missing values after imputation:")
print(X.isnull().sum().sum())

Y = df[TARGET_COL].copy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# 80% for training, 20% for testing

model = XGBRegressor(
    n_estimators = 200, # 200 small decision trees 
    max_depth = 3, 
    learning_rate = 0.05, # thus 200 trees
    subsample = 0.8, #each trees trains on a random 80% of rows 
    colsample_bytree = 0.8, # each tree only considers 80% of columns
    #sirfaces secondary patterns, more diverse trees 
    # prevents overfitting 
    random_state = 42
)

model.fit(X_train, Y_train)
##goes through the following loop:
'''
    starts w baseline guess for y, calculates residuals 
    the next tree is trained to correct the residuals (target as residuals, not y)
    the prediction is  = previous_prediction + learning_rate * next_tree_prediction
    continues n_extimator times 
    output is the final prediction
'''
pred = model.predict(X_test)
#runs the model again with the remaining 20% X test data, never seen 

#---------
#evaluate accuracy 
r2 = r2_score(Y_test, pred)
rmse = root_mean_squared_error(Y_test, pred)
mae = mean_absolute_error(Y_test, pred)

print("R2 Score: ", r2)
print("Root Mean Squared Error: ", rmse)
print("Mean Absolute Error: ", mae)

results = panda.DataFrame({
    "Metric": ["R2", "RMSE", "MAE"],
    "Value": [r2, rmse, mae]
})

results.to_csv('/Users/juliettecarson/Desktop/HIRES Research/XG_Boost_Results.csv', index=False)

#---------
#FEATURE IMPORTANCE WEIGHT
plt.figure(figsize=(10, 8))
# Plot weight-based feature importance, built in
plot_importance(model, importance_type='weight')
# weight: the number of times a feature is used to split the data across all trees
plt.title("Feature Importance XG Boosting (Weight)")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.savefig('/Users/juliettecarson/Desktop/HIRES Research/ML/figures/XG_Boosting_feature_importance_weight.png')
plt.show()

#FEATURE IMPORTANCE WEIGHT
plt.figure(figsize=(10, 8))
plot_importance(model, importance_type='gain')
plt.title("Feature Importance XG Boosting (Gain)")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.savefig('/Users/juliettecarson/Desktop/HIRES Research/ML/figures/XG_Boosting_feature_importance_gain.png')
plt.show()
#average improvement in accuracy each time that feature is used to split

#-----
#plot residuals v predicted 
residuals = Y_test - pred
plt.figure(figsize=(10, 8))
plt.scatter(pred, residuals)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='--')
plt.savefig('/Users/juliettecarson/Desktop/HIRES Research/ML/figures/XG_Boosting_residue_plot.png')
plt.show()

#residual distribution
plt.figure(figsize=(10, 8))
plt.hist(residuals, bins=30)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.savefig('/Users/juliettecarson/Desktop/HIRES Research/ML/figures/XG_Boosting_residue_distribution.png')
plt.show()

#----- 
# export gain-based importance for calibrated heat map
raw_gain_importance = model.get_booster().get_score(importance_type='gain')

# normalize to sum to 1 
gain_importance_series = panda.Series(raw_gain_importance)
gain_importance_series = gain_importance_series.reindex(FEATURE_COLS, fill_value=0)
normalized_gain = gain_importance_series / gain_importance_series.sum()

importance_df = panda.DataFrame({
    "Feature": FEATURE_COLS,
    "Gain_Importance": gain_importance_series.values,
    "Gain_Importance_Normalized": normalized_gain.values,
    "Gain_Importance_Normalized_Percent": normalized_gain.values * 100
})

importance_df.to_csv('/Users/juliettecarson/Desktop/HIRES Research/ML/feature_importance.csv', index=False)


weighted_feature_matrix = panda.DataFrame(index=df.index)
for feature in FEATURE_COLS:
    feature_weight = importance_df.loc[importance_df['Feature'] == feature, 'Gain_Importance_Normalized'].iloc[0]
    weighted_feature_matrix[f'{feature}_Weighted'] = df[feature].fillna(df[feature].median()) * feature_weight

weighted_export = panda.concat([
    df[['County', 'FIPS']].copy(),
    weighted_feature_matrix
], axis=1)

# ----  feature-level gain importance into pathway indices ----
groups = {
    'Agriculture_Index': ['Agricultural_Land_Percent','Livestock_Density_per_sq_mi','Irrigated_Land_Acres'],
    'Wastewater_Index': ['Biosolids_Application_Sites','WWTP_Count','WWTP_Total_Treatment_Capacity_MGD'],
    'Transportation_Index': ['Interstate_Mileage_mi','Road_Density_mi_per_sq_mi'],
    'Demographics_Index': ['Population','Population_Density_per_sq_mi','Urbanization_Percent']
}

imp = importance_df.set_index('Feature')['Gain_Importance_Normalized']

group_weights = {k: imp[cols].sum() for k, cols in groups.items()}
total = sum(group_weights.values())
group_weights = {k: v/total for k, v in group_weights.items()}

print("ML-derived index weights:")
for k, v in group_weights.items():
    print(f"  {k}: {v:.3f} ({v*100:.1f}%)")

# ----  weighted composite to  254 counties ----
df['ML_Weighted_Composite'] = (
    df['Agriculture_Index'].fillna(0)    * group_weights['Agriculture_Index'] +
    df['Wastewater_Index'].fillna(0)     * group_weights['Wastewater_Index'] +
    df['Transportation_Index'].fillna(0) * group_weights['Transportation_Index'] +
    df['Demographics_Index'].fillna(0)   * group_weights['Demographics_Index']
)

# ---- export for QGIS----
OUTPUT_DIR = '/Users/juliettecarson/Desktop/HIRES Research/ML/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
weighted_export.to_csv(os.path.join(OUTPUT_DIR, 'Weighted_X_Values.csv'), index=False)
df.to_csv(os.path.join(OUTPUT_DIR, 'ML_Weighted_Composite.csv'), index=False)
print(weighted_export[['County','FIPS'] + [c for c in weighted_export.columns if c.endswith('_Weighted')]].head())
print(df[['County','FIPS','My_Composite_Risk_Score','ML_Weighted_Composite']].head())
