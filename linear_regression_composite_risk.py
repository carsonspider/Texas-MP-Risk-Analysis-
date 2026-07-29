"""
1. Load the county CSV and select X (features) and y (target).
2. Split into 80% train / 20% test with a fixed random_state so results are reproducible.
3. Impute missing feature values with each column's *training* median
   (fit on train only; apply those medians to test — no row dropping, no test leakage).
4. Fit ordinary least squares (OLS) linear regression on the imputed training data.
5. Evaluate R², RMSE, and MAE on train and test.
6. Export:
   - lr_metrics.csv        — fit quality by split
   - lr_coefficients.csv   — raw + standardized coefficients
   - lr_predictions.csv    — actual vs predicted (+ residuals) by county


Caveats for interpretation
--------------------------
- Keep in mind highly correlated predictors (WWTP count vs capacity) 
- Multicollinearity can make individual coefficients unstable (signs/magnitudes hard to trust in isolation) even when overall predictive fit (R²) looks strong.

"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "ML_Ready_Sheet - ML_Ready_Sheet.csv"
OUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Feature set (X) — predictors of composite risk
# Names match columns in the CSV exactly.
# ---------------------------------------------------------------------------
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

# Identifiers kept in the predictions export (not used as model inputs)
ID_COLS = ["County", "FIPS"]

# Reproducibility: only the train/test split uses this seed
RANDOM_STATE = 42
TEST_SIZE = 0.20  # 20% held out for testing; 80% for training


def main() -> None:
    # Load data
    # -----------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    # Fail early if any required column is missing or renamed in the CSV
    missing = [c for c in FEATURE_COLS + [TARGET_COL] + ID_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Convert predictors/target to numeric.
    # errors="coerce" turns non-numeric / blank cells into NaN so imputation can fill them.
    X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # Target must be complete — we impute X only, never y
    if y.isna().any():
        raise ValueError(f"Target has {y.isna().sum()} missing values; cannot train.")

    # -----------------------------------------------------------------------
    # Train / test split (80 / 20)
    
    # -----------------------------------------------------------------------
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # -----------------------------------------------------------------------
    # Median imputation (train-only fit → apply to test)
    #
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)  # ndarray, same column order as FEATURE_COLS
    X_test_imp = imputer.transform(X_test)

    # -----------------------------------------------------------------------
    # Fit OLS linear regression on raw feature scales
    #
    # Minimizes sum of squared residuals on the training set.
    # -----------------------------------------------------------------------
    model = LinearRegression()
    model.fit(X_train_imp, y_train)

    # -----------------------------------------------------------------------
    # Standardized coefficients (for comparing features to each other)
    #
    # StandardScaler.fit learns mean and SD of each feature on train.
    # scaler.scale_ is the train SD per feature (population SD, ddof=0).
    # -----------------------------------------------------------------------
    scaler = StandardScaler()
    scaler.fit(X_train_imp)
    standardized_coefs = model.coef_ * scaler.scale_

    # -----------------------------------------------------------------------
    # 5. Predictions
    # -----------------------------------------------------------------------
    y_train_pred = model.predict(X_train_imp)
    y_test_pred = model.predict(X_test_imp)

    # -----------------------------------------------------------------------
    #  Evaluation metrics
    # -----------------------------------------------------------------------
    def metrics(y_true, y_pred) -> dict:
        return {
            "R2": r2_score(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": mean_absolute_error(y_true, y_pred),
        }

    train_metrics = metrics(y_train, y_train_pred)
    test_metrics = metrics(y_test, y_test_pred)

    # Save metrics table for reporting / figures
    metrics_df = pd.DataFrame(
        [
            {"split": "train", **train_metrics},
            {"split": "test", **test_metrics},
        ]
    )
    metrics_path = OUT_DIR / "lr_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Console summary
    print(f"n = {len(df)} | train = {len(X_train)} | test = {len(X_test)}")
    print(f"Median imputation fit on train only (random_state={RANDOM_STATE})\n")
    for split_name, m in [("Train", train_metrics), ("Test", test_metrics)]:
        print(f"{split_name}:")
        print(f"  R²   = {m['R2']:.4f}")
        print(f"  RMSE = {m['RMSE']:.6f}")
        print(f"  MAE  = {m['MAE']:.6f}")
        print()

    # -----------------------------------------------------------------------
    #  Coefficient table
    # -----------------------------------------------------------------------
    coef_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "coefficient": model.coef_,  # per +1 in original units of the feature
            "standardized_coefficient": standardized_coefs,  # per +1 SD (train)
        }
    ).sort_values("standardized_coefficient", key=abs, ascending=False)

    # Append intercept as its own row (raw only)
    coef_df = pd.concat(
        [
            coef_df,
            pd.DataFrame(
                [
                    {
                        "feature": "intercept",
                        "coefficient": model.intercept_,
                        "standardized_coefficient": np.nan,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    coef_path = OUT_DIR / "lr_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)

    print("Coefficients (sorted by |standardized_coefficient|):")
    print(coef_df.to_string(index=False))

    # -----------------------------------------------------------------------
    # County-level predictions export
    #
    # -----------------------------------------------------------------------
    pred_frames = []
    for split_name, idx, y_true, y_pred in [
        ("train", idx_train, y_train, y_train_pred),
        ("test", idx_test, y_test, y_test_pred),
    ]:
        part = df.loc[idx, ID_COLS].copy()
        part["split"] = split_name
        part[TARGET_COL] = y_true.values
        part["predicted"] = y_pred
        part["residual"] = part[TARGET_COL] - part["predicted"]
        pred_frames.append(part)

    pred_df = pd.concat(pred_frames, ignore_index=True)
    pred_path = OUT_DIR / "lr_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    print(f"\nWrote {metrics_path.name}")
    print(f"Wrote {coef_path.name}")
    print(f"Wrote {pred_path.name}")


# Run only when executed as a script (not when imported as a module)
if __name__ == "__main__":
    main()
