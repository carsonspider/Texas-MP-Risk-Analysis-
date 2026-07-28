"""
Linear regression to predict My_Composite_Risk_Score from county-level features.
80/20 train/test split (random_state=42). Missing X values imputed with column median.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent / "ML_Ready_Sheet - ML_Ready_Sheet.csv"
OUT_DIR = Path(__file__).resolve().parent

FEATURE_COLS = [
    "Agricultural_Land_Percent",
    "Livestock_Density_per_sq_mi",
    "Irrigated_Land_Acres",
    "Interstate_Mileage_mi",
    "Road_Density_mi_per_sq_mi",
    "Biosolids_Application_Sites",
    "WWTP_Count",
    "WWTP_Total_Treatment_Capacity_MGD",
    "Population",
    "Population_Density_per_sq_mi",
    "Urbanization_Percent",
]
TARGET_COL = "My_Composite_Risk_Score"
ID_COLS = ["County", "FIPS"]
RANDOM_STATE = 42
TEST_SIZE = 0.20


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    missing = [c for c in FEATURE_COLS + [TARGET_COL] + ID_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")

    if y.isna().any():
        raise ValueError(f"Target has {y.isna().sum()} missing values; cannot train.")

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_imp, y_train)

    y_train_pred = model.predict(X_train_imp)
    y_test_pred = model.predict(X_test_imp)

    def report(split: str, y_true, y_pred) -> None:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        print(f"{split}:")
        print(f"  R²   = {r2_score(y_true, y_pred):.4f}")
        print(f"  RMSE = {rmse:.6f}")
        print(f"  MAE  = {mean_absolute_error(y_true, y_pred):.6f}")

    print(f"n = {len(df)} | train = {len(X_train)} | test = {len(X_test)}")
    print(f"Median imputation fit on train only (random_state={RANDOM_STATE})\n")
    report("Train", y_train, y_train_pred)
    print()
    report("Test", y_test, y_test_pred)

    coef_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "coefficient": model.coef_,
        }
    ).sort_values("coefficient", key=abs, ascending=False)
    coef_df.loc[len(coef_df)] = ["intercept", model.intercept_]
    coef_path = OUT_DIR / "lr_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)

    print("\nCoefficients (sorted by |coef|):")
    print(coef_df.to_string(index=False))

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

    print(f"\nWrote {coef_path.name}")
    print(f"Wrote {pred_path.name}")


if __name__ == "__main__":
    main()
