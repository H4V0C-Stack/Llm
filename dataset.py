"""
Build report table from Ames Housing (Kaggle-style or OpenML-style columns).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# Kaggle train.csv names -> internal (OpenML) names
KAGGLE_TO_INTERNAL = {
    "MSSubClass": "MS_SubClass",
    "LotFrontage": "Lot_Frontage",
    "LotArea": "Lot_Area",
    "LotShape": "Lot_Shape",
    "LandContour": "Land_Contour",
    "Utilities": "Utilities",
    "LotConfig": "Lot_Config",
    "LandSlope": "Land_Slope",
    "Neighborhood": "Neighborhood",
    "Condition1": "Condition_1",
    "Condition2": "Condition_2",
    "BldgType": "Bldg_Type",
    "HouseStyle": "House_Style",
    "OverallQual": "Overall_Qual",
    "OverallCond": "Overall_Cond",
    "YearBuilt": "Year_Built",
    "YearRemodAdd": "Year_Remod_Add",
    "RoofStyle": "Roof_Style",
    "RoofMatl": "Roof_Matl",
    "Exterior1st": "Exterior_1st",
    "Exterior2nd": "Exterior_2nd",
    "MasVnrType": "Mas_Vnr_Type",
    "MasVnrArea": "Mas_Vnr_Area",
    "ExterQual": "Exter_Qual",
    "ExterCond": "Exter_Cond",
    "Foundation": "Foundation",
    "BsmtQual": "Bsmt_Qual",
    "BsmtCond": "Bsmt_Cond",
    "BsmtExposure": "Bsmt_Exposure",
    "BsmtFinType1": "BsmtFin_Type_1",
    "BsmtFinSF1": "BsmtFin_SF_1",
    "BsmtFinType2": "BsmtFin_Type_2",
    "BsmtFinSF2": "BsmtFin_SF_2",
    "BsmtUnfSF": "Bsmt_Unf_SF",
    "TotalBsmtSF": "Total_Bsmt_SF",
    "Heating": "Heating",
    "HeatingQC": "Heating_QC",
    "CentralAir": "Central_Air",
    "Electrical": "Electrical",
    "FirstFlrSF": "First_Flr_SF",
    "SecondFlrSF": "Second_Flr_SF",
    "LowQualFinSF": "Low_Qual_Fin_SF",
    "GrLivArea": "Gr_Liv_Area",
    "BsmtFullBath": "Bsmt_Full_Bath",
    "BsmtHalfBath": "Bsmt_Half_Bath",
    "FullBath": "Full_Bath",
    "HalfBath": "Half_Bath",
    "BedroomAbvGr": "Bedroom_AbvGr",
    "KitchenAbvGr": "Kitchen_AbvGr",
    "KitchenQual": "Kitchen_Qual",
    "TotRmsAbvGrd": "TotRms_AbvGrd",
    "Functional": "Functional",
    "Fireplaces": "Fireplaces",
    "FireplaceQu": "Fireplace_Qu",
    "GarageType": "Garage_Type",
    "GarageFinish": "Garage_Finish",
    "GarageCars": "Garage_Cars",
    "GarageArea": "Garage_Area",
    "GarageQual": "Garage_Qual",
    "GarageCond": "Garage_Cond",
    "PavedDrive": "Paved_Drive",
    "WoodDeckSF": "Wood_Deck_SF",
    "OpenPorchSF": "Open_Porch_SF",
    "EnclosedPorch": "Enclosed_Porch",
    "ThreeSsnPorch": "Three_season_porch",
    "ScreenPorch": "Screen_Porch",
    "PoolArea": "Pool_Area",
    "PoolQC": "Pool_QC",
    "Fence": "Fence",
    "MiscFeature": "Misc_Feature",
    "MiscVal": "Misc_Val",
    "MoSold": "Mo_Sold",
    "YrSold": "Year_Sold",
    "SaleType": "Sale_Type",
    "SaleCondition": "Sale_Condition",
    "SalePrice": "Sale_Price",
}


def normalize_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Kaggle columns to internal names when present."""
    rename = {k: v for k, v in KAGGLE_TO_INTERNAL.items() if k in df.columns}
    if rename:
        df = df.rename(columns=rename)
    return df


def load_raw(train_path: Path) -> pd.DataFrame:
    df = pd.read_csv(train_path)
    df = normalize_raw_columns(df)
    required = [
        "Sale_Price",
        "Neighborhood",
        "Year_Sold",
        "Bldg_Type",
        "Gr_Liv_Area",
        "Garage_Cars",
        "Condition_1",
        "Sale_Type",
        "Sale_Condition",
        "Overall_Qual",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"train.csv missing columns: {missing}")
    return df


def assign_product_category(row: pd.Series) -> str:
    """Three categories: house, apartment, studio (small living area)."""
    b = str(row["Bldg_Type"])
    area = float(row["Gr_Liv_Area"]) if pd.notna(row["Gr_Liv_Area"]) else np.nan
    if b in ("Duplex",):
        return "apartment"
    if pd.notna(area) and area <= 550:
        return "studio"
    return "house"


def assign_discount_flag(row: pd.Series) -> int:
    """1 = non-standard sale (proxy for developer discount / special terms)."""
    cond = str(row["Sale_Condition"])
    st = str(row["Sale_Type"])
    if cond != "Normal":
        return 1
    if st in ("COD", "ConLI", "ConLD", "ConLw", "CWD"):
        return 1
    return 0


QUAL_TO_NUM = {
    "Very_Poor": 1,
    "Poor": 2,
    "Fair": 3,
    "Below_Average": 4,
    "Average": 5,
    "Above_Average": 6,
    "Good": 7,
    "Very_Good": 8,
    "Excellent": 9,
    "Very_Excellent": 10,
}


def overall_qual_numeric(series: pd.Series) -> pd.Series:
    """Map OpenML string labels or Kaggle 1–10 integers to numeric scores."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return series.map(QUAL_TO_NUM)


def build_report_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["Garage_Cars"] = df["Garage_Cars"].fillna(0).astype(int)
    df["Overall_Qual"] = overall_qual_numeric(df["Overall_Qual"])

    df["Price"] = df["Sale_Price"].astype(float)
    df["Region"] = df["Neighborhood"].astype(str)

    counts = df.groupby(["Region", "Year_Sold"], observed=True).size().rename("Sales").reset_index()
    df = df.merge(counts, on=["Region", "Year_Sold"], how="left")

    df["Product_Category"] = df.apply(assign_product_category, axis=1)
    df["Discount"] = df.apply(
        lambda r: f"{r['Sale_Condition']}|{r['Sale_Type']}", axis=1
    )
    df["Discount_Flag"] = df.apply(assign_discount_flag, axis=1).astype(int)

    med_region = df.groupby("Region", observed=True)["Price"].transform("median")
    df["Price_Elasticity"] = (df["Price"] - med_region) / med_region.replace(0, np.nan)

    reg_year_med = (
        df.groupby(["Region", "Year_Sold"], observed=True)["Price"]
        .median()
        .reset_index()
        .sort_values(["Region", "Year_Sold"])
    )
    reg_year_med["prev_med"] = reg_year_med.groupby("Region")["Price"].shift(1)
    reg_year_med["Sales_Change_Pct"] = (
        (reg_year_med["Price"] - reg_year_med["prev_med"]) / reg_year_med["prev_med"]
    )
    df = df.merge(
        reg_year_med[["Region", "Year_Sold", "Sales_Change_Pct"]],
        on=["Region", "Year_Sold"],
        how="left",
    )

    out = df[
        [
            "Price",
            "Sales",
            "Region",
            "Product_Category",
            "Discount",
            "Discount_Flag",
            "Price_Elasticity",
            "Sales_Change_Pct",
            "Year_Sold",
            "Garage_Cars",
            "Condition_1",
            "Overall_Qual",
        ]
    ].copy()
    return out


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_sqlite(df: pd.DataFrame, db_path: Path, table: str = "sales") -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table, conn, if_exists="replace", index=False)
    finally:
        conn.close()


def ensure_openml_parquet_fallback(raw_dir: Path) -> Path:
    """If train.csv is missing, download OpenML parquet into train.csv (needs pyarrow)."""
    train_csv = raw_dir / "train.csv"
    if train_csv.exists() and train_csv.stat().st_size > 0:
        return train_csv
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = "https://data.openml.org/datasets/0004/41211/dataset_41211.pq"
    try:
        df = pd.read_parquet(url)
    except ImportError as exc:
        raise ImportError(
            "Install pyarrow to download the default dataset, or place Kaggle train.csv in data/raw/"
        ) from exc
    df.to_csv(train_csv, index=False)
    return train_csv
