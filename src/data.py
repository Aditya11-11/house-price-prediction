"""Download / cache the Ames Housing dataset (the Kaggle 'House Prices' version, 1,460 homes)."""
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

TARGET = "SalePrice"
OPENML_ID = 42165  # "house_prices" on OpenML, identical to Kaggle's train.csv


def load_data() -> pd.DataFrame:
    """Return the raw Ames dataframe. Cached in data/ after the first download (~500 KB)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bunch = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto", data_home=str(DATA_DIR))
    df = bunch.frame.copy()
    # Turn pandas' newer string dtype into plain object so sklearn treats it predictably
    for col in df.columns:
        if df[col].dtype != "int64" and df[col].dtype != "float64":
            df[col] = df[col].astype(object).where(df[col].notna(), None)
    return df.drop(columns="Id")


def remove_known_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the handful of huge-but-cheap houses that Dean De Cock (who compiled the data)
    explicitly recommends removing: > 4,000 sq ft of living area sold for under $300k.
    These are partial sales and they drag a linear fit badly."""
    mask = (df["GrLivArea"] > 4000) & (df[TARGET] < 300_000)
    return df.loc[~mask].copy()
