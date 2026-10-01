"""Predict prices for houses in a CSV (same columns as the Kaggle/Ames data, no SalePrice).

Example:
    python -m src.predict samples/example_houses.csv
"""
import argparse

import joblib
import numpy as np
import pandas as pd

from src.data import MODELS_DIR
from src.features import prepare


def predict_prices(df: pd.DataFrame) -> np.ndarray:
    model = joblib.load(MODELS_DIR / "house_price_model.joblib")
    df = df.drop(columns=["Id", "SalePrice"], errors="ignore")
    df = df.astype({c: object for c in df.columns if df[c].dtype == "string"})
    return np.expm1(model.predict(prepare(df)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="CSV file with one house per row")
    args = parser.parse_args()
    df = pd.read_csv(args.csv, keep_default_na=False, na_values=[""])
    for i, (price, row) in enumerate(zip(predict_prices(df), df.itertuples())):
        print(f"House {i + 1}: {row.Neighborhood:<8} {row.GrLivArea:>5} sq ft, built {row.YearBuilt}"
              f"  ->  ${price:,.0f}")


if __name__ == "__main__":
    main()
