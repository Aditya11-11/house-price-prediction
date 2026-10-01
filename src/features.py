"""Cleaning + feature engineering for Ames.

The big realisation here: most of the "missing" values are not missing at all.
In this dataset NA in PoolQC means "no pool", NA in GarageType means "no garage",
and so on. Treating those as real categories is worth more than any clever imputer.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Columns where NA literally means "this house doesn't have one"
NONE_MEANS_ABSENT = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]
ZERO_MEANS_ABSENT = [
    "GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
]

# Ordinal quality scales -> numbers, so the model knows Ex > Gd > TA > Fa > Po
QUALITY_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
QUALITY_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]

# Area-type columns with long right tails; log1p makes them friendlier to a linear model
SKEWED = [
    "LotArea", "GrLivArea", "1stFlrSF", "TotalSF", "MasVnrArea",
    "WoodDeckSF", "OpenPorchSF", "TotalPorchSF", "LotFrontage",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NONE_MEANS_ABSENT:
        df[col] = df[col].fillna("None")
    for col in ZERO_MEANS_ABSENT:
        df[col] = df[col].fillna(0)
    # MSSubClass is a dwelling-type code (20, 60, 120...) — numbers, but not numeric
    df["MSSubClass"] = df["MSSubClass"].astype(str)
    df["MoSold"] = df["MoSold"].astype(str)
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in QUALITY_COLS:
        df[col] = df[col].fillna("None").map(QUALITY_MAP).fillna(0)

    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"] = df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    df["TotalPorchSF"] = df[["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]].sum(axis=1)
    df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    df["YearsSinceRemodel"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    df["IsNew"] = (df["YrSold"] == df["YearBuilt"]).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
    # Quality matters more in a big house than a small one
    df["QualxArea"] = df["OverallQual"] * df["GrLivArea"]

    for col in SKEWED:
        df[col] = np.log1p(df[col])
    # Utilities is "AllPub" for all but one house — zero information
    return df.drop(columns=["Utilities"], errors="ignore")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    return engineer(clean(df))


class NeighborhoodMedianImputer(BaseEstimator, TransformerMixin):
    """Fill LotFrontage with the median of the house's neighbourhood.

    Houses on the same street tend to have similar frontage, so this beats a
    global median. Fitted on the training data only to avoid leakage.
    """

    def __init__(self, column="LotFrontage", group="Neighborhood"):
        self.column = column
        self.group = group

    def fit(self, X, y=None):
        self.medians_ = X.groupby(self.group)[self.column].median()
        self.global_median_ = X[self.column].median()
        return self

    def transform(self, X):
        X = X.copy()
        fill = X[self.group].map(self.medians_).fillna(self.global_median_)
        X[self.column] = X[self.column].fillna(fill)
        return X
