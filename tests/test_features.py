import numpy as np
import pandas as pd

from src.data import load_data
from src.features import NeighborhoodMedianImputer, clean, prepare


def test_absent_features_are_not_left_missing():
    df = clean(load_data())
    assert df["PoolQC"].isna().sum() == 0
    assert (df["PoolQC"] == "None").sum() > 1400  # almost nobody has a pool
    assert df["GarageArea"].isna().sum() == 0


def test_engineered_columns_exist_and_are_sane():
    df = prepare(load_data().head(50))
    for col in ["TotalSF", "TotalBath", "HouseAge", "QualxArea", "HasGarage"]:
        assert col in df
    assert (df["HouseAge"] >= 0).all()
    assert df["ExterQual"].between(0, 5).all()


def test_neighborhood_imputer_uses_group_median():
    X = pd.DataFrame({"Neighborhood": ["A", "A", "A", "B"], "LotFrontage": [10.0, 20.0, np.nan, 100.0]})
    out = NeighborhoodMedianImputer().fit(X).transform(X)
    assert out.loc[2, "LotFrontage"] == 15.0


def test_prediction_is_in_a_plausible_range():
    from src.predict import predict_prices

    prices = predict_prices(load_data().drop(columns="SalePrice").head(10))
    assert ((prices > 30_000) & (prices < 800_000)).all()
