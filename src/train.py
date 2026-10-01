"""Train Linear Regression, Ridge and Lasso on Ames and compare them.

The model predicts log(SalePrice). That makes errors relative (a $20k miss on a
$100k house matters more than on a $500k house) and it's also how the Kaggle
competition scores submissions.

Run:  python -m src.train
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import FIGURES_DIR, MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT, TARGET, load_data, remove_known_outliers
from src.features import NeighborhoodMedianImputer, prepare

RANDOM_STATE = 42


def make_preprocessor():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
    ])
    columns = ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include=np.number)),
        ("cat", categorical, make_column_selector(dtype_include=object)),
    ])
    return Pipeline([("lot_frontage", NeighborhoodMedianImputer()), ("columns", columns)])


def make_models():
    alphas = np.logspace(-4, 3, 60)
    return {
        "Baseline (median)": DummyRegressor(strategy="median"),
        "Linear Regression": LinearRegression(),
        "Ridge": RidgeCV(alphas=alphas),
        "Lasso": LassoCV(alphas=np.logspace(-5, -1, 40), max_iter=50_000, cv=5, random_state=RANDOM_STATE),
    }


def evaluate(y_true_log, y_pred_log):
    y_true, y_pred = np.expm1(y_true_log), np.expm1(y_pred_log)
    return {
        "rmse_log": round(float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))), 4),
        "rmse_dollars": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 0),
        "mae_dollars": round(float(mean_absolute_error(y_true, y_pred)), 0),
        "r2": round(float(r2_score(y_true_log, y_pred_log)), 4),
    }


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_data()
    train_raw, test_raw = train_test_split(raw, test_size=0.2, random_state=RANDOM_STATE)
    # Keep a few untouched raw rows around so predict.py has something to chew on
    samples_dir = PROJECT_ROOT / "samples"
    samples_dir.mkdir(exist_ok=True)
    test_raw.head(5).drop(columns=TARGET).to_csv(samples_dir / "example_houses.csv", index=False)

    n_before = len(train_raw)
    train_raw = remove_known_outliers(train_raw)  # only from training data!
    print(f"Removed {n_before - len(train_raw)} outliers from the training set")

    X_train, y_train = prepare(train_raw.drop(columns=TARGET)), np.log1p(train_raw[TARGET])
    X_test, y_test = prepare(test_raw.drop(columns=TARGET)), np.log1p(test_raw[TARGET])

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results, fitted = {}, {}
    for name, model in make_models().items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
        cv_rmse = -cross_val_score(pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
        pipe.fit(X_train, y_train)
        test_metrics = evaluate(y_test, pipe.predict(X_test))
        results[name] = {"cv_rmse_log_mean": round(float(cv_rmse.mean()), 4),
                         "cv_rmse_log_std": round(float(cv_rmse.std()), 4), **test_metrics}
        if hasattr(model, "alpha_"):
            results[name]["alpha"] = float(pipe.named_steps["model"].alpha_)
        fitted[name] = pipe
        print(f"{name:<20} CV RMSE(log) {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f} | "
              f"test RMSE(log) {test_metrics['rmse_log']:.4f} | MAE ${test_metrics['mae_dollars']:,.0f} | R² {test_metrics['r2']:.3f}")

    candidates = {k: v for k, v in results.items() if not k.startswith("Baseline")}
    best_name = min(candidates, key=lambda n: candidates[n]["cv_rmse_log_mean"])
    best = fitted[best_name]
    print(f"\nSelected: {best_name}")

    # ---- Plots -------------------------------------------------------------
    preds = best.predict(X_test)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(np.expm1(y_test), np.expm1(preds), s=12, alpha=0.6)
    lims = [0, max(np.expm1(y_test).max(), np.expm1(preds).max()) * 1.05]
    axes[0].plot(lims, lims, "r--", lw=1)
    axes[0].set(xlim=lims, ylim=lims, xlabel="Actual price ($)", ylabel="Predicted price ($)",
                title=f"{best_name}: predicted vs actual (test set)")
    residuals = y_test - preds
    axes[1].scatter(preds, residuals, s=12, alpha=0.6, color="#55A868")
    axes[1].axhline(0, color="red", ls="--", lw=1)
    axes[1].set(xlabel="Predicted log(price)", ylabel="Residual (log)", title="Residuals — no obvious pattern is good")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "predicted_vs_actual.png", dpi=120)
    plt.close(fig)

    # Lasso coefficients: which features actually carry weight?
    lasso = fitted["Lasso"]
    names = lasso.named_steps["prep"].named_steps["columns"].get_feature_names_out()
    coefs = pd.Series(lasso.named_steps["model"].coef_, index=[n.split("__", 1)[1] for n in names])
    n_zero = int((coefs == 0).sum())
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    top.plot.barh(ax=ax, color=["#C44E52" if v < 0 else "#4C72B0" for v in top])
    ax.set_title(f"Lasso: top 20 coefficients ({n_zero} of {len(coefs)} features set to exactly zero)")
    ax.set_xlabel("Effect on log(price) per 1 std change")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "lasso_coefficients.png", dpi=120)
    plt.close(fig)

    # Model comparison bar chart
    comp = pd.DataFrame(results).T
    fig, ax = plt.subplots(figsize=(7, 3.8))
    comp["rmse_log"].plot.bar(ax=ax, color=["#999999", "#DD8452", "#4C72B0", "#55A868"])
    ax.set_ylabel("Test RMSE on log(price)  (lower = better)")
    ax.set_title("Model comparison")
    ax.tick_params(axis="x", rotation=0)
    for i, v in enumerate(comp["rmse_log"]):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=120)
    plt.close(fig)

    joblib.dump(best, MODELS_DIR / "house_price_model.joblib")
    summary = {
        "selected_model": best_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features_after_encoding": int(len(coefs)),
        "lasso_zeroed_features": n_zero,
        "lasso_top_features": {k: round(float(v), 4) for k, v in top.iloc[::-1].head(10).items()},
        "models": results,
    }
    (OUTPUTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"Model saved to {MODELS_DIR / 'house_price_model.joblib'}")


if __name__ == "__main__":
    main()
