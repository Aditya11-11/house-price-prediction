"""Exploratory plots for Ames Housing.

Run:  python -m src.eda
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data import FIGURES_DIR, TARGET, load_data


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    df = load_data()
    print("Shape:", df.shape)
    print(df[TARGET].describe().round(0))

    # 1. Target distribution — raw vs log
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df[TARGET], bins=50, ax=axes[0], color="#4C72B0")
    axes[0].set_title(f"SalePrice (skew = {df[TARGET].skew():.2f})")
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v / 1000:.0f}k"))
    sns.histplot(np.log1p(df[TARGET]), bins=50, ax=axes[1], color="#55A868")
    axes[1].set_title(f"log(SalePrice) (skew = {np.log1p(df[TARGET]).skew():.2f})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "target_distribution.png", dpi=120)
    plt.close(fig)

    # 2. Missing values — most of these turn out to mean "doesn't have one"
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0] * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    missing.plot.barh(ax=ax, color="#C44E52")
    ax.invert_yaxis()
    ax.set_xlabel("% missing")
    ax.set_title("Missing values by column")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "missing_values.png", dpi=120)
    plt.close(fig)

    # 3. Top correlations with price
    corr = df.select_dtypes("number").corr()[TARGET].drop(TARGET).sort_values(ascending=False)
    print("\nTop correlations with SalePrice:\n", corr.head(10).round(3))
    fig, ax = plt.subplots(figsize=(7, 5))
    corr.head(12).iloc[::-1].plot.barh(ax=ax, color="#4C72B0")
    ax.set_title("Numeric features most correlated with price")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "top_correlations.png", dpi=120)
    plt.close(fig)

    # 4. Living area vs price — this is where the outliers jump out
    fig, ax = plt.subplots(figsize=(7, 4.5))
    outlier = (df["GrLivArea"] > 4000) & (df[TARGET] < 300_000)
    ax.scatter(df.loc[~outlier, "GrLivArea"], df.loc[~outlier, TARGET], s=10, alpha=0.5, label="houses")
    ax.scatter(df.loc[outlier, "GrLivArea"], df.loc[outlier, TARGET], s=60, color="red", label="outliers (removed)")
    ax.set_xlabel("Above-ground living area (sq ft)")
    ax.set_ylabel("Sale price ($)")
    ax.legend()
    ax.set_title("Bigger house, higher price... mostly")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "area_vs_price.png", dpi=120)
    plt.close(fig)

    # 5. Neighbourhood effect
    order = df.groupby("Neighborhood")[TARGET].median().sort_values().index
    fig, ax = plt.subplots(figsize=(12, 4.5))
    sns.boxplot(data=df, x="Neighborhood", y=TARGET, order=order, ax=ax, color="#8172B2", fliersize=2)
    ax.tick_params(axis="x", rotation=60)
    ax.set_title("Location, location, location")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "neighborhood_prices.png", dpi=120)
    plt.close(fig)
    print(f"\nFigures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
