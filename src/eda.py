from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def run_eda(df: pd.DataFrame, fig_dir: str = "results/figures") -> pd.DataFrame:
    """Save required EDA figures and return missing ratio table."""
    fig_path = Path(fig_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    missing = (df.isna().mean() * 100).sort_values(ascending=False).reset_index()
    missing.columns = ["column", "missing_ratio_percent"]
    missing.to_csv("results/metrics/missing_ratio.csv", index=False)

    plt.figure(figsize=(7, 4))
    sns.histplot(df["Age"], kde=True)
    plt.title("Age Distribution")
    plt.tight_layout()
    plt.savefig(fig_path / "hist_age.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.boxplot(x=df["Fare"])
    plt.title("Fare Boxplot")
    plt.tight_layout()
    plt.savefig(fig_path / "boxplot_fare.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    numeric = df.select_dtypes(include="number")
    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(fig_path / "heatmap_corr.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(x="Survived", data=df)
    plt.title("Target Distribution")
    plt.tight_layout()
    plt.savefig(fig_path / "countplot_survived.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.barplot(x="Sex", y="Survived", data=df)
    plt.title("Survival Rate by Sex")
    plt.tight_layout()
    plt.savefig(fig_path / "barplot_sex_survival.png", dpi=150)
    plt.close()

    return missing
