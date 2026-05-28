import numpy as np
import pandas as pd


def add_titanic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create required derived features for the Titanic classification task."""
    out = df.copy()
    out["FamilySize"] = out["SibSp"].fillna(0) + out["Parch"].fillna(0) + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["FarePerPerson"] = out["Fare"] / out["FamilySize"].replace(0, np.nan)
    out["AgeGroup"] = pd.cut(
        out["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teen", "Young", "Adult", "Senior"],
        include_lowest=True,
    )
    out["Title"] = out["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).fillna("Unknown")
    rare_titles = out["Title"].value_counts()[out["Title"].value_counts() < 10].index
    out["Title"] = out["Title"].replace(rare_titles, "Rare")
    return out
