from pathlib import Path
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


def load_titanic(data_path: str = "data/titanic.csv") -> pd.DataFrame:
    """Load Titanic data. If local file does not exist, download it from a public GitHub mirror."""
    path = Path(data_path)
    if path.exists():
        return pd.read_csv(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_URL)
    df.to_csv(path, index=False)
    return df
