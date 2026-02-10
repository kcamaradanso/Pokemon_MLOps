
import pandas as pd


def extract_data(path: str = "data/pokemon.csv") -> pd.DataFrame:
    """
    Data Collection stage:
    Load the raw pokemon dataset
    """
    df = pd.read_csv(path)
    print(f"Data collected: {len(df)} rows")
    return df
