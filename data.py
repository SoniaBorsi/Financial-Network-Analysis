# data.py
from pathlib import Path
import pandas as pd

DEFAULT_PATH = Path("data/returns.csv")

def get_returns(csv_path: Path = DEFAULT_PATH) -> pd.DataFrame:
    """
    Read Date-indexed CSV and return a sorted DataFrame.
    Assumes:
      • first column is a parseable Date, becomes the index
      • remaining columns are pre-cleaned return series
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=[0])
    df.index = pd.to_datetime(df.index, errors="coerce")
    df.sort_index(inplace=True)
    return df
