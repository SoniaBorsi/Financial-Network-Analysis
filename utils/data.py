from pathlib import Path
import pandas as pd
from typing import List, Optional, Union

DEFAULT_PATH = Path("data/returns.csv")

def get_returns(
    csv_path: Path = DEFAULT_PATH,
    years: Optional[List[int]] = None,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None
) -> pd.DataFrame:
    """
    Load return series and filter by year list or date range.
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=[0])
    df.index = pd.to_datetime(df.index, errors="coerce")
    df.sort_index(inplace=True)

    if years is not None:
        df = df[df.index.year.isin(years)]

    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    return df
