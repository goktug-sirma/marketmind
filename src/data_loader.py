import pandas as pd
from pathlib import Path

def load_sales_data(file_path: str) -> pd.DataFrame:
    """Load sales data from CSV file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{file_path} not found.")
    df = pd.read_csv(file_path, parse_dates=["date"], encoding="latin1", engine="python")

    return df
