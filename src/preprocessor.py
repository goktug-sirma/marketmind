import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create features and prepare data for training."""
    df = df.copy()
    df["revenue"] = df["units_sold"] * df["unit_price"]
    df["day_of_week"] = df["date"].dt.day_name()

    # Keep original columns for visualization
    dummies = pd.get_dummies(df[["product", "day_of_week"]], drop_first=True)
    df = pd.concat([df, dummies], axis=1)

    return df
