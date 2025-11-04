import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def train_model(df: pd.DataFrame):
    """Train a simple linear regression model."""
    df = df.copy()

    # --- Only keep numeric columns ---
    df_numeric = df.select_dtypes(include=["number"])

    X = df_numeric.drop(columns=["revenue"])
    y = df_numeric["revenue"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mae, r2
