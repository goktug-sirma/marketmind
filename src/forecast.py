import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def predict_future_sales(df, days_ahead=7):
    """Forecast future revenue for each product using multi-feature regression."""
    df = df.copy()
    df["timestamp"] = (df["date"] - df["date"].min()).dt.days

    features = [c for c in df.columns if c not in ["date", "revenue"]]
    future_results = []

    for product in df["product"].unique():
        sub = df[df["product"] == product]
        X = sub[features].select_dtypes(include=["number"])
        y = sub["revenue"]

        if len(sub) < 3:
            continue

        model = LinearRegression()
        model.fit(X, y)

        # gelecekteki n gün için feature’lar
        try:
            future_dates = pd.date_range(df["date"].max(), periods=days_ahead + 1, closed="right")
        except TypeError:
            future_dates = pd.date_range(df["date"].max(), periods=days_ahead + 1, inclusive="right")

        base = sub.iloc[-1:].copy()
        preds = []
        for date in future_dates:
            base["timestamp"] += 1
            pred = model.predict(base[features].select_dtypes(include=["number"]))
            preds.append({"date": date, "product": product, "predicted_revenue": pred[0]})

        future_results.extend(preds)

    return pd.DataFrame(future_results)
