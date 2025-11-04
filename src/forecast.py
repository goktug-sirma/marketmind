import pandas as pd
from prophet import Prophet

def predict_future_sales(df, days_ahead=30):
    """Forecast future revenue trends for each product using Prophet."""
    df = df.copy()
    df["ds"] = df["date"]
    df["y"] = df["revenue"]

    results = []

    for product in df["product"].unique():
        sub = df[df["product"] == product][["ds", "y"]].sort_values("ds")

        if len(sub) < 5:
            continue  # veri çok azsa atla

        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.5
        )
        model.fit(sub)

        future = model.make_future_dataframe(periods=days_ahead)
        forecast = model.predict(future)
        forecast["product"] = product
        results.append(forecast[["ds", "product", "yhat", "yhat_lower", "yhat_upper"]])

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
