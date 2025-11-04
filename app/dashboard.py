import sys
from pathlib import Path

# --- Fix import path dynamically ---
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
from data_loader import load_sales_data
from preprocessor import preprocess_data
from visualizer import plot_sales_trends
from model_trainer import train_model
from forecast import predict_future_sales  # yeni eklendi

# --- Streamlit UI Config ---
st.set_page_config(
    page_title="GO Innovations | MarketMind",
    layout="wide",
    page_icon="📊"
)

st.title("📊 GO Innovations | MarketMind Dashboard")
st.caption("Analyze, predict, and visualize your sales data — powered by GO Innovations.")

# === LOAD & PREPROCESS DATA ===
df = load_sales_data("data/sales.csv")
df_clean = preprocess_data(df)

# === SIDEBAR FILTERS ===
st.sidebar.header("🔍 Filters")
if "product" in df_clean.columns:
    products = df["product"].unique().tolist()
    selected_product = st.sidebar.selectbox("Select Product", ["All"] + products)
else:
    st.sidebar.warning("⚠️ 'product' column not found. Using all data.")
    selected_product = "All"

# Date filter
min_date, max_date = df["date"].min(), df["date"].max()
selected_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])

filtered_df = df_clean.copy()
if selected_product != "All":
    filtered_df = filtered_df[filtered_df["product"] == selected_product]
if isinstance(selected_range, list) and len(selected_range) == 2:
    start, end = pd.to_datetime(selected_range[0]), pd.to_datetime(selected_range[1])
    filtered_df = filtered_df[(filtered_df["date"] >= start) & (filtered_df["date"] <= end)]

# === TABS ===
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔮 Forecast", "🧠 Model"])

# --- OVERVIEW TAB ---
with tab1:
    st.subheader("Filtered Sales Data")
    st.dataframe(filtered_df.head(10))

    st.subheader("Revenue Trend")
    fig = plot_sales_trends(filtered_df)
    st.pyplot(fig)
    st.success("Visualization updated successfully ✅")

# --- FORECAST TAB ---
with tab2:
    st.subheader("Sales Forecast by Product")
    days = st.slider("Select forecast horizon (days)", 7, 30, 7)
    if st.button("🔮 Generate Forecast"):
        forecast_df = predict_future_sales(df_clean, days)

        if not forecast_df.empty:
            st.success(f"Forecast generated for next {days} days ✅")
            for product in forecast_df["product"].unique():
                product_df = forecast_df[forecast_df["product"] == product]
                st.write(f"### 📦 {product}")
                st.line_chart(product_df.set_index("date")["predicted_revenue"])
        else:
            st.warning("Not enough data to forecast.")


# --- MODEL TAB ---
with tab3:
    st.subheader("Train Regression Model")
    if st.button("🚀 Train Model"):
        model, mae, r2 = train_model(df_clean)
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**R²:** {r2:.2f}")

        st.info("📈 Example Prediction:")
        sample = filtered_df.sample(1)
        sample_numeric = sample.select_dtypes(include=["number"])
        sample_numeric = sample_numeric.drop(columns=["revenue"], errors="ignore")
        pred = model.predict(sample_numeric)



st.markdown("---")
st.caption("© 2025 GO Innovations | MarketMind — Forecast & Analytics MVP")
