# 📊 GO Innovations | MarketMind

**MarketMind** is a data-driven sales analytics and forecasting platform built with **FastAPI**, **Streamlit**, and **Machine Learning**.  
It provides real-time insights, visualizations, and AI-powered forecasts for product sales trends.

---

## 🚀 Features

- 🧾 **Data Loading & Cleaning** – Load and preprocess sales data from CSV files  
- 📈 **Visualization** – Interactive revenue and trend charts  
- 🧠 **Machine Learning** – Train regression models to predict future revenue  
- 🔮 **Forecasting** – Predict long-term trends with Prophet  
- 🧮 **Modular Architecture** – Clean structure with separate modules for data, models, and visualization  

---

## 🗂️ Project Structure
marketmind/
│
├── app/
│ └── dashboard.py # Streamlit dashboard
│
├── src/
│ ├── data_loader.py # Load sales data
│ ├── preprocessor.py # Clean & prepare features
│ ├── visualizer.py # Matplotlib/Plotly visualizations
│ ├── model_trainer.py # Train linear regression model
│ └── forecast.py # Prophet-based forecasting
│
├── data/
│ ├── sales.csv # Example dataset
│ └── generate_sales.py # Synthetic data generator
│
├── requirements.txt
├── README.md
└── main.py


---

## 🧠 Local Setup

```bash
# 1️⃣ Create environment
python -m venv venv
venv\Scripts\activate  # (Windows)

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run dashboard
streamlit run app/dashboard.py
```
---

## 🔮 Prophet Forecast Upgrade

MarketMind now includes Facebook Prophet integration for advanced time series forecasting.

✨ What’s New
- Detects trend and seasonality automatically
- Generates 30–60 day forecasts for each product
- Includes upper/lower confidence intervals (yhat_lower, yhat_upper)
- Displays interactive trend and seasonality components using Plotly

---

## 📊 Usage
```bash
streamlit run app/dashboard.py
```

Then navigate to the Forecast tab:
1. Choose a forecast horizon (7–60 days)
2. Click “Generate Prophet Forecast”
3. View product-level forecast charts and Prophet components

---

## 🧩 Technical Details

- Model: Prophet (Stan backend)
- Visualization: Plotly + Streamlit
- Files: src/forecast.py, app/dashboard.py

---

## 📘 License

MIT License © 2025 GO Innovations
Developed by Göktuğ Sırma