import matplotlib.pyplot as plt

def plot_sales_trends(df):
    # Günlük toplam geliri hesapla
    daily_revenue = df.groupby("date")["revenue"].sum()

    # Grafik oluştur
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily_revenue.index, daily_revenue.values, marker="o", linewidth=2, color="royalblue")
    ax.set_title("Daily Revenue Trend", fontsize=14, weight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Revenue", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()

    # Streamlit için fig objesini döndür
    return fig
