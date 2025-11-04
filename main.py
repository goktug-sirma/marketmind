from src.data_loader import load_sales_data
from preprocessor import preprocess_data
from src.model_trainer import train_model
from src.visualizer import plot_sales_trends

def main():
    print("=== GO Innovations | MarketMind ===")
    print("Loading data...")
    df = load_sales_data("data/sales.csv")

    print("Preprocessing...")
    df_clean = preprocess_data(df)

    print("Plotting sales trends...")
    plot_sales_trends(df_clean)

    print("Training model...")
    mae, r2 = train_model(df_clean)
    print(f"Model trained. MAE: {mae:.2f}, R²: {r2:.2f}")
    print("All steps completed successfully ✅")

if __name__ == "__main__":
    main()
