import pandas as pd
from pathlib import Path
import numpy as np

Path("data").mkdir(exist_ok=True)

dates = pd.date_range("2025-01-01", periods=30)
data = {
    "date": dates,
    "product": np.random.choice(["Phone", "Laptop", "Headphones"], 30),
    "units_sold": np.random.randint(10, 100, 30),
    "unit_price": np.random.choice([500, 1200, 150], 30),
    "marketing_spend": np.random.randint(50, 300, 30),
}
df = pd.DataFrame(data)
df.to_csv("data/sales.csv", index=False, encoding="utf-8")
print("✅ sales.csv yeniden oluşturuldu.")
