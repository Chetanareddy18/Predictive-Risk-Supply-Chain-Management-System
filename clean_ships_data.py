import pandas as pd
import os

DATA_PATH = r"C:\Users\Chetana\OneDrive\Desktop\SCM\DATA"
CLEAN_PATH = os.path.join(DATA_PATH, "Cleaned_data")
os.makedirs(CLEAN_PATH, exist_ok=True)

file = os.path.join(DATA_PATH, "Cleaned_ships_data.csv")

# Load
df = pd.read_csv(file)

# Optional: standardize column names
df.columns = [c.strip().lower() for c in df.columns]

# Save to both formats
csv_out = os.path.join(CLEAN_PATH, "ships_data_cleaned.csv")
parquet_out = os.path.join(CLEAN_PATH, "ships_data_cleaned.parquet")

df.to_csv(csv_out, index=False)
df.to_parquet(parquet_out, index=False)

print("✅ Ships data ready for master merge")
print(f"Rows: {len(df)}, Cols: {len(df.columns)}")
