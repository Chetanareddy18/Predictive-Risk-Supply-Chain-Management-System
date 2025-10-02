import pandas as pd
import os

# === 1️⃣ Paths ===
DATA_PATH = r"C:\Users\Chetana\OneDrive\Desktop\SCM\DATA"
CLEANED_PATH = os.path.join(DATA_PATH, "cleaned_data")
os.makedirs(CLEANED_PATH, exist_ok=True)

INPUT_FILE = os.path.join(DATA_PATH, "Effects-of-COVID-19-on-trade-1-February-29-July-2020-provisional.csv")  # Change to your file name
OUTPUT_FILE = os.path.join(CLEANED_PATH, "cleaned_trade_exports.csv")
PARQUET_FILE = os.path.join(CLEANED_PATH, "cleaned_trade_exports.parquet")

print(f"📂 Loading {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

print(f"✅ Loaded shape: {df.shape}")
print(f"📝 Columns: {list(df.columns)}")

# === 2️⃣ Parse Dates ===
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df["Current_Match"] = pd.to_datetime(df["Current_Match"], dayfirst=True, errors="coerce")

# === 3️⃣ Clean Numeric Columns ===
df["Value"] = df["Value"].replace('[\$,]', '', regex=True).astype(float)
df["Cumulative"] = df["Cumulative"].astype(float)

# === 4️⃣ Save Cleaned Data ===
df.to_csv(OUTPUT_FILE, index=False)
df.to_parquet(PARQUET_FILE, index=False)

print(f"✅ Cleaned Trade Exports data saved: {OUTPUT_FILE}")
print(f"📦 Parquet file saved: {PARQUET_FILE}")
print(f"🔢 Final rows: {len(df)}, Columns: {len(df.columns)}")
print(df.head(5))
