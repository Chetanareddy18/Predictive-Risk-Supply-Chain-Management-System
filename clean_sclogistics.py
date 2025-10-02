import pandas as pd
import os

# === 1️⃣ Paths ===
DATA_PATH = r"C:\Users\Chetana\OneDrive\Desktop\SCM\DATA"
CLEANED_PATH = os.path.join(DATA_PATH, "cleaned_data")
os.makedirs(CLEANED_PATH, exist_ok=True)

INPUT_FILE = os.path.join(DATA_PATH, "dynamic_supply_chain_logistics_dataset.csv")
OUTPUT_FILE = os.path.join(CLEANED_PATH, "cleaned_dynamic_supply_chain_logistics.csv")
PARQUET_FILE = os.path.join(CLEANED_PATH, "cleaned_dynamic_supply_chain_logistics.parquet")

print(f"📂 Loading {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

print(f"✅ Loaded shape: {df.shape}")
print(f"📝 Columns: {list(df.columns)}")

# === 2️⃣ Parse Timestamp ===
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# === 3️⃣ Clean Categorical Columns ===
categorical_cols = [
    "order_fulfillment_status",
    "cargo_condition_status",
    "risk_classification",
    "weather_condition_severity"
]
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower().str.replace(" ", "_")

# === 4️⃣ Convert Numeric Columns Safely ===
numeric_cols = df.columns.difference(["timestamp"] + categorical_cols)
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# === 5️⃣ Handle Missing Values ===
# Fill categorical NAs with "unknown"
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna("unknown")

# Fill numeric NAs with column mean
for col in numeric_cols:
    if df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

# === 6️⃣ Save Cleaned Data ===
df.to_csv(OUTPUT_FILE, index=False)
df.to_parquet(PARQUET_FILE, index=False)

print(f"✅ Cleaned Dynamic Supply Chain data saved: {OUTPUT_FILE}")
print(f"📦 Parquet file saved: {PARQUET_FILE}")
print(f"🔢 Final rows: {len(df)}, Columns: {len(df.columns)}")
print(df.head(5))
