import pandas as pd
import os

# === 1️⃣ Paths ===
DATA_PATH = r"C:\Users\Chetana\OneDrive\Desktop\SCM\DATA"
CLEANED_PATH = os.path.join(DATA_PATH, "cleaned_data")
os.makedirs(CLEANED_PATH, exist_ok=True)

INPUT_FILE = os.path.join(DATA_PATH, "usa_county_wise.csv")
OUTPUT_FILE = os.path.join(CLEANED_PATH, "cleaned_usa_country_wise.csv")
PARQUET_FILE = os.path.join(CLEANED_PATH, "cleaned_usa_country_wise.parquet")

print(f"📂 Loading {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

print(f"✅ Loaded shape: {df.shape}")
print(f"📝 Columns: {list(df.columns)}")

# === 2️⃣ Clean & Normalize ===

# Strip whitespace from string columns
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Convert date column to datetime
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Convert numeric columns safely
numeric_cols = ["UID", "code3", "FIPS", "Lat", "Long_", "Confirmed", "Deaths"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill NaN values where appropriate
df["Province_State"] = df["Province_State"].fillna("Unknown")
df["Admin2"] = df["Admin2"].fillna("Unknown")

# Drop duplicates
before = len(df)
df = df.drop_duplicates()
print(f"🧹 Removed {before - len(df)} duplicate rows")

# Reset index
df.reset_index(drop=True, inplace=True)

# === 3️⃣ Save Cleaned Data ===
df.to_csv(OUTPUT_FILE, index=False)
df.to_parquet(PARQUET_FILE, index=False)

print(f"✅ Cleaned USA Country-Wise data saved: {OUTPUT_FILE}")
print(f"📦 Parquet file saved: {PARQUET_FILE}")
print(f"🔢 Final rows: {len(df)}, Columns: {len(df.columns)}")
print(df.head(10))
