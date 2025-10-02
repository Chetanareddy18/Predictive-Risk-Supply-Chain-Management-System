import pandas as pd
import os

# === 1️⃣ Paths ===
DATA_PATH = r"C:\Users\Chetana\OneDrive\Desktop\SCM\DATA"
CLEANED_PATH = os.path.join(DATA_PATH, "cleaned_data")
os.makedirs(CLEANED_PATH, exist_ok=True)

INPUT_FILE = os.path.join(DATA_PATH, "Port_locations.csv")  # Change if needed
OUTPUT_FILE = os.path.join(CLEANED_PATH, "cleaned_port_locations.csv")
PARQUET_FILE = os.path.join(CLEANED_PATH, "cleaned_port_locations.parquet")

print(f"📂 Loading {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

print(f"✅ Loaded shape: {df.shape}")
print(f"📝 Columns: {list(df.columns)}")

# === 2️⃣ Clean & Deduplicate ===
# Remove leading/trailing whitespace
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Convert country and port_name to proper case (first letter capital)
df["country"] = df["country"].str.title()
df["port_name"] = df["port_name"].str.title()

# Drop exact duplicates
before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"🧹 Removed {before - after} duplicate rows")

# === 3️⃣ Save Cleaned Data ===
df.to_csv(OUTPUT_FILE, index=False)
df.to_parquet(PARQUET_FILE, index=False)

print(f"✅ Cleaned Port Locations data saved: {OUTPUT_FILE}")
print(f"📦 Parquet file saved: {PARQUET_FILE}")
print(f"🔢 Final rows: {len(df)}, Columns: {len(df.columns)}")
print(df.head(10))
