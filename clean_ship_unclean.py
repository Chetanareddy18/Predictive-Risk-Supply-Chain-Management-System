import pandas as pd
import os

# === 1️⃣ Paths ===
DATA_PATH = r"C:\Users\Chetana\OneDrive\Desktop\SCM\DATA"
CLEANED_PATH = os.path.join(DATA_PATH, "cleaned_data")
os.makedirs(CLEANED_PATH, exist_ok=True)

INPUT_FILE = os.path.join(DATA_PATH, "ship_uncleaned.csv")
OUTPUT_FILE = os.path.join(CLEANED_PATH, "cleaned_ship_data.csv")
PARQUET_FILE = os.path.join(CLEANED_PATH, "cleaned_ship_data.parquet")

print(f"📂 Loading {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

print(f"✅ Loaded shape: {df.shape}")
print(f"📝 Columns: {list(df.columns)}")

# === 2️⃣ Clean & Deduplicate ===
# Strip whitespace from strings
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Drop exact duplicate rows
before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"🧹 Removed {before - after} duplicate rows")

# Ensure numeric columns are truly numeric
for col in ["built_year", "gt", "dwt"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Optionally split "size" into length & breadth if possible (format: "489 / 74")
if "size" in df.columns:
    size_split = df["size"].str.split("/", expand=True)
    if size_split.shape[1] == 2:
        df["length_m"] = pd.to_numeric(size_split[0], errors="coerce")
        df["breadth_m"] = pd.to_numeric(size_split[1], errors="coerce")
    df.drop(columns=["size"], inplace=True)

# Reset index for a clean final dataset
df = df.reset_index(drop=True)

# === 3️⃣ Save Cleaned Data ===
df.to_csv(OUTPUT_FILE, index=False)
df.to_parquet(PARQUET_FILE, index=False)

print(f"✅ Cleaned Ship Data saved: {OUTPUT_FILE}")
print(f"📦 Parquet file saved: {PARQUET_FILE}")
print(f"🔢 Final rows: {len(df)}, Columns: {len(df.columns)}")
print(df.head(10))
