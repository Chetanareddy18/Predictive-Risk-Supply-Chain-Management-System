import pandas as pd
import os

# === 1️⃣ Paths ===
DATA_PATH = r"C:\Users\Chetana\OneDrive\Desktop\SCM\DATA"
CLEANED_PATH = os.path.join(DATA_PATH, "cleaned_data")
os.makedirs(CLEANED_PATH, exist_ok=True)

INPUT_FILE = os.path.join(DATA_PATH, "worldometer_data.csv")
OUTPUT_FILE = os.path.join(CLEANED_PATH, "cleaned_worldometer_data.csv")
PARQUET_FILE = os.path.join(CLEANED_PATH, "cleaned_worldometer_data.parquet")

print(f"📂 Loading {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

print(f"✅ Loaded shape: {df.shape}")
print(f"📝 Columns: {list(df.columns)}")

# === 2️⃣ Clean Column Names ===
df.columns = [c.strip().replace(" ", "_").replace("/", "_per_") for c in df.columns]

# === 3️⃣ Convert numeric columns safely ===
numeric_cols = [
    "Population", "TotalCases", "NewCases", "TotalDeaths", "NewDeaths", 
    "TotalRecovered", "NewRecovered", "ActiveCases", "Serious_Critical",
    "Tot_Cases_per_1M_pop", "Deaths_per_1M_pop", "TotalTests", "Tests_per_1M_pop"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# === 4️⃣ Handle missing WHO region (if any) ===
if "WHO_Region" in df.columns:
    df["WHO_Region"].fillna("Unknown", inplace=True)

# === 5️⃣ Save cleaned data ===
df.to_csv(OUTPUT_FILE, index=False)
df.to_parquet(PARQUET_FILE, index=False)

print(f"✅ Cleaned Worldometer data saved to: {OUTPUT_FILE}")
print(f"📦 Parquet version saved to: {PARQUET_FILE}")
print(f"🔢 Final rows: {len(df)}, Columns: {len(df.columns)}")
print(df.head(10))
