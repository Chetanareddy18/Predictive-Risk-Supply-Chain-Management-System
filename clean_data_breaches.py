import pandas as pd
import os

# === 1️⃣ Paths ===
DATA_PATH = r"C:\Users\Chetana\OneDrive\Desktop\SCM\DATA"
CLEANED_PATH = os.path.join(DATA_PATH, "cleaned_data")
os.makedirs(CLEANED_PATH, exist_ok=True)

INPUT_FILE = os.path.join(DATA_PATH, "data_breaches_global.csv")
OUTPUT_FILE = os.path.join(CLEANED_PATH, "cleaned_data_breaches.csv")
PARQUET_FILE = os.path.join(CLEANED_PATH, "cleaned_data_breaches.parquet")

print(f"📂 Loading {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

print(f"✅ Loaded shape: {df.shape}")
print(f"📝 Columns: {list(df.columns)}")

# === 2️⃣ Rename & Clean ===
df = df.rename(columns={
    "Entity": "organization",
    "Year": "year",
    "Records": "records_compromised",
    "Organization type": "organization_type",
    "Method": "breach_method",
    "Sources": "sources"
})

# Strip whitespace
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# 🔑 Convert records_compromised to numeric
df["records_compromised"] = pd.to_numeric(df["records_compromised"], errors="coerce").fillna(0)

# Fill NAs
df.fillna({
    "organization_type": "Unknown",
    "breach_method": "Unknown"
}, inplace=True)

# === 3️⃣ Add severity score safely ===
def severity_score(row):
    base = 1
    method = row['breach_method'].lower()
    if "hack" in method:
        base += 2
    elif "poor" in method:
        base += 1.5
    elif "lost" in method or "stolen" in method:
        base += 1
    return round(base * (min(row['records_compromised'], 10_000_000) / 1_000_000), 2)

df["incident_severity_score"] = df.apply(severity_score, axis=1)

# === 4️⃣ Save Cleaned Data ===
df.to_csv(OUTPUT_FILE, index=False)
df.to_parquet(PARQUET_FILE, index=False)

print(f"✅ Cleaned Data Breaches data saved: {OUTPUT_FILE}")
print(f"📦 Parquet file saved: {PARQUET_FILE}")
print(f"🔢 Final rows: {len(df)}, Columns: {len(df.columns)}")
print(df.head(10))
