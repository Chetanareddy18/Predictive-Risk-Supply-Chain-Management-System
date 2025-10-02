import pandas as pd
import os

# === 1️⃣ Paths ===
DATA_PATH = r"C:\Users\Chetana\OneDrive\Desktop\SCM\DATA"
CLEANED_PATH = os.path.join(DATA_PATH, "cleaned_data")
os.makedirs(CLEANED_PATH, exist_ok=True)

FILES = [
    os.path.join(DATA_PATH, "StormEvents_Locations_2023.csv"),
    os.path.join(DATA_PATH, "StormEvents_Locations_2024.csv"),
]

OUTPUT_FILE = os.path.join(CLEANED_PATH, "cleaned_stormevent_locations.csv")
OUTPUT_PARQUET = os.path.join(CLEANED_PATH, "cleaned_stormevent_locations.parquet")

# === 2️⃣ Load & Merge ===
dfs = []
for f in FILES:
    print(f"📂 Loading {f}")
    df = pd.read_csv(f)
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)
print(f"✅ Merged shape: {merged.shape}")

# === 3️⃣ Clean Columns ===
merged = merged.rename(columns={
    "YEARMONTH": "yearmonth",
    "EPISODE_ID": "episode_id",
    "EVENT_ID": "event_id",
    "LOCATION_INDEX": "location_index",
    "RANGE": "range_miles",
    "AZIMUTH": "azimuth",
    "LOCATION": "location",
    "LATITUDE": "latitude",
    "LONGITUDE": "longitude",
    "LAT2": "lat2",
    "LON2": "lon2"
})

# Ensure numeric types
numeric_cols = ["yearmonth", "episode_id", "event_id", "location_index",
                "range_miles", "latitude", "longitude", "lat2", "lon2"]

for col in numeric_cols:
    merged[col] = pd.to_numeric(merged[col], errors="coerce")

# Strip whitespace
merged["location"] = merged["location"].astype(str).str.strip()
merged["azimuth"] = merged["azimuth"].astype(str).str.strip()

# === 4️⃣ Save Cleaned Versions ===
merged.to_csv(OUTPUT_FILE, index=False)
merged.to_parquet(OUTPUT_PARQUET, index=False)

print(f"✅ Cleaned StormEvent Locations saved as CSV: {OUTPUT_FILE}")
print(f"✅ Cleaned StormEvent Locations saved as Parquet: {OUTPUT_PARQUET}")
print(f"🔢 Final rows: {len(merged)}, Columns: {len(merged.columns)}")
print(merged.head(10))
