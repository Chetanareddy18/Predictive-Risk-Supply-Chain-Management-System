import pandas as pd
import os

# === 1️⃣ Paths ===
DATA_PATH = r"C:\Users\Chetana\OneDrive\Desktop\SCM\DATA"
CLEANED_PATH = os.path.join(DATA_PATH, "cleaned_data")
os.makedirs(CLEANED_PATH, exist_ok=True)

INPUT_FILE = os.path.join(DATA_PATH, "day_wise.csv")
OUTPUT_FILE = os.path.join(CLEANED_PATH, "cleaned_daywise_global.csv")

# === 2️⃣ Load CSV ===
print(f"📂 Loading {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

print(f"✅ Loaded shape: {df.shape}")
print(f"📝 Columns: {list(df.columns)}")

# === 3️⃣ Clean & Process ===
# Convert Date to datetime format
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Rename columns to match naming convention (snake_case)
df = df.rename(columns={
    "Date": "date",
    "Confirmed": "global_confirmed",
    "Deaths": "global_deaths",
    "Recovered": "global_recovered",
    "Active": "global_active",
    "New cases": "global_new_cases",
    "New deaths": "global_new_deaths",
    "New recovered": "global_new_recovered",
    "Deaths / 100 Cases": "death_rate_per_100",
    "Recovered / 100 Cases": "recovery_rate_per_100",
    "Deaths / 100 Recovered": "deaths_per_100_recovered",
    "No. of countries": "reporting_countries"
})

# Optional: Fill NA values (just in case)
df.fillna(0, inplace=True)

# Sort by date
df.sort_values(by="date", inplace=True)

print(f"✅ Cleaned shape: {df.shape}")
print(df.head(10))

# === 4️⃣ Save Cleaned CSV ===
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Cleaned global day-wise data saved: {OUTPUT_FILE}")
print(f"🔢 Final rows: {len(df)}, Columns: {len(df.columns)}")
