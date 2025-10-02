import pandas as pd
import os

# === 1️⃣ Paths ===
DATA_PATH = r"C:\Users\Chetana\OneDrive\Desktop\SCM\DATA"
CLEANED_PATH = os.path.join(DATA_PATH, "cleaned_data")
os.makedirs(CLEANED_PATH, exist_ok=True)

INPUT_FILE = os.path.join(DATA_PATH, "covid_19_clean_complete.csv")
OUTPUT_FILE = os.path.join(CLEANED_PATH, "cleaned_covid_timeseries.csv")

# === 2️⃣ Load CSV ===
print(f"📂 Loading {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

print(f"✅ Loaded shape: {df.shape}")
print(f"📝 Columns: {list(df.columns)}")

# === 3️⃣ Clean & Process ===
# Fill missing Province/State with "Unknown"
df["Province/State"] = df["Province/State"].fillna("Unknown")

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Group by WHO Region + Date → aggregate totals
covid_region_daily = (
    df.groupby(["WHO Region", "Date"])
      .agg(
          total_confirmed=("Confirmed", "sum"),
          total_deaths=("Deaths", "sum"),
          total_recovered=("Recovered", "sum"),
          active_cases=("Active", "sum")
      )
      .reset_index()
      .sort_values(by=["WHO Region", "Date"])
)

print(f"✅ Aggregated shape: {covid_region_daily.shape}")
print(covid_region_daily.head(10))

# === 4️⃣ Save Cleaned CSV ===
covid_region_daily.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Cleaned covid timeseries saved: {OUTPUT_FILE}")
print(f"🔢 Final rows: {len(covid_region_daily)}, Columns: {len(covid_region_daily.columns)}")
