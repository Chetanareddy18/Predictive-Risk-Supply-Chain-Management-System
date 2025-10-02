import pandas as pd
import os

DATA_PATH = r"C:\Users\Chetana\OneDrive\Desktop\SCM\DATA"
CLEANED_PATH = os.path.join(DATA_PATH, "cleaned_data")
os.makedirs(CLEANED_PATH, exist_ok=True)

INPUT_FILE = os.path.join(DATA_PATH, "country_wise_latest.csv")
OUTPUT_FILE = os.path.join(CLEANED_PATH, "cleaned_country_wise_latest.csv")

print(f" Loading {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

print(f" Loaded shape: {df.shape}")
print(f" Columns: {list(df.columns)}")


numeric_cols = df.select_dtypes(include="number").columns
df[numeric_cols] = df[numeric_cols].fillna(0)


region_summary = (
    df.groupby("WHO Region")
      .agg(
          total_confirmed=("Confirmed", "sum"),
          total_deaths=("Deaths", "sum"),
          total_recovered=("Recovered", "sum"),
          active_cases=("Active", "sum"),
          new_cases=("New cases", "sum"),
          new_deaths=("New deaths", "sum"),
          new_recovered=("New recovered", "sum"),
          avg_death_rate=("Deaths / 100 Cases", "mean"),
          avg_recovery_rate=("Recovered / 100 Cases", "mean")
      )
      .reset_index()
)

print("Region-level summary created!")
print(region_summary.head())


region_summary.to_csv(OUTPUT_FILE, index=False)
print(f"Cleaned country-wise dataset saved to: {OUTPUT_FILE}")
print(f" Final rows: {len(region_summary)}, Columns: {len(region_summary.columns)}")
