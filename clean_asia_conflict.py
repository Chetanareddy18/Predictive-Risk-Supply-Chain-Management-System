import pandas as pd
import os

DATA_PATH = r"C:\Users\Chetana\OneDrive\Desktop\SCM\DATA"
CLEAN_PATH = os.path.join(DATA_PATH, "Cleaned_data")
os.makedirs(CLEAN_PATH, exist_ok=True)

conflicts_file = os.path.join(DATA_PATH, "asia_conflicts.csv")  
print(f"Loading {conflicts_file} ...")
df = pd.read_csv(conflicts_file, low_memory=False)


df.rename(columns=lambda x: x.strip().lower(), inplace=True)


if "event_date" in df.columns:
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

if "fatalities" in df.columns:
    df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0).astype(int)
df = df.drop_duplicates()

df = df.dropna(axis=1, how="all")

for col in ["country", "region", "admin1", "location", "event_type", "actor1", "actor2"]:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip().str.title()

csv_out = os.path.join(CLEAN_PATH, "asia_conflicts_cleaned.csv")
parquet_out = os.path.join(CLEAN_PATH, "asia_conflicts_cleaned.parquet")

df.to_csv(csv_out, index=False)        
df.to_parquet(parquet_out, index=False) 

print(f" Cleaned Conflicts data saved as:\n - {csv_out}\n - {parquet_out}")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
