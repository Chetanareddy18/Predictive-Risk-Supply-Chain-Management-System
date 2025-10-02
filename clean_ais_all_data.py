import pandas as pd
import os

DATA_PATH = r"C:\Users\Chetana\OneDrive\Desktop\SCM\DATA"
CLEAN_PATH = os.path.join(DATA_PATH, "Cleaned_data")
os.makedirs(CLEAN_PATH, exist_ok=True)

ais_file = os.path.join(DATA_PATH, "ais_data_all.csv") 
print(f"Loading {ais_file} ...")
df = pd.read_csv(ais_file, low_memory=False)
df.rename(columns={"BaseDateTime": "timestamp"}, inplace=True)

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

df = df.drop_duplicates()

df = df.dropna(axis=1, how="all")

csv_out = os.path.join(CLEAN_PATH, "ais_data_cleaned.csv")
parquet_out = os.path.join(CLEAN_PATH, "ais_data_cleaned.parquet")

df.to_csv(csv_out, index=False)         
df.to_parquet(parquet_out, index=False) 

print(f"Cleaned AIS data saved as:\n - {csv_out}\n - {parquet_out}")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
