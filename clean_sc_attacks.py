import pandas as pd
import os

# === 1️⃣ Paths ===
DATA_PATH = r"C:\Users\Chetana\OneDrive\Desktop\SCM\DATA"
CLEANED_PATH = os.path.join(DATA_PATH, "cleaned_data")
os.makedirs(CLEANED_PATH, exist_ok=True)

INPUT_FILE = os.path.join(DATA_PATH, "supply_chain_cloud_attacks_dataset.csv")
OUTPUT_FILE = os.path.join(CLEANED_PATH, "cleaned_supply_chain_attacks.csv")
PARQUET_FILE = os.path.join(CLEANED_PATH, "cleaned_supply_chain_attacks.parquet")

print(f"📂 Loading {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

print(f"✅ Loaded shape: {df.shape}")
print(f"📝 Columns: {list(df.columns)}")

# === 2️⃣ Clean & Normalize ===

# Convert event_time to proper datetime
df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')

# Strip whitespace from string columns
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Convert boolean-like columns safely (True/False, 0/1)
bool_cols = [
    "checksum_valid", "signed_commit", "dependency_added", "dependency_version_change",
    "unusual_dependency_source", "new_service_deployed", "detected_malware_signature",
    "obfuscated_code", "unknown_binary_hash", "typosquatting_match",
    "region_sensitive_data_accessed", "is_supply_chain_attack", "response_required"
]
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})

# Convert numeric columns safely
numeric_cols = [
    "static_analysis_score", "network_connections", "file_modifications",
    "unexpected_syscalls", "volume_mount_attempts", "time_since_last_activity",
    "burst_activity_window", "anomaly_score", "rate_of_change_in_activity",
    "deviation_from_baseline"
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop duplicates if any
before = len(df)
df = df.drop_duplicates()
print(f"🧹 Removed {before - len(df)} duplicate rows")

# Reset index
df.reset_index(drop=True, inplace=True)

# === 3️⃣ Save Cleaned Data ===
df.to_csv(OUTPUT_FILE, index=False)
df.to_parquet(PARQUET_FILE, index=False)

print(f"✅ Cleaned Supply Chain Attacks data saved: {OUTPUT_FILE}")
print(f"📦 Parquet file saved: {PARQUET_FILE}")
print(f"🔢 Final rows: {len(df)}, Columns: {len(df.columns)}")
print(df.head(10))
