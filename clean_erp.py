import pandas as pd
import os

# === 1️⃣ Paths ===
DATA_PATH = r"C:\Users\Chetana\OneDrive\Desktop\SCM\DATA"
CLEANED_PATH = os.path.join(DATA_PATH, "cleaned_data")
os.makedirs(CLEANED_PATH, exist_ok=True)

INPUT_FILE = os.path.join(DATA_PATH, "ERP_ERP.csv")
OUTPUT_FILE = os.path.join(CLEANED_PATH, "cleaned_erp_modules.csv")
OUTPUT_PARQUET = os.path.join(CLEANED_PATH, "cleaned_erp_modules.parquet")

print(f"📂 Loading {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

print(f"✅ Loaded shape: {df.shape}")
print(f"📝 Columns: {list(df.columns)}")

# === 2️⃣ Rename & Strip ===
df = df.rename(columns={
    "Component Name": "component_name",
    "Module Code": "module_code",
    "Module Name": "module_name",
    "Table Name": "table_name",
    "Table Category": "table_category",
    "Table Category Notes": "table_category_notes",
    "Dependency Tables Name": "dependency_tables_name",
    "Dependency Tables Count": "dependency_tables_count",
    "Cross-module Dependencies Names": "cross_module_dependencies_names",
    "Cross-module Dependencies Count": "cross_module_dependencies_count",
    "Complexity": "complexity_level"
})

# Strip spaces from string columns
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# === 3️⃣ Fix datatypes ===
df["dependency_tables_count"] = pd.to_numeric(df["dependency_tables_count"], errors="coerce").fillna(0).astype(int)
df["cross_module_dependencies_count"] = pd.to_numeric(df["cross_module_dependencies_count"], errors="coerce").fillna(0).astype(int)

# Fill missing values
df.fillna({
    "dependency_tables_name": "None",
    "cross_module_dependencies_names": "None",
    "complexity_level": "Unknown"
}, inplace=True)

# === 4️⃣ Save Cleaned Versions ===
df.to_csv(OUTPUT_FILE, index=False)
df.to_parquet(OUTPUT_PARQUET, index=False)

print(f"✅ Cleaned ERP Modules data saved as CSV: {OUTPUT_FILE}")
print(f"✅ Cleaned ERP Modules data saved as Parquet: {OUTPUT_PARQUET}")
print(f"🔢 Final rows: {len(df)}, Columns: {len(df.columns)}")
print(df.head(10))
