import warnings, os, joblib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score
)
from sklearn.ensemble import RandomForestClassifier


try:
    import lightgbm as lgb; HAS_LGB = True
except ImportError:
    HAS_LGB = False
try:
    import xgboost as xgb; HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

PROJECT_DIR = r"C:\Users\Chetana\OneDrive\Desktop\SCM\Project"
DATA_FILE   = os.path.join(PROJECT_DIR, "fact_with_custom_risk_final.csv")
OUT_DIR     = os.path.join(PROJECT_DIR, "models_and_outputs")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

DATE_COL   = "date"
LABEL_COL  = "disruption_likelihood_score"
THRESH_BIN = 0.2            
SHIFT_DAYS = 5
CUTOFF_DATE = "2022-01-01"  
SEED       = 42


df = pd.read_csv(DATA_FILE, parse_dates=[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)

df["future_disruption"] = df[LABEL_COL].shift(-SHIFT_DAYS)
df = df.dropna(subset=["future_disruption"])
df["y_bin"] = (df["future_disruption"] >= THRESH_BIN).astype(int)

print(f"Total rows after 5-day shift: {len(df)}")
print("Positive rate:", df["y_bin"].mean().round(3))

#
leak_cols = {
    DATE_COL, LABEL_COL, "future_disruption", "y_bin",
    "delay_probability", "delivery_time_deviation",
    "risk_score", "coverage_score", "custom_risk"
}
feature_cols = [
    c for c in df.columns
    if c not in leak_cols and pd.api.types.is_numeric_dtype(df[c])
]
print(f"Using {len(feature_cols)} numeric features after leak removal.")

train_df = df[df[DATE_COL] < CUTOFF_DATE]
val_df   = df[df[DATE_COL] >= CUTOFF_DATE]

X_train = train_df[feature_cols].fillna(train_df[feature_cols].median())
X_val   = val_df[feature_cols].fillna(train_df[feature_cols].median())
y_train = train_df["y_bin"].values
y_val   = val_df["y_bin"].values

rng = np.random.default_rng(SEED)
X_train += rng.normal(0, 0.01, X_train.shape)
X_val   += rng.normal(0, 0.01, X_val.shape)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)

print(f"Train size: {len(train_df)}   Validation size: {len(val_df)}")

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=400, n_jobs=-1, random_state=SEED, class_weight="balanced")
}
if HAS_XGB:
    models["XGBoost"] = xgb.XGBClassifier(
        n_estimators=800, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, eval_metric="logloss", random_state=SEED,
        scale_pos_weight=(len(y_train)-y_train.sum())/(y_train.sum()+1e-6)
    )
if HAS_LGB:
    models["LightGBM"] = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, class_weight="balanced", random_state=SEED
    )

results, predictions = [], {}
for name, model in models.items():
    print(f"\n=== Training {name} ===")
    model.fit(X_train_s, y_train)
    prob = model.predict_proba(X_val_s)[:, 1]
    y_pred = (prob >= 0.5).astype(int)

    auc  = roc_auc_score(y_val, prob)
    pr   = average_precision_score(y_val, prob)
    f1   = f1_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec  = recall_score(y_val, y_pred)

    print(f"{name}: AUC={auc:.3f}  PR_AUC={pr:.3f}  "
          f"F1={f1:.3f}  Precision={prec:.3f}  Recall={rec:.3f}")

    results.append([name, auc, pr, f1, prec, rec])
    predictions[name] = prob

metrics = pd.DataFrame(results,
    columns=["Model","AUC","PR_AUC","F1","Precision","Recall"])
metrics.to_csv(os.path.join(OUT_DIR,"model_comparison_metrics.csv"), index=False)
metrics.to_parquet(os.path.join(OUT_DIR,"model_comparison_metrics.parquet"), index=False)

pred_df = val_df[[DATE_COL,"future_disruption","y_bin"]].copy()
for name, prob in predictions.items():
    pred_df[f"pred_prob_{name}"] = prob
pred_df.to_csv(os.path.join(OUT_DIR,"predictions_all_models.csv"), index=False)
pred_df.to_parquet(os.path.join(OUT_DIR,"predictions_all_models.parquet"), index=False)

best_name = metrics.sort_values("AUC", ascending=False).iloc[0]["Model"]
joblib.dump(models[best_name], os.path.join(OUT_DIR, f"best_model_{best_name}.pkl"))
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))

print("\n=== DONE ===")
print("Metrics saved to:", OUT_DIR)
print(metrics)
print("Best model by AUC:", best_name)
