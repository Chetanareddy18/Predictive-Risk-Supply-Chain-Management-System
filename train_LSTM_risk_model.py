# train_lstm_risk_model.py
"""
LSTM layer for supply-chain disruption prediction.

- Input: lagged feature CSV (has date and disruption_likelihood_score).
- Output: model .h5, scaler .pkl, predictions csv/parquet, metrics csv.
"""

import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.utils import compute_class_weight

# Keras / TF
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

PROJECT_DIR = r"C:\Users\Chetana\OneDrive\Desktop\SCM\Project"   
DATA_FILE   = os.path.join(PROJECT_DIR, "fact_with_custom_risk_final.csv")  
OUT_DIR     = os.path.join(PROJECT_DIR, "models_and_outputs")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

DATE_COL    = "date"
LABEL_COL   = "disruption_likelihood_score"
SHIFT_DAYS  = 14          
SEQ_LEN     = 14          
BATCH_SIZE  = 64
EPOCHS      = 40
RANDOM_SEED = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("Loading:", DATA_FILE)
df = pd.read_csv(DATA_FILE, parse_dates=[DATE_COL], dayfirst=False, infer_datetime_format=True)
df = df.sort_values(DATE_COL).reset_index(drop=True)
print("Rows:", len(df), "Cols:", df.shape[1])

df["future_disruption"] = df[LABEL_COL].shift(-SHIFT_DAYS)
df = df.dropna(subset=["future_disruption"]).reset_index(drop=True)
df["y_bin"] = (df["future_disruption"] >= 0.5).astype(int)  

print("After shift rows:", len(df), "positive rate:", df["y_bin"].mean())


exclude = {DATE_COL, LABEL_COL, "future_disruption", "y_bin"}
exclude |= {c for c in df.columns if c.startswith("pred_prob_") or c in ["custom_risk", "pred_prob_lr", "risk_score"]}

feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
feature_cols = sorted(feature_cols)  
print("Using", len(feature_cols), "features for LSTM:", feature_cols[:10], "..." if len(feature_cols)>10 else "")

if len(feature_cols) == 0:
    raise SystemExit("No numeric features found. Check your input CSV and exclude list.")

def build_sequences(df, feature_cols, seq_len=SEQ_LEN, group_col=None, date_col=DATE_COL, label_col="y_bin"):
    X_list, y_list, idx_dates = [], [], []
    if group_col and group_col in df.columns:
        for g, gdf in df.groupby(group_col):
            gdf = gdf.sort_values(date_col).reset_index(drop=True)
            arr = gdf[feature_cols].values
            labels = gdf[label_col].values
            dates = gdf[date_col].astype(str).values
            n = len(gdf)
            for i in range(n - seq_len + 1):
                X_list.append(arr[i:i+seq_len])
                y_list.append(labels[i+seq_len-1])  # target aligned to end of window
                idx_dates.append((g, dates[i+seq_len-1]))
    else:
        arr = df[feature_cols].values
        labels = df[label_col].values
        dates = df[date_col].astype(str).values
        n = len(df)
        for i in range(n - seq_len + 1):
            X_list.append(arr[i:i+seq_len])
            y_list.append(labels[i+seq_len-1])
            idx_dates.append(("ALL", dates[i+seq_len-1]))
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y, idx_dates

group_col = "port_name" if "port_name" in df.columns else None
print("Group by:", group_col)
X, y, idx_dates = build_sequences(df, feature_cols, seq_len=SEQ_LEN, group_col=group_col)
print("Sequences built:", X.shape, "Labels:", y.shape)

if X.shape[0] == 0:
    raise SystemExit("No sequences created. Check SEQ_LEN or data length per group.")


dates = np.array([pd.to_datetime(d[1]) for d in idx_dates])
cutoff_time = np.quantile(dates.astype("datetime64[ns]").astype(int), 0.8)
cutoff_time = pd.to_datetime(int(cutoff_time))
print("Cutoff date for train/val:", cutoff_time.date())

train_mask = dates < cutoff_time
val_mask   = ~train_mask

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
dates_train = dates[train_mask]
dates_val   = dates[val_mask]

print("Train sequences:", len(X_train), "Val sequences:", len(X_val))
print("Train pos rate:", y_train.mean(), "Val pos rate:", y_val.mean())

n_features = X_train.shape[2]
scaler = MinMaxScaler()

X_train_2d = X_train.reshape(-1, n_features)
X_val_2d = X_val.reshape(-1, n_features)
scaler.fit(X_train_2d)
X_train_scaled = scaler.transform(X_train_2d).reshape(X_train.shape)
X_val_scaled   = scaler.transform(X_val_2d).reshape(X_val.shape)

joblib.dump(scaler, os.path.join(OUT_DIR, "lstm_scaler.pkl"))
print("Scaler saved.")

if y_train.sum() == 0:
    print("Warning: no positive samples in training set. LSTM will not learn. Aborting.")
    raise SystemExit("No positives in training set.")

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {int(cls): weight for cls, weight in zip(np.unique(y_train), class_weights)}
print("Class weights:", class_weight_dict)


def build_lstm_model(seq_len, n_features, units=64, dropout=0.3):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(seq_len, n_features)))
    model.add(LSTM(units, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc")])
    return model

model = build_lstm_model(SEQ_LEN, n_features, units=128, dropout=0.3)
model.summary()


model_file = os.path.join(OUT_DIR, "lstm_best_model.h5")
chk = ModelCheckpoint(model_file, save_best_only=True, monitor="val_auc", mode="max", verbose=1)
es = EarlyStopping(monitor="val_auc", mode="max", patience=6, restore_best_weights=True, verbose=1)
rlrop = ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=3, min_lr=1e-6, verbose=1)


history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=[chk, es, rlrop],
    verbose=2
)

try:
    model.load_weights(model_file)
    print("Loaded best weights from:", model_file)
except Exception:
    pass

model.save(os.path.join(OUT_DIR, "lstm_final_model.h5"))
print("Saved LSTM model to:", OUT_DIR)
y_prob_val = model.predict(X_val_scaled, batch_size=BATCH_SIZE).ravel()
y_pred_val = (y_prob_val >= 0.5).astype(int)

auc_val = roc_auc_score(y_val, y_prob_val)
pr_auc_val = average_precision_score(y_val, y_prob_val)
f1_val = f1_score(y_val, y_pred_val)
prec_val = precision_score(y_val, y_pred_val, zero_division=0)
rec_val = recall_score(y_val, y_pred_val, zero_division=0)

metrics = {
    "AUC": auc_val,
    "PR_AUC": pr_auc_val,
    "F1": f1_val,
    "Precision": prec_val,
    "Recall": rec_val,
    "Train_pos_rate": float(y_train.mean()),
    "Val_pos_rate": float(y_val.mean()),
    "Train_size": int(len(X_train)),
    "Val_size": int(len(X_val))
}
print("Validation metrics:", metrics)


pred_rows = []
val_indices = np.where(val_mask)[0] 
pred_df = pd.DataFrame({
    "date": dates_val,
    "y_true": y_val,
    "y_prob": y_prob_val,
    "y_pred": y_pred_val
})
pred_df = pred_df.sort_values("date").reset_index(drop=True)
pred_df.to_csv(os.path.join(OUT_DIR, "lstm_val_predictions.csv"), index=False)
try:
    pred_df.to_parquet(os.path.join(OUT_DIR, "lstm_val_predictions.parquet"), index=False)
except Exception:
    pass
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(os.path.join(OUT_DIR, "lstm_metrics.csv"), index=False)
metrics_df.to_parquet(os.path.join(OUT_DIR, "lstm_metrics.parquet"), index=False)

joblib.dump(model, os.path.join(OUT_DIR, "lstm_keras_model_joblib.pkl"))  
print("Saved predictions and metrics to:", OUT_DIR)

print("Done.")
