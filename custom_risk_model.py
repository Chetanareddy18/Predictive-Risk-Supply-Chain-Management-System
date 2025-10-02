
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("Feature_engineered_master.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

key_cols = [
    "storm_events_count","cyber_attack_count","total_confirmed",
    "port_congestion_level","traffic_congestion_level",
    "ship_count","avg_ship_speed","trade_value"
]
for c in key_cols:
    df[f"{c}_r7"] = df[c].rolling(7, min_periods=1).mean()

features = [
    "storm_events_count_r7","weather_condition_severity",
    "cyber_attack_count_r7","total_confirmed_r7","total_deaths","active_cases",
    "port_congestion_level_r7","traffic_congestion_level_r7",
    "warehouse_inventory_level","loading_unloading_time",
    "ship_count_r7","avg_ship_speed_r7","route_risk_level",
    "fuel_consumption_rate","driver_behavior_score","fatigue_monitoring_score",
    "trade_value_r7","historical_demand","shipping_costs",
    "avg_neighbor_risk"
]
for f in features:
    if f not in df.columns:  
        df[f] = 0.0

scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features].fillna(0))

df["f_storm_weather"] = df[["storm_events_count_r7","weather_condition_severity"]].mean(axis=1)
df["f_cyber"]         = df["cyber_attack_count_r7"]
df["f_health"]        = df[["total_confirmed_r7","total_deaths","active_cases"]].mean(axis=1)
df["f_congestion_ops"]= df[["port_congestion_level_r7","traffic_congestion_level_r7",
                            "warehouse_inventory_level","loading_unloading_time"]].mean(axis=1)
df["f_shipping_route"]= df[["ship_count_r7","avg_ship_speed_r7","route_risk_level",
                            "fuel_consumption_rate","driver_behavior_score",
                            "fatigue_monitoring_score"]].mean(axis=1)
df["f_trade"]         = df[["trade_value_r7","historical_demand","shipping_costs"]].mean(axis=1)
df["f_neighbor"]      = df["avg_neighbor_risk"]   

weights = dict(
    storm_weather = 0.20,
    cyber         = 0.15,
    health        = 0.10,
    congestion_ops= 0.15,
    shipping_route= 0.20,
    trade         = 0.15,
    neighbor      = 0.05
)

df["custom_risk"] = (
      df["f_storm_weather"]  * weights["storm_weather"]
    + df["f_cyber"]          * weights["cyber"]
    + df["f_health"]         * weights["health"]
    + df["f_congestion_ops"] * weights["congestion_ops"]
    + df["f_shipping_route"] * weights["shipping_route"]
    + df["f_trade"]          * weights["trade"]
    + df["f_neighbor"]       * weights["neighbor"]
).clip(0, 1)

if "disruption_likelihood_score" in df.columns:
    y = (df["disruption_likelihood_score"] > 0.5).astype(int)
    X = df[[
        "f_storm_weather","f_cyber","f_health",
        "f_congestion_ops","f_shipping_route",
        "f_trade","f_neighbor"
    ]]
    lr = LogisticRegression(max_iter=1000).fit(X, y)
    learned = abs(lr.coef_[0])
    learned /= learned.sum()
    print("Learned weights:",
          dict(zip(X.columns, learned)))
    df["pred_prob_lr"] = lr.predict_proba(X)[:,1]

df.to_csv("fact_with_custom_risk_final.csv", index=False)
print(" Final custom risk file saved: fact_with_custom_risk_final.csv")

df.to_parquet("fact_with_custom_risk_final.parquet", index=False)
print(" Also saved to fact_with_custom_risk_final.parquet")
