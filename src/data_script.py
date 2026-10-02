import pandas as pd
import numpy as np

N = 500

np.random.seed(42)

timestamps = pd.date_range(
    start="2026-01-01",
    periods=N,
    freq="min"
)

temperature = np.random.normal(
    loc=70,
    scale=5,
    size=N
)

vibration = np.random.normal(
    loc=0.30,
    scale=0.05,
    size=N
)

pressure = np.random.normal(
    loc=100,
    scale=3,
    size=N
)

rpm = np.random.normal(
    loc=1500,
    scale=50,
    size=N
)


df = pd.DataFrame({
    "timestamp": timestamps,
    "temperature": temperature,
    "vibration": vibration,
    "pressure": pressure,
    "rpm": rpm
})


df["is_anomaly"] = 0

point_indices = np.random.choice(
    N,
    size=10,
    replace=False
)

df.loc[point_indices, "temperature"] += np.random.uniform(
    30, 50, size=len(point_indices)
)

df.loc[point_indices, "vibration"] += np.random.uniform(
    0.3, 0.6, size=len(point_indices)
)

df.loc[point_indices, "is_anomaly"] = 1



df.to_csv("machine_data.csv", index=False)

print(df.head())
print("\nAnomalies:", df["is_anomaly"].sum())

anomalies_df = df[df["is_anomaly"] == 1]
print(anomalies_df)