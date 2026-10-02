import pandas as pd
from scipy.stats import zscore
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("machine_data.csv")
normal_data = df[df["is_anomaly"] == 0]

features = [
    "temperature",
    "vibration",
    "pressure",
    "rpm"
]

X = df[features]
y = df["is_anomaly"]

mean = normal_data[features].mean()
std = normal_data[features].std()

X_z = (X - mean) / std

predicted = (
    X_z.abs().max(axis=1) > 3
).astype(int)

print("Classification Report:")
print(classification_report(y, predicted))

print("Confusion Matrix:")
print(confusion_matrix(y, predicted))

df["predicted_anomaly"] = predicted

detected = df[df["predicted_anomaly"] == 1]

print("\nDetected anomalies:")
print(detected[features + ["is_anomaly", "predicted_anomaly"]])