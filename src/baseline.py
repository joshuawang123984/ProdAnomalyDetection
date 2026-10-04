import pandas as pd
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
for index, row in detected.iterrows():
    print(f"\nRow {index}")
    print(f"Actual anomaly: {row['is_anomaly']}")

    for feature in features:
        print(
            f"{feature}: "
            f"value={row[feature]:.2f}, "
            f"z-score={X_z.loc[index, feature]:.2f}"
        )