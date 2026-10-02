import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("machine_data.csv")

features = [
    "temperature",
    "vibration",
    "pressure",
    "rpm"
]

X = df[features]
y = df["is_anomaly"]

model = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42
)

predicted = model.fit_predict(X)
predicted = (predicted == -1).astype(int)


print("Classification Report:")
print(classification_report(y, predicted))

print("Confusion Matrix:")
print(confusion_matrix(y, predicted))