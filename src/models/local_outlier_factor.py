import pandas as pd

from sklearn.neighbors import LocalOutlierFactor
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

model = LocalOutlierFactor(
    n_neighbors=20,
    contamination="auto"
)

predicted = model.fit_predict(X)
predicted = (predicted == -1).astype(int)


print("Classification Report:")
print(classification_report(y, predicted))

print("Confusion Matrix:")
print(confusion_matrix(y, predicted))