import joblib
import os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

import src.constants as constants


def main():
    if os.path.exists(constants.BASELINE_PATH):
        model = joblib.load(constants.BASELINE_PATH)
    else: 
        X_train = np.load(constants.TRAIN_WINDOWS_PATH)
        X_train = X_train.reshape(X_train.shape[0], -1)

        model = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=42,
        )

        model.fit(X_train)

        joblib.dump(model, constants.BASELINE_PATH)

    # testing:
    X_val = np.load(constants.VAL_WINDOWS_PATH)
    X_val = X_val.reshape(X_val.shape[0], -1)
    predictions = model.predict(X_val)

    # Isolation Forest: 1 = normal, -1 = anomaly
    anomalies = predictions == -1

    print(f"Test windows: {len(X_val)}")
    print(f"Anomalies detected: {anomalies.sum()}")
    print(f"Anomaly rate: {anomalies.mean():.2%}")
if __name__ == "__main__":
    main()