import pandas as pd
import torch
import joblib
from helper import *
from model import Autoencoder

def main():
    df = add_column_header("CMAPSSDATA/test_FD001.txt")

    features = [col for col in df.columns if col not in ["unit_id", "cycle"]]

    model = Autoencoder(len(features), latent_size=4)
    model.load_state_dict(
        torch.load("models/autoencoder/autoencoder.pth")
    )

    scaler = joblib.load("models/autoencoder/scaler.joblib")
    threshold = joblib.load("models/autoencoder/threshold.joblib")

    X_test = scaler.transform(df[features])
    X_test = torch.tensor(X_test, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        reconstructed = model(X_test)

    errors = torch.mean((X_test - reconstructed) ** 2, dim=1)
    predicted = (errors > threshold).int()

    print("Threshold:", threshold)
    print("Anomalies:", predicted.sum().item())
    print("Total:", len(predicted))
    print("Anomaly rate:", predicted.float().mean().item())

if __name__ == '__main__':
    main()