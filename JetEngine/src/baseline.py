import pandas as pd
import torch
import os
import joblib
import torch.nn as nn
from helper import *
from model import Autoencoder
from sklearn.preprocessing import StandardScaler

def train(model, X_train, epochs=2000):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    for epoch in range(epochs):
        reconstructed = model(X_train)
        loss = criterion(reconstructed, X_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], "f"Loss: {loss.item():.4f}")

    return model

def evaluate(model, X):
    model.eval()
    with torch.no_grad():
        reconstructed = model(X)

    errors = torch.mean((X - reconstructed) ** 2, dim=1)
    return errors

def main():
    df = add_column_header("CMAPSSDATA/train_FD001.txt")

    train_df = df.iloc[:int(0.7 * len(df))]
    validation_df = df.iloc[int(0.7 * len(df)):]

    scaler = StandardScaler()
    features = [col for col in df.columns if col not in ["unit_id", "cycle"]]

    X_train = scaler.fit_transform(train_df[features])
    X_validation = scaler.transform(validation_df[features])

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_validation = torch.tensor(X_validation, dtype=torch.float32)

    model = Autoencoder(len(features), latent_size=4)
    model = train(model, X_train)

    normal_errors = evaluate(model, X_train)
    threshold = (normal_errors.mean() + 3 * normal_errors.std())

    validation_errors = evaluate(model, X_validation)
    predicted = (validation_errors > threshold).int()

    torch.save(model.state_dict(), "models/autoencoder/autoencoder.pth")
    joblib.dump(scaler, "models/autoencoder/scaler.joblib")
    joblib.dump(threshold.item(), "models/autoencoder/threshold.joblib")

    result = {
        "experiment": "baseline",
        "feature_count": len(features),
        "normal_error": normal_errors.mean().item(),
        "threshold": threshold.item(),
        "validation_error": validation_errors.mean().item(),
        "anomalies": predicted.sum().item(),
        "total": len(predicted),
        "anomaly_rate": predicted.float().mean().item()
    }

    pd.DataFrame([result]).to_csv(
        "features_subset_results.csv",
        mode="a",
        header=not os.path.exists("features_subset_results.csv"),
        index=False
    )

if __name__ == '__main__':
    main()