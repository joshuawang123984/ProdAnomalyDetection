import pandas as pd
import torch
import torch.nn as nn
from helper import *
from model import Autoencoder
from sklearn.preprocessing import StandardScaler

def train(model, X_train, epochs=1000):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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

    model = Autoencoder(len(features))
    model = train(model, X_train)

    normal_errors = evaluate(model, X_train)
    threshold = (normal_errors.mean() + 3 * normal_errors.std())

    validation_errors = evaluate(model, X_validation)
    predicted = (validation_errors > threshold).int()

    print("Normal error:", normal_errors.mean().item())
    print("Threshold:", threshold.item())
    print("Validation error:", validation_errors.mean().item())
    print("Anomalies:", predicted.sum().item())
    print("Total:", len(predicted))

    validation_results = validation_df.copy()
    validation_results["reconstruction_error"] = validation_errors.numpy()
    validation_results["predicted_anomaly"] = predicted.numpy()

    print(validation_results[validation_results["predicted_anomaly"] == 1])

if __name__ == '__main__':
    main()