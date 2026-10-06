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

    print(df.head())
    print(df.shape)
    print(df.describe().T)
    print(df.isna().sum())
    print(df.nunique())

    features_to_remove = [
        "setting_3",
        "sensor_1",
        "sensor_5",
        "sensor_6",
        "sensor_10",
        "sensor_16",
        "sensor_18",
        "sensor_19"
    ]

    df = remove_features(df, features_to_remove)
    print(df.head())
    print(df.shape)

    train_df = df.iloc[:int(0.7 * len(df))]
    validation_df = df.iloc[int(0.7 * len(df)):int(0.85 * len(df))]
    test_df = df.iloc[int(0.85 * len(df)):]

    scaler = StandardScaler()
    features = df.columns

    X_train = scaler.fit_transform(train_df[features])
    X_validation = scaler.transform(validation_df[features])
    X_test = scaler.transform(test_df[features])

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_validation = torch.tensor(X_validation, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    model = Autoencoder(len(features))
    model = train(model, X_train)

    train_errors = evaluate(model, X_train)
    validation_errors = evaluate(model, X_validation)
    test_errors = evaluate(model, X_test)

    if True:
        test_df = test_df.copy()
        test_df["reconstruction_error"] = test_errors.numpy()

        engine = test_df[test_df["unit_id"] == 86]

        plt.plot(engine["cycle"], engine["reconstruction_error"])

        plt.xlabel("Cycle")
        plt.ylabel("Reconstruction Error")
        plt.title("Engine 86 Reconstruction Error")
        plt.show()

    print("Train:")
    print(train_errors.mean().item())

    print("Validation:")
    print(validation_errors.mean().item())

    print("Test:")
    print(test_errors.mean().item())

if __name__ == '__main__':
    main()