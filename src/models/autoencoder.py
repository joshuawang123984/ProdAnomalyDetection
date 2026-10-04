import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, input_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

def main():
    df = pd.read_csv("machine_data.csv")

    features = [
        "temperature",
        "vibration",
        "pressure",
        "rpm"
    ]

    normal_data = df[df["is_anomaly"] == 0]

    mean = normal_data[features].mean()
    std = normal_data[features].std()

    X_normal = (normal_data[features] - mean) / std

    X_normal = torch.tensor(
        X_normal.values,
        dtype=torch.float32
    )

    model = Autoencoder(len(features))

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 1000
    for epoch in range(epochs):

        reconstructed = model(X_normal)

        loss = criterion(reconstructed, X_normal)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    X = (df[features] - mean) / std
    X = torch.tensor(
        X.values,
        dtype=torch.float32
    )

    model.eval()
    with torch.no_grad():
        reconstructed = model(X)
        normal_reconstructed = model(X_normal)

    errors = torch.mean((X - reconstructed) ** 2, dim=1)
    df["reconstruction_error"] = errors.numpy()

    print(df[["reconstruction_error", "is_anomaly"]]
        .sort_values("reconstruction_error", ascending=False)
        .head(20))

    print(df[df["is_anomaly"] == 1][features + ["is_anomaly", "reconstruction_error"]].sort_values("reconstruction_error"))
    
    normal_errors = torch.mean((X_normal - normal_reconstructed) ** 2, dim=1)
    threshold = normal_errors.mean() + 3 * normal_errors.std()

    predicted = (errors > threshold).int()

    df["predicted_anomaly"] = predicted.numpy()
    y = df["is_anomaly"]

    print("Classification Report:")
    print(classification_report(y, predicted))

    print("Confusion Matrix:")
    print(confusion_matrix(y, predicted))

if __name__ == '__main__':
    main()