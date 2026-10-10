import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import os

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_size=64):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)

        seq_len = x.size(1)
        context = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)

        decoded, _ = self.decoder(context)

        return self.output_layer(decoded)

def load_and_train(train_windows: str, val_windows: str, epochs: int) -> tuple[LSTMAutoencoder, torch.tensor, torch.tensor]:

    X_train = np.load(train_windows)
    X_val = np.load(val_windows)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train), batch_size=128, shuffle=True)

    model = LSTMAutoencoder(n_features=X_train.shape[2])

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for (batch,) in train_loader:
            optimizer.zero_grad()

            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.size(0)

        train_loss = total_loss / len(X_train)

        model.eval()
        with torch.no_grad():
            val_reconstructed = model(X_val)
            val_loss = criterion(val_reconstructed, X_val).item()

        print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"Train loss: {train_loss:.6f} | "
                f"Val loss: {val_loss:.6f}"
            )

    return model, X_train, X_val

def main():
    train_windows = "train_windows.npy"
    val_windows = "val_windows.npy"

    model_path = "processed/lstm_autoencoder.pth"
    threshold_path = "processed/anomaly_threshold.npy"

    if os.path.exists(model_path) and os.path.exists(threshold_path):
        X_train = torch.tensor(np.load(train_windows), dtype=torch.float32)
        X_val = torch.tensor(np.load(val_windows), dtype=torch.float32)

        model = LSTMAutoencoder(n_features=X_train.shape[2])
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

        threshold = float(np.load(threshold_path))

    else:
        epochs = 50
        model, X_train, X_val = load_and_train(train_windows, val_windows, epochs)

        model.eval()
        with torch.no_grad():
            train_reconstructed = model(X_train)

            train_errors = torch.mean((X_train - train_reconstructed) ** 2, dim=(1, 2)).numpy()

        threshold = train_errors.mean() + 3 * train_errors.std()
        torch.save(model.state_dict(), model_path)
        np.save(threshold_path, np.array(threshold))

    model.eval() 
    with torch.no_grad(): 
        train_reconstructed = model(X_train) 
        val_reconstructed = model(X_val) 
        train_errors = torch.mean((X_train - train_reconstructed) ** 2, dim=(1, 2)).numpy() 
        val_errors = torch.mean( (X_val - val_reconstructed) ** 2, dim=(1, 2)).numpy()

    print("\nThreshold:", threshold)
    print("Validation windows:", len(val_errors))
    print("Validation flagged:", np.sum(val_errors > threshold))
    print("Validation flagged (%):", 100 * np.mean(val_errors > threshold))

    np.save("processed/train_errors.npy", train_errors)
    np.save("processed/val_errors.npy", val_errors)

if __name__ == '__main__':
    main()