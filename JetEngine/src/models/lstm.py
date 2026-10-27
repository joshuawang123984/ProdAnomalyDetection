import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import os

import src.constants as constants

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_size):
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

def train_and_save(epochs: int) -> tuple[LSTMAutoencoder, float, torch.tensor, torch.tensor]:

    X_train = np.load(constants.TRAIN_WINDOWS_PATH)
    X_val = np.load(constants.VAL_WINDOWS_PATH)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train), batch_size=128, shuffle=True)

    model = LSTMAutoencoder(n_features=len(constants.features), hidden_size=constants.HIDDEN_SIZE)

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
        
    model.eval()
    with torch.no_grad():
        train_reconstructed = model(X_train)
        train_errors = torch.mean((X_train - train_reconstructed) ** 2, dim=(1, 2)).numpy()

    # threshold = train_errors.mean() + 3 * train_errors.std()
    threshold = np.percentile(train_errors, 95)

    torch.save(model.state_dict(), constants.MODEL_PATH)
    np.save(constants.THRESHOLD_PATH, np.array(threshold))

    return model, threshold, X_train, X_val

def load_model() -> tuple[LSTMAutoencoder, float, torch.tensor, torch.tensor]:
    X_train = np.load(constants.TRAIN_WINDOWS_PATH)
    X_val = np.load(constants.VAL_WINDOWS_PATH)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)

    model = LSTMAutoencoder(n_features=len(constants.features), hidden_size=constants.HIDDEN_SIZE)
    model.load_state_dict(torch.load(constants.MODEL_PATH, map_location="cpu"))

    threshold = float(np.load(constants.THRESHOLD_PATH))

    return model, threshold, X_train, X_val

def main():
    if os.path.exists(constants.MODEL_PATH) and os.path.exists(constants.THRESHOLD_PATH):
        model, threshold, X_train, X_val = load_model()
        print("Model Load Test Passed.")

    else:
        epochs = 1
        model, threshold, X_train, X_val = train_and_save(epochs)

    model.eval() 
    with torch.no_grad(): 
        train_reconstructed = model(X_train) 
        val_reconstructed = model(X_val) 
        train_errors = torch.mean((X_train - train_reconstructed) ** 2, dim=(1, 2)).numpy() 
        val_errors = torch.mean( (X_val - val_reconstructed) ** 2, dim=(1, 2)).numpy()

    np.save(constants.TRAIN_ERRORS_PATH, train_errors)
    np.save(constants.VAL_ERRORS_PATH, val_errors)

if __name__ == '__main__':
    main()