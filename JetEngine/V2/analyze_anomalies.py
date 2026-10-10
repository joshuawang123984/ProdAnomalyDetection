import numpy as np
import torch
from lstm import LSTMAutoencoder

if __name__ == '__main__':
    X_val = torch.tensor(np.load("val_windows.npy"), dtype=torch.float32)

    val_errors = np.load("processed/val_errors.npy")

    model = LSTMAutoencoder(n_features=X_val.shape[2])
    model.load_state_dict(torch.load("processed/lstm_autoencoder.pth", map_location="cpu"))

    model.eval()
    with torch.no_grad():
        reconstructed = model(X_val)

    feature_errors = torch.mean((X_val - reconstructed) ** 2,dim=1).numpy()

    for idx in [1198, 2123, 4486]:
        top_features = np.argsort(feature_errors[idx])[::-1][:10]

        print(f"\nWindow {idx}: total error = {val_errors[idx]:.4f}")
        print("Top contributing features:")

        for j in top_features:
            print(f"  Feature {j}: {feature_errors[idx, j]:.4f}")
