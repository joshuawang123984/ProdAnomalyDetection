import os
import torch
import numpy as np
from lstm import LSTMAutoencoder

def main():
    test_windows = "processed/test_windows.npy"

    model_path = "processed/lstm_autoencoder.pth"
    threshold_path = "processed/anomaly_threshold.npy"

    if os.path.exists(model_path) and os.path.exists(threshold_path):
        X_test = torch.tensor(np.load(test_windows), dtype=torch.float32)

        model = LSTMAutoencoder(n_features=X_test.shape[2], hidden_size=64)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

        threshold = float(np.load(threshold_path))

    else:
        raise Exception

    model.eval() 
    with torch.no_grad(): 
        test_reconstructed = model(X_test) 
        test_errors = torch.mean((X_test - test_reconstructed) ** 2, dim=(1, 2)).numpy() 

    anomaly_indices = np.where(test_errors > threshold)[0]

    print(f"Threshold: {threshold:.4f}")
    print(f"Total test windows: {len(test_errors)}")
    print(f"Flagged windows: {len(anomaly_indices)}")
    print(f"Flagged percentage: {len(anomaly_indices) / len(test_errors) * 100:.2f}%")
    print(f"Maximum reconstruction error: {test_errors.max():.4f}")

    np.save("processed/test_errors.npy", test_errors)

if __name__ == '__main__':
    main()