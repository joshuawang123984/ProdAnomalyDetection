import numpy as np
import torch
from lstm import LSTMAutoencoder
import matplotlib.pyplot as plt

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

    feature_idx = 8

    for idx in [1198, 2123, 4486]:
        original = X_val[idx, :, feature_idx].numpy()
        reconstructed_feature = reconstructed[idx, :, feature_idx].numpy()

        plt.figure(figsize=(12, 4))
        plt.plot(original, label="Original")
        plt.plot(reconstructed_feature, label="Reconstructed")
        plt.title(f"Window {idx} — Feature {feature_idx}")
        plt.legend()
        plt.show()

    sensor_8 = X_val[:, :, 8].numpy()

    sensor_8 = sensor_8.reshape(-1)

    plt.figure(figsize=(15, 5))
    plt.plot(sensor_8)
    plt.xlabel("Time step")
    plt.ylabel("Sensor 8 measurement")
    plt.title("Sensor 8 Across Validation Data")
    plt.tight_layout()
    plt.show()
