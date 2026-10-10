import matplotlib.pyplot as plt
import numpy as np

def visualize(errors, threshold):
    plt.figure(figsize=(14, 5))

    plt.plot(errors, label="Reconstruction error")
    plt.axhline(
        threshold,
        color="red",
        linestyle="--",
        label="Anomaly threshold"
    )

    anomaly_indices = np.where(errors > threshold)[0]

    plt.scatter(
        anomaly_indices,
        errors[anomaly_indices],
        color="red",
        label="Detected anomalies",
        zorder=3
    )

    plt.xlabel("Validation window index")
    plt.ylabel("Reconstruction error")
    plt.title("LSTM Autoencoder Anomaly Detection")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    val_errors = np.load("processed/val_errors.npy")
    threshold = np.load("processed/anomaly_threshold.npy")

    anomaly_indices = np.where(val_errors > threshold)[0]
    
    print("Threshold:", float(threshold))
    print("Total flagged windows:", len(anomaly_indices))

    for idx in anomaly_indices:
        print(f"Window {idx}: error = {val_errors[idx]:.6f}")
        
    visualize(val_errors, threshold)
