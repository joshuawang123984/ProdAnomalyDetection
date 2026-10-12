import matplotlib.pyplot as plt
import numpy as np

def group_anomalies(anomaly_indices, max_gap=2):
    if len(anomaly_indices) == 0:
        return []

    groups = []
    start = prev = anomaly_indices[0]

    for idx in anomaly_indices[1:]:
        if idx - prev > max_gap + 1:
            groups.append((start, prev))
            start = idx

        prev = idx

    groups.append((start, prev))
    return groups

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
    test = True
    if test:
        errors = np.load("processed/test_errors.npy")
    else:
        errors = np.load("processed/val_errors.npy")

    threshold = np.load("processed/anomaly_threshold.npy")

    anomaly_indices = np.where(errors > threshold)[0]
    anomaly_indices = np.where(errors > threshold)[0]
    
    print("Threshold:", float(threshold))
    print("Total flagged windows:", len(anomaly_indices))

    groups = group_anomalies(anomaly_indices)

    for start, end in groups:
        errors_info = errors[start:end + 1]

        print(
            f"{start}-{end}: "
            f"peak={errors_info.max():.4f}, "
            f"mean={errors_info.mean():.4f}"
        )

    visualize(errors, threshold)
