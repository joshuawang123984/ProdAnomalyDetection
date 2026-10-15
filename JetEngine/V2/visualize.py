import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def visualize(errors, threshold, metadata, unit_id):
    engine = metadata[metadata["unit_id"] == unit_id].copy()
    engine = engine.sort_values("end_cycle")

    plt.figure(figsize=(14, 5))

    plt.plot(
        engine["end_cycle"],
        engine["error"],
        label="Reconstruction error"
    )

    plt.axhline(
        threshold,
        color="red",
        linestyle="--",
        label="Anomaly threshold"
    )

    anomalies = engine[engine["error"] > threshold]

    plt.scatter(
        anomalies["end_cycle"],
        anomalies["error"],
        color="red",
        label="Detected anomalies",
        zorder=3
    )

    plt.xlabel("Operating cycle")
    plt.ylabel("Reconstruction error")
    plt.title(f"LSTM Autoencoder — Engine {unit_id}")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    test = True

    if test:
        errors = np.load("processed/test_errors.npy")
        metadata = pd.read_csv("processed/test_window_metadata.csv")
    else:
        errors = np.load("processed/val_errors.npy")
        metadata = pd.read_csv("processed/val_window_metadata.csv")

    threshold = float(np.load("processed/anomaly_threshold.npy"))

    assert len(errors) == len(metadata), "Errors and metadata lengths do not match"
    metadata["error"] = errors

    if test:
        rul_values = np.loadtxt("../CMAPSSDATA/RUL_FD001.txt")
        metadata["final_cycle"] = metadata.groupby("unit_id")["end_cycle"].transform("max")

        metadata["final_rul"] = metadata["unit_id"].map(lambda uid: rul_values[int(uid) - 1])

        metadata["window_rul"] = (
            metadata["final_rul"]
            + metadata["final_cycle"]
            - metadata["end_cycle"]
        )

    anomaly_indices = np.where(errors > threshold)[0]

    print("Threshold:", threshold)
    print("Total flagged windows:", len(anomaly_indices))

    groups = group_anomalies(anomaly_indices)

    for start, end in groups:
        errors_info = errors[start:end + 1]
        print(
            f"{start}-{end}: "
            f"peak={errors_info.max():.4f}, "
            f"mean={errors_info.mean():.4f}"
        )

    visualize(errors, threshold, metadata=metadata, unit_id=24)