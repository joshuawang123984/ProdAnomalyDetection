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

def visualize_rul(metadata, unit_id):
    engine = metadata[metadata["unit_id"] == unit_id].sort_values("window_rul")

    plt.figure(figsize=(14, 5))
    plt.plot(engine["window_rul"], engine["error"])

    plt.axhline(
        float(np.load("processed/anomaly_threshold.npy")),
        color="red",
        linestyle="--",
        label="Anomaly threshold"
    )

    plt.xlabel("Remaining Useful Life (cycles)")
    plt.ylabel("Reconstruction error")
    plt.title(f"Reconstruction Error vs RUL — Engine {unit_id}")
    plt.gca().invert_xaxis()
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
        rul_values = np.loadtxt("CMAPSSDATA/RUL_FD001.txt")
        metadata["final_cycle"] = metadata.groupby("unit_id")["end_cycle"].transform("max")

        metadata["final_rul"] = metadata["unit_id"].map(lambda uid: rul_values[int(uid) - 1])

        metadata["window_rul"] = (
            metadata["final_rul"]
            + metadata["final_cycle"]
            - metadata["end_cycle"]
        )
    else:
        # validation engines come from training data, which runs to failure
        metadata["final_cycle"] = metadata.groupby("unit_id")["end_cycle"].transform("max")
        metadata["window_rul"] = metadata["final_cycle"] - metadata["end_cycle"]

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

    for i in range(1, 100):
        visualize_rul(metadata, unit_id=i)