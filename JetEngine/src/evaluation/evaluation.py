import numpy as np
import torch
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt

import src.constants as constants
from src.data_processing import add_column_header, create_windows
from src.models import load_model

def visualize_rul(metadata, unit_id: int, threshold: int) -> None:
    engine = metadata[metadata["unit_id"] == unit_id].sort_values("window_rul")

    plt.figure(figsize=(14, 5))
    plt.plot(engine["window_rul"], engine["error"])

    plt.axhline(
        threshold,
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

def main(): 
    X = add_column_header(constants.TEST_DATASET_PATH)
    if os.path.exists(constants.SCALER_PATH):
        scaler = joblib.load(constants.SCALER_PATH)
        features = scaler.keep
    else:
        raise FileNotFoundError("Scaler File not found.")
    
    X = scaler.transform(X)

    windows, metadata = create_windows(X, features, window_size=constants.WINDOW_SIZE)
    windows = torch.tensor(windows, dtype=torch.float32)

    model, _, _ = load_model(features)

    model.eval()
    with torch.no_grad():
        reconstructed = model(windows)

        errors = torch.mean((windows - reconstructed) ** 2, dim=(1, 2)).cpu().numpy()

    metadata["error"] = errors
    rul_values = np.loadtxt(constants.RUL_DATASET_PATH)

    metadata["final_cycle"] = metadata.groupby("unit_id")["end_cycle"].transform("max")
    metadata["final_rul"] = metadata["unit_id"].map(lambda uid: rul_values[int(uid) - 1])
    metadata["window_rul"] = (metadata["final_rul"] + metadata["final_cycle"] - metadata["end_cycle"])

    threshold = float(np.load(constants.THRESHOLD_PATH))
    for i in range(1, 100):
            visualize_rul(metadata, unit_id=i, threshold=threshold)

if __name__ == '__main__':
    main()