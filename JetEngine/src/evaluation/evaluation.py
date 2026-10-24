import numpy as np
import torch
import joblib
import pandas as pd

import src.constants as constants
from src.data_processing import add_column_header, create_windows
from src.models import load_model
from src.visualization import visualize_rul

def main(): 
    X = np.load(constants.TEST_WINDOWS_PATH)

    X = add_column_header(constants.TEST_DATASET_PATH)
    scaler = joblib.load(constants.SCALER_PATH)
    X[constants.features] = scaler.transform(X[constants.features])

    windows, metadata = create_windows(X, constants.features, window_size=constants.WINDOW_SIZE)
    windows = torch.tensor(windows, dtype=torch.float32)

    model, threshold, _, _ = load_model()

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