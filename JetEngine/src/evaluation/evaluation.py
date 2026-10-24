import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt

import src.constants as constants
from src.models import load_model
from src.data_processing import add_column_header, create_windows

def main(): 
    X = add_column_header(constants.TEST_DATASET_PATH)

    scaler = joblib.load(constants.SCALER_PATH)
    X[constants.features] = scaler.transform(X[constants.features])

    windows, metadata = create_windows(X, constants.features, window_size=25)

    X_tensor = torch.tensor(windows, dtype=torch.float32)

    print("Test windows shape:", X_tensor.shape)
    print("Test min:", X_tensor.min().item())
    print("Test max:", X_tensor.max().item())
    print("Test mean:", X_tensor.mean().item())

    model, threshold, _, _ = load_model()

    model.eval()
    with torch.no_grad():
        reconstructed = model(X_tensor)

        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=(1, 2)).cpu().numpy()

    rul_values = np.loadtxt(constants.RUL_DATASET_PATH)

    unit_id = 5

    # Find the windows belonging to this engine
    engine_mask = metadata["unit_id"] == unit_id
    engine_metadata = metadata.loc[engine_mask]
    engine_errors = errors[engine_mask.to_numpy()]

    # Plot reconstruction errors and threshold
    plt.figure(figsize=(12, 5))
    plt.plot(
        engine_metadata["end_cycle"],
        engine_errors,
        label="Reconstruction Error"
    )
    plt.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        label=f"Threshold ({threshold:.4f})"
    )

    plt.xlabel("Engine Cycle")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.title(f"Engine {unit_id}: Reconstruction Error")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()