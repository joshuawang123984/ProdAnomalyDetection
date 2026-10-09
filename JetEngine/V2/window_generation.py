import numpy as np
import pandas as pd

from helper import add_column_header

def create_windows(df: pd.DataFrame, features: list[str], window_size: int = 30) -> tuple[np.ndarray, np.ndarray]:
    windows = []
    metadata = []

    df = df.sort_values(["unit_id", "cycle"])

    for unit_id, engine in df.groupby("unit_id"):
        values = engine[features].to_numpy()

        for end in range(window_size, len(values) + 1):
            windows.append(values[end - window_size:end])

            metadata.append({
                "unit_id": unit_id,
                "start_cycle": int(engine["cycle"].iloc[end - window_size]),
                "end_cycle": int(engine["cycle"].iloc[end - 1])
            })

    return np.array(windows), pd.DataFrame(metadata)

if __name__ == '__main__':
    df = add_column_header("../CMAPSSDATA/train_FD001.txt")
    features = [col for col in df.columns if col not in ["unit_id", "cycle"]]
    windows, metadata = create_windows(df, features, window_size=30)

    np.save("train_windows.npy", windows)
    metadata.to_csv("train_window_metadata.csv", index=False)

    print("Windows shape:", windows.shape)
    print("Metadata shape:", metadata.shape)
