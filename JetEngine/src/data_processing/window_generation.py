import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from src.constants import columns, features

def add_column_header(filename: str) -> pd.DataFrame:
    df = pd.read_csv(
        filename,
        sep=r"\s+",
        header=None,
        names=columns
    )
    return df

def create_windows(df: pd.DataFrame, features: list[str], window_size: int = 30) -> tuple[np.ndarray, pd.DataFrame]:
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

def scale_df(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unit_ids = train_df["unit_id"].unique()

    train_units = unit_ids[:int(0.7 * len(unit_ids))]
    val_units = unit_ids[int(0.7 * len(unit_ids)):]

    train_split = train_df[train_df["unit_id"].isin(train_units)].copy()
    val_split = train_df[train_df["unit_id"].isin(val_units)].copy()
    test_split = test_df.copy()

    scaler = StandardScaler()
    scaler.fit(train_split[features])

    train_split[features] = scaler.transform(train_split[features])
    val_split[features] = scaler.transform(val_split[features])
    test_split[features] = scaler.transform(test_split[features])

    joblib.dump(scaler, "processed/scaler.joblib")
    return train_split, val_split, test_split

if __name__ == '__main__':
    train_dataset = "CMAPSSDATA/train_FD001.txt"
    test_dataset = "CMAPSSDATA/test_FD001.txt"
    df = add_column_header(train_dataset)
    test_df = add_column_header(test_dataset)

    # assumes train and test have same features
    train_df, val_df, test_df = scale_df(df, test_df, features)

    window_size = 25
    train_windows, train_metadata = create_windows(train_df, features, window_size=window_size)
    val_windows, val_metadata = create_windows(val_df, features, window_size=window_size)
    test_windows, test_metadata = create_windows(test_df, features, window_size=window_size)

    np.save("processed/train_windows.npy", train_windows)
    np.save("processed/val_windows.npy", val_windows)
    np.save("processed/test_windows.npy", test_windows)

    train_metadata.to_csv("processed/train_window_metadata.csv", index=False)
    val_metadata.to_csv("processed/val_window_metadata.csv", index=False)
    test_metadata.to_csv("processed/test_window_metadata.csv", index=False)
