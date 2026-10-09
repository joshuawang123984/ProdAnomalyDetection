import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def add_column_header(filename: str) -> pd.DataFrame:
    columns = [
        "unit_id",
        "cycle",
        "setting_1",
        "setting_2",
        "setting_3",
        "sensor_1",
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_5",
        "sensor_6",
        "sensor_7",
        "sensor_8",
        "sensor_9",
        "sensor_10",
        "sensor_11",
        "sensor_12",
        "sensor_13",
        "sensor_14",
        "sensor_15",
        "sensor_16",
        "sensor_17",
        "sensor_18",
        "sensor_19",
        "sensor_20",
        "sensor_21"
    ]

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

    joblib.dump(scaler, "scaler.joblib")
    return train_split, val_split, test_split

if __name__ == '__main__':
    df = add_column_header("../CMAPSSDATA/train_FD001.txt")
    test_df = add_column_header("../CMAPSSDATA/test_FD001.txt")

    # assumes train and test have same features
    features = [col for col in df.columns if col not in ["unit_id", "cycle"]]
    train_df, val_df, test_df = scale_df(df, test_df, features)

    train_windows, train_metadata = create_windows(train_df, features, window_size=30)
    val_windows, val_metadata = create_windows(val_df, features, window_size=30)
    test_windows, test_metadata = create_windows(test_df, features, window_size=30)

    np.save("train_windows.npy", train_windows)
    np.save("val_windows.npy", val_windows)
    np.save("test_windows.npy", test_windows)

    train_metadata.to_csv("train_window_metadata.csv", index=False)
    val_metadata.to_csv("val_window_metadata.csv", index=False)
    test_metadata.to_csv("test_window_metadata.csv", index=False)
