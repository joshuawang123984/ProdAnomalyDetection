import numpy as np
import pandas as pd
import joblib

from src.scaler import CondScaler
import src.constants as constants

def add_column_header(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, sep=r"\s+", header=None)
    df.columns = constants.columns

    return df

def split_units(train_df: pd.DataFrame, val_frac: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]: 
    ids = np.sort(train_df["unit_id"].unique())
    rng = np.random.default_rng(seed)

    train_units, val_units, start = [], [], 0
    for n in constants.ENGINES_PER_DATASET:
        block = rng.permutation(ids[start: start + n])
        k = int(round(val_frac * n))
        val_units.extend(block[:k])
        train_units.extend(block[k:])
        start += n

    return np.array(train_units), np.array(val_units)

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

def scale_df(train_df: pd.DataFrame, features: list[str], n_conditions=6) -> tuple[pd.DataFrame, pd.DataFrame, CondScaler]:
    train_units, val_units = split_units(train_df)

    train_split = train_df[train_df["unit_id"].isin(train_units)].copy()
    val_split = train_df[train_df["unit_id"].isin(val_units)].copy()

    scaler = CondScaler(features, n_conditions=n_conditions)
    scaler.fit(train_split)

    train_split = scaler.transform(train_split)
    val_split = scaler.transform(val_split)

    joblib.dump(scaler, constants.SCALER_PATH)
    return train_split, val_split, scaler

if __name__ == '__main__':
    df = add_column_header(constants.TRAIN_DATASET_PATH)

    # assumes train and test have same features
    train_df, val_df, scaler = scale_df(df, constants.features)
    new_features = scaler.keep

    window_size = constants.WINDOW_SIZE
    train_windows, train_metadata = create_windows(train_df, new_features, window_size=window_size)
    val_windows, val_metadata = create_windows(val_df, new_features, window_size=window_size)

    np.save(constants.TRAIN_WINDOWS_PATH, train_windows)
    np.save(constants.VAL_WINDOWS_PATH, val_windows)

    train_metadata.to_csv(constants.TRAIN_WINDOW_METADATA_PATH, index=False)
    val_metadata.to_csv(constants.VAL_WINDOW_METADATA_PATH, index=False)
