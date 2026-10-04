import pandas as pd
from helper import add_column_header, remove_features

def main():
    df = add_column_header("CMAPSSDATA/train_FD001.txt")

    print(df.head())
    print(df.shape)
    print(df.describe().T)
    print(df.isna().sum())
    print(df.nunique())

    features_to_remove = [
        "setting_3",
        "sensor_1",
        "sensor_5",
        "sensor_10",
        "sensor_16",
        "sensor_18",
        "sensor_19"
    ]

    df = remove_features(df, features_to_remove)
    print(df.head())
    print(df.shape)

if __name__ == '__main__':
    main()