import pandas as pd

import src.constants as constants

def combine_datasets() -> pd.DataFrame:
    datasets = []
    offset = 0

    for dataset_id in range(1, 5):
        filename = f"CMAPSSData/train_FD00{dataset_id}.txt"

        df = pd.read_csv(filename, sep=r"\s+", header=None)
        assert df.shape[1] == len(constants.columns), f"{filename}: {df.shape[1]} fields"
        df.columns = constants.columns

        df["unit_id"] += offset
        offset = df["unit_id"].max()
        datasets.append(df)

    combined = pd.concat(datasets, ignore_index=True)

    assert not combined.duplicated(["unit_id", "cycle"]).any(), "unit_id collision"

    combined.to_csv(constants.COMBINED_DATASET_PATH, sep=" ", index=False, header=False)
    return combined

if __name__ == "__main__":
    combine_datasets()