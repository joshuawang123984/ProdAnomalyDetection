
import pandas as pd

import src.constants as constants

def combine_datasets() -> pd.DataFrame:
    datasets = []

    for dataset_id in range(1, 5):
        filename = f"CMAPSSData/train_FD00{dataset_id}.txt"

        df = pd.read_csv(
            filename,
            sep=r"\s+",
            header=None,
            names=constants.columns,
        )

        df["original_unit_id"] = df["unit_id"]
        df["dataset_id"] = dataset_id

        df["unit_id"] = (df["unit_id"] + (dataset_id - 1) * 100)
        datasets.append(df)

    combined = pd.concat(datasets, ignore_index=True)

    output_path = constants.COMBINED_DATASET_PATH
    combined.to_csv(output_path, sep=" ", index=False, header=False)

    return combined

if __name__ == "__main__":
    combine_datasets()
