import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import src.constants as constants
from src.data_processing import add_column_header, create_windows
from src.evaluation.evaluation import summarize

def main():
    scaler = joblib.load(constants.SCALER_PATH)

    if os.path.exists(constants.BASELINE_PATH):
        model = joblib.load(constants.BASELINE_PATH)
    else:
        X_train = np.load(constants.TRAIN_WINDOWS_PATH)
        model = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
        model.fit(X_train.reshape(len(X_train), -1))
        joblib.dump(model, constants.BASELINE_PATH)

    def score(windows):
        return -model.score_samples(windows.reshape(len(windows), -1))  

    X_val = np.load(constants.VAL_WINDOWS_PATH)
    vm = pd.read_csv(constants.VAL_WINDOW_METADATA_PATH)
    last = vm.groupby("unit_id")["end_cycle"].transform("max")
    healthy = ((vm["end_cycle"] / last) < 0.5).to_numpy()
    threshold = np.percentile(score(X_val)[healthy], 99)

    results = {}
    for n in (1, 2, 3, 4):
        X = scaler.transform(add_column_header(f"CMAPSSData/test_FD00{n}.txt"))
        windows, meta = create_windows(X, scaler.keep, window_size=constants.WINDOW_SIZE)
        meta["error"] = score(windows)

        rul = np.loadtxt(f"CMAPSSData/RUL_FD00{n}.txt")
        meta["final_cycle"] = meta.groupby("unit_id")["end_cycle"].transform("max")
        meta["window_rul"] = rul[meta["unit_id"].to_numpy().astype(int) - 1] + meta["final_cycle"] - meta["end_cycle"]

        res = summarize(meta, threshold)
        res["dropped (too short)"] = len(rul) - res["engines"]
        results[f"FD00{n}"] = res

    print(pd.DataFrame(results).T.round(3))


if __name__ == "__main__":
    main()