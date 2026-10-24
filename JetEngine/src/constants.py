'''
NEXT STEP: finish evaluation script. 
change the constants to the right filename to generate correct windows, then properly visualize in evaluation
'''

# model load and training details
MODEL_PATH = "processed/lstm_autoencoder.pth"
SCALER_PATH = "processed/scaler.joblib"
THRESHOLD_PATH = "processed/anomaly_threshold.npy"

#cmaps datasets
TRAIN_DATASET_PATH = "CMAPSSDATA/train_FD001.txt"
TEST_DATASET_PATH = "CMAPSSDATA/test_FD001.txt"
RUL_DATASET_PATH = "CMAPSSDATA/RUL_FD001.txt"

#window generation metadata
TRAIN_WINDOW_METADATA_PATH = "processed/train_window_metadata.csv"
VAL_WINDOW_METADATA_PATH = "processed/val_window_metadata.csv"
TEST_WINDOW_METADATA_PATH = "processed/test_window_metadata.csv"

# npy array of windows after generation
TRAIN_WINDOWS_PATH = "processed/train_windows.npy"
VAL_WINDOWS_PATH = "processed/val_windows.npy"
TEST_WINDOWS_PATH = "processed/test_windows.npy"

# model errors 
TRAIN_ERRORS_PATH = "processed/train_errors.npy"
VAL_ERRORS_PATH = "processed/val_errors.npy"
TEST_ERRORS_PATH = "processed/test_errors.npy"

HIDDEN_SIZE = 128

# All columns
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

# All fetures
features = [
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

WINDOW_SIZE = 25