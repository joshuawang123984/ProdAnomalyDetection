# Drift Detection Pipeline

An anomaly detection pipeline for NASA's C-MAPSS turbofan engine dataset using an LSTM autoencoder.

## Overview

- Preprocess and scale sensor data using `StandardScaler`.
- Create sliding windows of sensor readings for each engine.
- Train an LSTM autoencoder to reconstruct normal operating patterns.
- Use reconstruction error and a threshold to identify anomalous windows.
- Visualize detected anomalies across engine cycles.

## Tech Stack

- Python
- PyTorch
- NumPy and Pandas
- Scikit-learn
- Matplotlib

## Usage

### 1. Clone the repository

```bash
git clone <repository-url>
```

### 2. Navigate to correct directory

```bash
cd JetEngine
```

### 3. Generate sliding windows

```bash
python -m src.data_processing.window_generation
```

This preprocesses the dataset, scales the sensor features, and generates sliding windows for training, validation, and testing.

### 4. Train the model

```bash
python -m src.models.lstm
```

This trains the LSTM autoencoder and saves the trained model and anomaly threshold.

### 5. Visualize anomalies

```bash
python -m src.visualization.visualize
```

This visualizes reconstruction errors and detected anomalies across engine cycles.

## Dataset

This project uses NASA's C-MAPSS turbofan engine degradation dataset.

Dataset link: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

Download the dataset and place the required files in the directory expected by the preprocessing script.

## Notes

The trained model, scaler, and anomaly threshold are saved for reuse during inference, avoiding the need to retrain the model every time.
