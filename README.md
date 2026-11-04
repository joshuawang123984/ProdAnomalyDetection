# Turbofan Degradation Detection (NASA C-MAPSS)

Unsupervised detection of engine degradation on NASA's C-MAPSS turbofan dataset (FD001-FD004), comparing an LSTM autoencoder against an Isolation Forest baseline. Each test engine gets one score at its last observed window and is evaluated on whether it is within 30 cycles of failure.

## Results

AUROC per test subset (higher is better; 0.5 is chance):

| Model | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| LSTM autoencoder (hidden 128, all cycles) | 0.62 | 0.82 | 0.66 | 0.74 |
| LSTM autoencoder (hidden 32, all cycles) | 0.54 | 0.83 | 0.64 | 0.77 |
| Isolation Forest (all cycles) | 0.96 | 0.97 | 0.90 | 0.82 |
| Near-failure engines in test set | 25 | 61 | 20 | 53 |

With only 20-61 near-failure engines per subset, AUROCs are uncertain by roughly +/-0.07 to +/-0.12, and each row is a single training run.

## Findings

The Isolation Forest scored higher than the LSTM autoencoder on every subset. The gap is large on FD001-FD003 and small on FD004, where it is within noise. Hidden size (128 vs 32) made no distinguishable difference.

A possible explanation, not tested here: the autoencoder was trained on every cycle, including degraded ones, so it may learn to reconstruct the drifted sensor levels that the Isolation Forest flags as unusual. Also, the # of epochs used was 10, which is slightly low. However, an increase in epochs would also require a technique to offset the increase in memorization, so techniques like dropout would need to be used.

## Approach

**Data.** C-MAPSS FD001-FD004: 709 training engines and, across the four test sets, 507 engines (after dropping 10 too short to window), each with a true remaining useful life (RUL) at its last cycle. The four training sets are combined with unique engine ids.

**Preprocessing.**
- normalization: the three operating settings are clustered into 6 regimes, and each sensor is z-scored within its regime. Sensors with no variation inside a regime are dropped, and values are clipped to +/-10 z-score.
- The scaler is fit on training engines only.
- 25 cycle sliding windows are built per engine.
- 20% of engines from each subset are held out for validation.

**Models.**
- LSTM autoencoder: encoder LSTM, last hidden state repeated over the window, decoder LSTM, linear output. Trained with MSE for 10 epochs (Adam, lr 1e-3, batch 128). The anomaly score is reconstruction error.
- Isolation Forest: 200 trees on flattened windows. The anomaly score is the negative `score_samples` value.

**Evaluation.** For each test engine, the score of its last window is compared against the label "RUL <= 30" using AUROC, per subset. AUROC needs no threshold. Test engines shorter than one window are excluded (3 in FD002, 7 in FD004).

## Limitations

- Small positive counts and single training runs, as noted above.
- A single pooled threshold is miscalibrated across subsets -> Each regime has a different threshold, however I only used 1 global threshold lowering the accuracy of all the evaluations of the datasets, especially dataset 1.
- No standalone inference script: evaluation runs on the four C-MAPSS test files.

## Issues found along the way

- The first combined dataset file had two extra columns, which silently shifted every sensor by two positions.
- Engine ids collided across subsets, which merged different engines into the same windows.
- The validation split took the last 30% of ids in file order, so validation was entirely FD004.

## Tech Stack

Python, PyTorch, NumPy, Pandas, Scikit-learn, Matplotlib

## Reproduce

Download the dataset from https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data and place `train_FD00x.txt`, `test_FD00x.txt` and `RUL_FD00x.txt` in `CMAPSSData/`.

```bash
git clone <repository-url>
cd JetEngine
mkdir processed
pip install -r requirements.txt

python -m src.data_processing.combine_data       # combine the four training files
python -m src.data_processing.window_generation  # scale, split, window
python -m src.models.lstm                        # train LSTM autoencoder
python -m src.evaluation.evaluation              # per engine AUROC table (LSTM)
python -m src.baseline.baseline                  # same table for Isolation Forest
```

Training is skipped when saved models exist. To retrain after changing the data or the model, delete `processed/lstm_autoencoder.pth`, `processed/anomaly_threshold.npy` and `processed/isolation_forest.joblib` first.