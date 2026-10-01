import pandas as pd
import numpy as np

N = 10000

np.random.seed(42)

timestamps = pd.date_range(
    start="2026-01-01",
    periods=N,
    freq="min"
)

temperature = np.random.normal(
    loc=70,
    scale=5,
    size=N
)

vibration = np.random.normal(
    loc=0.30,
    scale=0.05,
    size=N
)

pressure = np.random.normal(
    loc=100,
    scale=3,
    size=N
)

rpm = np.random.normal(
    loc=1500,
    scale=50,
    size=N
)


df = pd.DataFrame({
    "timestamp": timestamps,
    "temperature": temperature,
    "vibration": vibration,
    "pressure": pressure,
    "rpm": rpm
})
