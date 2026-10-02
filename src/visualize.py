import matplotlib.pyplot as plt
from data_script import df

features = [
    "temperature",
    "vibration",
    "pressure",
    "rpm"
]

for feature in features:
    plt.figure(figsize=(12, 4))
    plt.plot(df[feature])
    plt.title(feature)
    plt.xlabel("Time")
    plt.ylabel(feature)
    plt.show()