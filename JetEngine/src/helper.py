import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def find_correlations(df: pd.DataFrame, features: list[str]) -> None:
    correlation = df[features].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Feature Correlation")
    plt.show()

def plot_features(df: pd.DataFrame, features: list[str], unit_id: int) -> None:
    engine = df[df["unit_id"] == unit_id]

    fig, axes = plt.subplots(5, 4, figsize=(16, 14))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        axes[i].plot(engine["cycle"], engine[feature])
        axes[i].set_title(feature)
        axes[i].set_xlabel("Cycle")
        axes[i].set_ylabel("Value")

    for i in range(len(features), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_feature(df: pd.DataFrame, feature: str, unit_id: int) -> None:
    engine = df[df["unit_id"] == unit_id]

    plt.plot(engine["cycle"], engine[feature])
    plt.xlabel("Cycle")
    plt.ylabel(feature)
    plt.title(f"{feature} over time - Engine {unit_id}")
    plt.show()

def remove_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    df = df.drop(columns=features)
    return df

def add_column_header(filename: str) -> pd.DataFrame:
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

    df = pd.read_csv(
        filename,
        sep=r"\s+",
        header=None,
        names=columns
    )

    return df