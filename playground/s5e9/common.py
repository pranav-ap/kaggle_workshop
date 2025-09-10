import os
import warnings
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import polars as pl
import xgboost as xgb
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


@dataclass
class Config:
    REPO_PATH: str = 'D:/code/kaggle_workshop'

    TRAIN_CSV_PATH: str = f'{REPO_PATH}/data/playground-series-s5e9/train.csv'
    TEST_CSV_PATH: str = f'{REPO_PATH}/data/playground-series-s5e9/test.csv'
    TRAIN_SPLIT_CSV_PATH: str = f'{REPO_PATH}/data/playground-series-s5e9/train_split.csv'
    VAL_SPLIT_CSV_PATH: str = f'{REPO_PATH}/data/playground-series-s5e9/val_split.csv'

    VAL_FRACTION: float = 0.2
    RANDOM_SEED: int = 42


def load_train_data() -> pl.DataFrame:
    return pl.read_csv(Config.TRAIN_CSV_PATH)


def load_test_data() -> pl.DataFrame:
    return pl.read_csv(Config.TEST_CSV_PATH)


def load_train_split_data() -> Tuple[np.array, np.array]:
    data = pl.read_csv(Config.TRAIN_SPLIT_CSV_PATH)

    X_train = data.drop('BeatsPerMinute').to_numpy()
    y_train = data['BeatsPerMinute'].to_numpy()

    return X_train, y_train


def load_val_split_data() -> Tuple[np.array, np.array]:
    data = pl.read_csv(Config.VAL_SPLIT_CSV_PATH)

    X_val = data.drop('BeatsPerMinute').to_numpy()
    y_val = data['BeatsPerMinute'].to_numpy()

    return X_val, y_val


def save_model(model: xgb.XGBRegressor):
    model_path = os.path.join(Config.REPO_PATH, "outputs", "s5e9", "xgb_model.json")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    return model_path


def load_model() -> xgb.XGBRegressor:
    model_path = os.path.join(Config.REPO_PATH, "outputs", "s5e9", "xgb_model.json")
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model


def plot_charts(model: xgb.XGBRegressor):
    results = model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)

    # RMSE plot
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
    ax.plot(x_axis, results['validation_1']['rmse'], label='Validation')
    ax.legend()
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("XGBoost RMSE")

    plot_path = os.path.join(Config.REPO_PATH, "outputs", "s5e9", "xgb_rmse.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved RMSE plot at {plot_path}")

    return
