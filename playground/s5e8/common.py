import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import polars as pl
import xgboost as xgb
from prefect import task
from prefect.assets import materialize
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


@dataclass
class Config:
    REPO_PATH: str = 'D:/code/kaggle_workshop'

    TRAIN_CSV_PATH: str = f'{REPO_PATH}/data/playground-series-s5e8/train.csv'
    TEST_CSV_PATH: str = f'{REPO_PATH}/data/playground-series-s5e8/test.csv'
    TRAIN_SPLIT_CSV_PATH: str = f'{REPO_PATH}/data/playground-series-s5e8/train_split.csv'
    VAL_SPLIT_CSV_PATH: str = f'{REPO_PATH}/data/playground-series-s5e8/val_split.csv'

    VAL_FRACTION: float = 0.2
    RANDOM_SEED: int = 42


@materialize(f'file:///{Config.TRAIN_CSV_PATH}')
def load_train_data() -> pl.DataFrame:
    return pl.read_csv(Config.TRAIN_CSV_PATH)


@materialize(f'file:///{Config.TEST_CSV_PATH}')
def load_test_data() -> pl.DataFrame:
    return pl.read_csv(Config.TEST_CSV_PATH)


@materialize(
    f'file:///{Config.TRAIN_SPLIT_CSV_PATH}',
    f'file:///{Config.VAL_SPLIT_CSV_PATH}'
)
def train_val_split(data: pl.DataFrame):
    n = data.height
    val_size = int(n * Config.VAL_FRACTION)

    train_split = data.sample(n - val_size, seed=Config.RANDOM_SEED)
    val_split = data.sample(val_size, seed=Config.RANDOM_SEED)

    train_split.write_csv(Config.TRAIN_SPLIT_CSV_PATH)
    val_split.write_csv(Config.VAL_SPLIT_CSV_PATH)

    return train_split, val_split


@task
def process_data(data: pl.DataFrame, stage='train') -> Tuple[np.array, np.array]:
    ids = data['id'].to_numpy()

    columns_to_drop = ['id', 'day', 'month', 'contact']
    data = data.drop(columns_to_drop)

    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome', 'y']

    if stage == 'test':
        cat_cols.remove('y')

    data = data.with_columns([
        pl.col(c)
        .cast(pl.Categorical)
        .to_physical()
        .cast(pl.Int32)
        for c in cat_cols
    ])

    if stage == 'test':
        X = data.to_numpy()
        return ids, X

    X = data.drop("y").to_numpy()
    y = data["y"].to_numpy()

    return X, y


@materialize(f"file:///{Config.REPO_PATH}/outputs/s5e8/xgb_model.json")
def save_model(model: xgb.XGBClassifier):
    model_path = os.path.join(Config.REPO_PATH, "outputs", "s5e8", "xgb_model.json")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    return model_path


@materialize(f"file:///{Config.REPO_PATH}/outputs/s5e8/xgb_model.json")
def load_model() -> xgb.XGBClassifier:
    model_path = os.path.join(Config.REPO_PATH, "outputs", "s5e8", "xgb_model.json")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model


@materialize(
    f"file:///{Config.REPO_PATH}/outputs/s5e8/xgb_logloss.png",
    log_prints=True,
)
def plot_charts(model: xgb.XGBClassifier):
    results = model.evals_result()
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)

    # Log Loss plot

    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Validation')
    ax.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.title("XGBoost Log Loss")

    plot_path = os.path.join(Config.REPO_PATH, "outputs", "s5e8", "xgb_logloss.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Saved log loss plot at {plot_path}")

    return

