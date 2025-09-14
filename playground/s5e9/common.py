import os
from dataclasses import dataclass
from typing import Tuple
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
import yaml


@dataclass
class Config:
    REPO_PATH: Path = Path('D:/code/kaggle_workshop')
    PROJECT_NAME: str = 's5e9'

    TRAIN_CSV_PATH: str = f'{REPO_PATH}/data/playground-series-{PROJECT_NAME}/train.csv'
    TEST_CSV_PATH: str = f'{REPO_PATH}/data/playground-series-{PROJECT_NAME}/test.csv'
    TRAIN_SPLIT_CSV_PATH: str = f'{REPO_PATH}/data/playground-series-{PROJECT_NAME}/train_split.csv'
    VAL_SPLIT_CSV_PATH: str = f'{REPO_PATH}/data/playground-series-{PROJECT_NAME}/val_split.csv'

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


def save_model(model: xgb.Booster):
    model_path = Config.REPO_PATH / "outputs" / Config.PROJECT_NAME / "xgb_model.json"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    return model_path


def load_model() -> xgb.Booster:
    model_path = Config.REPO_PATH / "outputs" / Config.PROJECT_NAME / "xgb_model.json"
    model = xgb.Booster()
    model.load_model(model_path)
    return model


def store_best_params(best_params):
    best_params_path = Config.REPO_PATH / "outputs" / Config.PROJECT_NAME / "best_params.yaml"
    with open(best_params_path, "w") as f:
        yaml.dump(best_params, f, sort_keys=False, default_flow_style=False)


def load_best_params():
    best_params_path = Config.REPO_PATH / "outputs" / Config.PROJECT_NAME / "best_params.yaml"
    with open(best_params_path, "r") as f:
        params = yaml.safe_load(f)

    return params


def store_metrics(metrics: dict):
    metrics_path = Config.REPO_PATH / "outputs" / Config.PROJECT_NAME / "metrics.yaml"
    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f, sort_keys=False, default_flow_style=False)


def load_metrics():
    metrics_path = Config.REPO_PATH / "outputs" / Config.PROJECT_NAME / "metrics.yaml"
    with open(metrics_path, "r") as f:
        metrics = yaml.safe_load(f)

    return metrics

