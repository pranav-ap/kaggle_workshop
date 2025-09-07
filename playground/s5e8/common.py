from dataclasses import dataclass

import polars as pl

from prefect import task
from prefect.assets import materialize

import warnings

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
def process_data(data: pl.DataFrame):
    columns_to_drop = ['id', 'day', 'month', 'contact']
    data = data.drop(columns_to_drop)

    cat_cols = [
        'job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome', 'y'
    ]

    data = data.with_columns([
        pl.col(c).cast(pl.Categorical) for c in cat_cols
    ])

    return data

