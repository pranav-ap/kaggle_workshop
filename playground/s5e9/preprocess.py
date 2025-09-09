import numpy as np
import polars as pl
from dataclasses import dataclass


@dataclass
class Config:
    REPO_PATH: str = 'D:/code/kaggle_workshop'

    TRAIN_CSV_PATH: str = f'{REPO_PATH}/data/playground-series-s5e9/train.csv'
    TEST_CSV_PATH: str = f'{REPO_PATH}/data/playground-series-s5e9/test.csv'
    TRAIN_SPLIT_CSV_PATH: str = f'{REPO_PATH}/data/playground-series-s5e9/train_split.csv'
    VAL_SPLIT_CSV_PATH: str = f'{REPO_PATH}/data/playground-series-s5e9/val_split.csv'

    VAL_FRACTION: float = 0.2
    RANDOM_SEED: int = 42


def train_val_split(data: pl.DataFrame):
    n = data.height
    val_size = int(n * Config.VAL_FRACTION)

    train_split = data.sample(n - val_size, seed=Config.RANDOM_SEED)
    val_split = data.sample(val_size, seed=Config.RANDOM_SEED)

    return train_split, val_split


def main():
    train_data = pl.read_csv('../../data/playground-series-s5e9/train.csv')
    test_data = pl.read_csv('../../data/playground-series-s5e9/test.csv')

    columns_to_drop = ['id']
    train_data = train_data.drop(columns_to_drop)

    train_split, val_split = train_val_split(train_data)

    train_split.write_csv(Config.TRAIN_SPLIT_CSV_PATH)
    val_split.write_csv(Config.VAL_SPLIT_CSV_PATH)
    test_data.write_csv(Config.TEST_CSV_PATH)


if __name__ == "__main__":
    main()
