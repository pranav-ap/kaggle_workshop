import polars as pl
from common import Config


def train_val_split(data: pl.DataFrame):
    n = data.height
    val_size = int(n * Config.VAL_FRACTION)

    train_split = data.sample(n - val_size, seed=Config.RANDOM_SEED)
    val_split = data.sample(val_size, seed=Config.RANDOM_SEED)

    return train_split, val_split


def main():
    train_data = pl.read_csv(f'../../data/playground-series-{Config.PROJECT_NAME}/train.csv')
    test_data = pl.read_csv(f'../../data/playground-series-{Config.PROJECT_NAME}/test.csv')

    columns_to_drop = ['id']
    train_data = train_data.drop(columns_to_drop)

    train_split, val_split = train_val_split(train_data)

    train_split.write_csv(Config.TRAIN_SPLIT_CSV_PATH)
    val_split.write_csv(Config.VAL_SPLIT_CSV_PATH)
    test_data.write_csv(Config.TEST_CSV_PATH)


if __name__ == "__main__":
    main()