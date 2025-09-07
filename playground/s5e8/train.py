import warnings
warnings.filterwarnings('ignore')

from prefect import flow
from common import load_train_data, train_val_split, process_data


@flow
def train() -> list[str]:
    data = load_train_data()
    data = train_val_split(data)
    train_split, val_split = process_data.map(data)

    results = []
    return results


if __name__ == "__main__":
    train()
