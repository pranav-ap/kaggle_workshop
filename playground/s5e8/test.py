from prefect import flow

from common import load_test_data, process_data

import warnings
warnings.filterwarnings('ignore')


@flow
def test() -> list[str]:
    data = load_test_data()
    data = process_data(data)

    results = []
    return results


if __name__ == "__main__":
    test()
