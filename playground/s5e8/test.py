import warnings

from prefect import flow, task
import polars as pl
import xgboost as xgb
from prefect.assets import materialize

from common import Config, load_model, load_test_data, process_data

warnings.filterwarnings('ignore')


@materialize(
    f"file:///{Config.REPO_PATH}/outputs/s5e8/submission.csv",
    log_prints=True
)
def create_submission(ids: pl.Series, y_pred):
    output_path = f"{Config.REPO_PATH}/outputs/s5e8/submission.csv"

    submission = pl.DataFrame({
        "id": ids,
        "y": y_pred
    })

    submission.write_csv(output_path)
    print(f"✅ Submission file saved to {output_path}")

    return output_path


@task(log_prints=True)
def test_model(model: xgb.XGBClassifier, X_test: pl.DataFrame):
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    return y_pred, y_probs


@flow
def test():
    X_test: pl.DataFrame = load_test_data()
    ids, X_test = process_data(X_test, stage='test')

    model = load_model()
    y_pred, _ = test_model(model, X_test)

    create_submission(ids, y_pred)

    return y_pred


if __name__ == "__main__":
    test()
