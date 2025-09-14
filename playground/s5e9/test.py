import xgboost as xgb
import polars as pl
from pathlib import Path
from common import load_model, load_test_data, Config


def main():
    X_test = load_test_data()
    ids = X_test["id"]
    X_test = X_test.drop("id").to_numpy()
    dtest = xgb.DMatrix(X_test)

    model = load_model()
    preds = model.predict(dtest)

    out_df = pl.DataFrame({
        "id": ids,
        "prediction": preds
    })

    output_path = Path(Config.REPO_PATH) / "outputs" / Config.PROJECT_NAME / "submission.csv"
    out_df.write_csv(output_path)

    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
