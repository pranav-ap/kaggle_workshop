import xgboost as xgb
from common import (
    load_train_split_data,
    load_val_split_data,
    load_best_params,
    save_model,
)


def train_model(params):
    train_x, train_y = load_train_split_data()
    val_x, val_y = load_val_split_data()

    dtrain = xgb.DMatrix(train_x, label=train_y)
    dval = xgb.DMatrix(val_x, label=val_y)

    evals = [(dtrain, "train"), (dval, "val")]

    params.update({
        "verbosity": 1,
        "eval_metric": "rmse",
        "n_jobs": -1,
    })

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=5000,
        evals=evals,
        verbose_eval=100,
        early_stopping_rounds=100,
    )

    return model


def train():
    params = load_best_params()
    model = train_model(params)
    save_model(model)


if __name__ == "__main__":
    train()
